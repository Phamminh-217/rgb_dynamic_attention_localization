import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from models.architecture import FeatureAttentionHierarchicalNet
from scripts.inference import denormalize_coordinates, build_attention_map_live

def main():
    parser = argparse.ArgumentParser(description="Inference on custom image paths or directories")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--paths", type=str, nargs="+", required=True, 
                        help="One or more image paths, or directories containing images to test.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Model
    model = FeatureAttentionHierarchicalNet(
        num_classes_topo=int(config["model"]["num_classes_topo"]),
        pretrained=False,
    ).to(device)

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded weights from: {args.model_path}")

    # Setup image transforms
    image_size = config["data"]["image_size"]
    h, w = int(image_size[0]), int(image_size[1])
    mean = config["image_normalization"]["mean"]
    std = config["image_normalization"]["std"]

    image_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    attention_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    alpha = float(config.get("attention", {}).get("alpha", 0.7))
    norm_bounds = config["global_normalization"]
    x_min, x_max = norm_bounds["x_min"], norm_bounds["x_max"]
    y_min, y_max = norm_bounds["y_min"], norm_bounds["y_max"]

    # Gather all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    target_files = []

    for path_str in args.paths:
        p = Path(path_str)
        if p.is_dir():
            # Gather all images inside directory
            for f_path in p.iterdir():
                if f_path.suffix.lower() in image_extensions:
                    target_files.append(f_path)
        elif p.is_file():
            target_files.append(p)
        else:
            print(f"Warning: path '{path_str}' does not exist or is invalid. Skipping.")

    if not target_files:
        print("No valid images found to perform inference on.")
        return

    # Sort files for neat representation
    target_files = sorted(target_files)
    print(f"Found {len(target_files)} target image(s) for inference.")

    # Try to load Ground Truth matching poses if available in total_poses.csv to calculate error
    total_csv_path = Path(config["data"]["total_csv"])
    poses_df = None
    if total_csv_path.exists():
        try:
            poses_df = pd.read_csv(total_csv_path)
            # Normalize absolute path strings in CSV for matching
            poses_df["image_path_abs"] = poses_df["image_path"].apply(lambda x: str(Path(x).resolve()))
        except Exception as e:
            print(f"Note: Could not parse total_poses.csv for ground-truth matching ({e}). Predictions will proceed without error calculations.")

    print("\n" + "="*90)
    print("CUSTOM INFERENCE PIPELINE RUN")
    print("="*90)

    for img_path in target_files:
        abs_img_path = str(img_path.resolve())
        print(f"\nProcessing image: {img_path.name}")
        
        # Build attention map
        try:
            attention_map = build_attention_map_live(img_path, alpha)
        except Exception as e:
            print(f"  [Error] Failed to build attention map on-the-fly: {e}")
            continue

        # Load image & apply transforms
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = image_transform(image).unsqueeze(0).to(device)
            attention_tensor = attention_transform(attention_map).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  [Error] Failed to read image file: {e}")
            continue

        # Forward prediction
        with torch.no_grad():
            topo_logits, coord_pred = model(image_tensor, attention_tensor)

        # Decode Topological Pred
        topo_prob = torch.softmax(topo_logits, dim=1).squeeze(0).cpu().numpy()
        pred_class = int(np.argmax(topo_prob))
        pred_topo = "Corridor" if pred_class == 1 else "Room"
        confidence = topo_prob[pred_class] * 100

        # Decode Coordinates
        coord_normalized = coord_pred.squeeze(0).cpu().numpy()
        pred_x, pred_y = denormalize_coordinates(
            coord_normalized[0], coord_normalized[1],
            x_min, x_max, y_min, y_max
        )

        print(f"  - Predicted Area       : {pred_topo} ({confidence:.2f}% Confidence)")
        print(f"  - Predicted Coordinates: X = {pred_x:+.4f} meters, Y = {pred_y:+.4f} meters")

        # Match Ground Truth if database exists
        if poses_df is not None:
            match = poses_df[poses_df["image_path_abs"] == abs_img_path]
            if not match.empty:
                row = match.iloc[0]
                true_topo = "Room" if row["topo_label"] == 0 else "Corridor"
                true_x, true_y = float(row["x"]), float(row["y"])
                
                dist_err = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2) * 100
                print(f"  - [GT Match Found]:")
                print(f"    * True Area  : {true_topo}")
                print(f"    * True Coords: X = {true_x:+.4f}m, Y = {true_y:+.4f}m")
                print(f"    * Error      : Distance error = {dist_err:.2f} cm")
            else:
                # Fallback: check by filename match only
                match_by_name = poses_df[poses_df["image_path"].apply(lambda x: Path(x).name == img_path.name)]
                if not match_by_name.empty:
                    row = match_by_name.iloc[0]
                    true_topo = "Room" if row["topo_label"] == 0 else "Corridor"
                    true_x, true_y = float(row["x"]), float(row["y"])
                    dist_err = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2) * 100
                    print(f"  - [GT Match Found (by filename)]:")
                    print(f"    * True Area  : {true_topo}")
                    print(f"    * True Coords: X = {true_x:+.4f}m, Y = {true_y:+.4f}m")
                    print(f"    * Error      : Distance error = {dist_err:.2f} cm")

    print("\n" + "="*90)
    print("Inference Pipeline completed.")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
