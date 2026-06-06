import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.architecture import FeatureAttentionHierarchicalNet
from scripts.inference import denormalize_coordinates, build_attention_map_live

def main():
    parser = argparse.ArgumentParser(description="Inference on custom image paths or directories and visual report generation")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--paths", type=str, nargs="+", required=True, 
                        help="One or more image paths, or directories containing images to test.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save visual report images")
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

    # Setup output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    target_files = sorted(target_files)
    print(f"Found {len(target_files)} target image(s) for inference.")

    # Try to load Ground Truth matching poses if available in total_poses.csv to calculate error
    total_csv_path = Path(config["data"]["total_csv"])
    poses_df = None
    if total_csv_path.exists():
        try:
            poses_df = pd.read_csv(total_csv_path)
            poses_df["image_path_abs"] = poses_df["image_path"].apply(lambda x: str(Path(x).resolve()))
        except Exception as e:
            print(f"Note: Could not parse total_poses.csv ({e}).")

    print("\n" + "="*90)
    print("CUSTOM VISUAL INFERENCE PIPELINE RUN")
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

        # Find Ground Truth
        true_topo = "Unknown"
        true_x, true_y = 0.0, 0.0
        has_gt = False
        dist_err_cm = 0.0

        if poses_df is not None:
            match = poses_df[poses_df["image_path_abs"] == abs_img_path]
            if match.empty:
                match = poses_df[poses_df["image_path"].apply(lambda x: Path(x).name == img_path.name)]
            
            if not match.empty:
                row = match.iloc[0]
                true_topo = "Room" if row["topo_label"] == 0 else "Corridor"
                true_x, true_y = float(row["x"]), float(row["y"])
                dist_err_cm = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2) * 100
                has_gt = True
                print(f"  - [GT Match Found]: True Area = {true_topo} | Distance error = {dist_err_cm:.2f} cm")

        # GENERATE VISUALIZATION PLOT (3 Panels)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        plt.subplots_adjust(top=0.85)

        # Title
        topo_match_text = f"{true_topo} == {pred_topo}" if has_gt else f"Predicted: {pred_topo}"
        error_text = f"Distance Error: {dist_err_cm:.2f} cm" if has_gt else "Ground Truth: N/A"
        fig.suptitle(
            f"Visual Localization Report - {img_path.name}\nArea Accuracy: {topo_match_text} ({confidence:.1f}% Conf) | {error_text}",
            fontsize=12, fontweight="bold"
        )

        # Panel 1: Original Image
        axes[0].imshow(image)
        axes[0].set_title(f"Original Image: {img_path.name}", fontsize=10)
        axes[0].axis("off")

        # Panel 2: Attention Map
        axes[1].imshow(attention_map, cmap="gray")
        axes[1].set_title("Generated Attention Map (YOLOv8)", fontsize=10)
        axes[1].axis("off")

        # Panel 3: Position comparison plot
        axes[2].set_title(f"Position Compare ({true_topo} -> {pred_topo})" if has_gt else "Predicted Position", fontsize=10)
        
        # Determine plot limits based on coordinates
        coords_to_plot = [[pred_x, pred_y]]
        if has_gt:
            coords_to_plot.append([true_x, true_y])
        
        coords_to_plot = np.array(coords_to_plot)
        center_x, center_y = np.mean(coords_to_plot[:, 0]), np.mean(coords_to_plot[:, 1])
        plot_range = 0.5  # +- 0.5 meters padding
        
        axes[2].set_xlim(center_x - plot_range, center_x + plot_range)
        axes[2].set_ylim(center_y - plot_range, center_y + plot_range)
        
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].set_xlabel("X Coordinate (m)", fontsize=9)
        axes[2].set_ylabel("Y Coordinate (m)", fontsize=9)
        axes[2].set_aspect("equal")

        # Plot GT if exists
        if has_gt:
            axes[2].scatter(true_x, true_y, color="green", s=150, marker="o", label=f"GT ({true_x:+.2f}, {true_y:+.2f})")
            
        # Plot Prediction
        axes[2].scatter(pred_x, pred_y, color="red", s=150, marker="x", linewidths=3, label=f"Pred ({pred_x:+.2f}, {pred_y:+.2f})")

        # Line connecting them if GT exists
        if has_gt:
            axes[2].plot([true_x, pred_x], [true_y, pred_y], "k--", alpha=0.6, label=f"Error: {dist_err_cm:.2f} cm")

        axes[2].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray", fontsize=8)

        # Save plot
        output_report_path = out_dir / f"report_{img_path.stem}.png"
        plt.savefig(output_report_path, bbox_inches="tight")
        plt.close()
        print(f"  - Successfully generated and saved report image: {output_report_path}")

    print("\n" + "="*90)
    print(f"Visual inference completed. All reports saved to: {out_dir.resolve()}")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
