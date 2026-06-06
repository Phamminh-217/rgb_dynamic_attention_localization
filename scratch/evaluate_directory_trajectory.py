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
    parser = argparse.ArgumentParser(description="Evaluate a full directory of images, plot predicted vs true trajectory")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images of the trajectory")
    parser.add_argument("--output_plot", type=str, default="results/trajectory_comparison.png", help="Path to save the output trajectory plot")
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

    # Ensure output plot parent directory exists
    plot_path = Path(args.output_plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

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
    images_dir = Path(args.images_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions])

    # Extract frame index for chronological sorting
    def extract_index(img_path):
        try:
            return int(img_path.stem.split("_")[-1])
        except ValueError:
            return 0

    image_files = sorted(image_files, key=extract_index)

    if not image_files:
        print(f"No images found in directory: {images_dir}")
        return

    print(f"Found {len(image_files)} images. Performing inference...")

    # Load total poses CSV for GT lookup if exists
    total_csv_path = Path(config["data"]["total_csv"])
    poses_df = None
    if total_csv_path.exists():
        try:
            poses_df = pd.read_csv(total_csv_path)
            poses_df["image_path_abs"] = poses_df["image_path"].apply(lambda x: str(Path(x).resolve()))
        except Exception as e:
            print(f"Note: Could not parse total_poses.csv ({e}). Proceeding without ground truth.")

    all_gt_coords = []
    all_pred_coords = []
    has_gt_list = []

    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_files):
            print(f"  Processed {idx + 1}/{len(image_files)} images...")

        # Build attention map
        try:
            attention_map = build_attention_map_live(img_path, alpha)
            image = Image.open(img_path).convert("RGB")
            image_tensor = image_transform(image).unsqueeze(0).to(device)
            attention_tensor = attention_transform(attention_map).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  [Error] Failed to process image {img_path.name}: {e}")
            continue

        # Predict
        with torch.no_grad():
            _, coord_pred = model(image_tensor, attention_tensor)

        # Decode Coords
        coord_normalized = coord_pred.squeeze(0).cpu().numpy()
        pred_x, pred_y = denormalize_coordinates(
            coord_normalized[0], coord_normalized[1],
            x_min, x_max, y_min, y_max
        )
        all_pred_coords.append([pred_x, pred_y])

        # Match Ground Truth
        matched = False
        if poses_df is not None:
            abs_img_path = str(img_path.resolve())
            match = poses_df[poses_df["image_path_abs"] == abs_img_path]
            if match.empty:
                match = poses_df[poses_df["image_path"].apply(lambda x: Path(x).name == img_path.name)]
            
            if not match.empty:
                row = match.iloc[0]
                true_x, true_y = float(row["x"]), float(row["y"])
                all_gt_coords.append([true_x, true_y])
                has_gt_list.append(True)
                matched = True

        if not matched:
            all_gt_coords.append([0.0, 0.0])
            has_gt_list.append(False)

    all_pred_coords = np.array(all_pred_coords)
    all_gt_coords = np.array(all_gt_coords)
    has_gt_list = np.array(has_gt_list)

    any_gt = np.any(has_gt_list)

    # Plot Trajectory
    plt.figure(figsize=(10, 8), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot predictions
    plt.plot(all_pred_coords[:, 0], all_pred_coords[:, 1], "r--", label="Predicted Trajectory", alpha=0.8, linewidth=1.5)
    plt.scatter(all_pred_coords[:, 0], all_pred_coords[:, 1], c="red", s=10, marker="x", alpha=0.7)

    # Plot GT if available
    if any_gt:
        gt_valid = all_gt_coords[has_gt_list]
        plt.plot(gt_valid[:, 0], gt_valid[:, 1], "g-", label="Ground Truth Trajectory", alpha=0.7, linewidth=2)
        plt.scatter(gt_valid[:, 0], gt_valid[:, 1], c="green", s=10, alpha=0.5)

        # Draw errors connecting lines
        step = max(1, len(all_pred_coords) // 100)
        for i in range(0, len(all_pred_coords), step):
            if has_gt_list[i]:
                plt.plot(
                    [all_gt_coords[i, 0], all_pred_coords[i, 0]],
                    [all_gt_coords[i, 1], all_pred_coords[i, 1]],
                    "k:", alpha=0.3, linewidth=0.8
                )

        # Metrics
        errors = np.sqrt(np.sum((gt_valid - all_pred_coords[has_gt_list]) ** 2, axis=1))
        mean_error = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        plt.title(
            f"Trajectory Comparison ({images_dir.name})\nMean Error: {mean_error*100:.2f} cm | RMSE: {rmse*100:.2f} cm",
            fontsize=12, fontweight="bold", pad=15
        )
    else:
        plt.title(f"Predicted Trajectory Only ({images_dir.name})", fontsize=12, fontweight="bold", pad=15)

    # Draw Start / End points
    plt.plot(all_pred_coords[0, 0], all_pred_coords[0, 1], "y*", markersize=16, markeredgecolor="black", label="START")
    plt.plot(all_pred_coords[-1, 0], all_pred_coords[-1, 1], "b*", markersize=16, markeredgecolor="black", label="END")

    plt.text(all_pred_coords[0, 0] + 0.03, all_pred_coords[0, 1] + 0.03, "START", fontsize=9, fontweight="bold", 
             color="black", bbox=dict(facecolor='yellow', alpha=0.8, boxstyle='round,pad=0.3'))
    plt.text(all_pred_coords[-1, 0] + 0.03, all_pred_coords[-1, 1] + 0.03, "END", fontsize=9, fontweight="bold", 
             color="white", bbox=dict(facecolor='blue', alpha=0.8, boxstyle='round,pad=0.3'))

    plt.xlabel("X Coordinate (meters)", fontsize=10)
    plt.ylabel("Y Coordinate (meters)", fontsize=10)
    plt.legend(frameon=True, facecolor="white", edgecolor="gray", loc="upper left")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nTrajectory evaluation plot successfully saved to: {plot_path.resolve()}")

if __name__ == "__main__":
    main()
