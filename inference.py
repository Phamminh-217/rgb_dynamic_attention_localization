import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.architecture import FeatureAttentionHierarchicalNet
from torch.utils.data import DataLoader, random_split
from utils.dataset import RobotLocalizationDataset

# Try importing YOLO to build attention maps on-the-fly for single images
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_attention_map_live(image_path: Path, alpha: float) -> Image.Image:
    """Run YOLOv8 on-the-fly to detect persons and generate an offline-conforming attention map."""
    if YOLO is None:
        raise ImportError(
            "The 'ultralytics' library is required for live YOLOv8 inference. "
            "Please run 'pip install ultralytics' first."
        )

    # Load YOLOv8 model (using standard yolov8n.pt weight file)
    yolo_model = YOLO("yolov8n.pt")
    
    with Image.open(image_path) as img:
        width, height = img.size
        # Start with static weight 1.0 (255 in grayscale)
        mask_array = np.full((height, width), 255, dtype=np.uint8)
        
        results = yolo_model(image_path, verbose=False)
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            if class_id == 0:  # COCO Class 0 = Person
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Clamp coordinates to boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                # Dampen weight of dynamic region to 1 - alpha
                damp_value = int(255 * (1.0 - alpha))
                mask_array[y1:y2, x1:x2] = damp_value
                
        return Image.fromarray(mask_array, mode="L")


def denormalize_coordinates(
    x_n: float, y_n: float, x_min: float, x_max: float, y_min: float, y_max: float
) -> Tuple[float, float]:
    """Convert normalized [0, 1] coordinates back to physical meters."""
    x = x_n * (x_max - x_min) + x_min
    y = y_n * (y_max - y_min) + y_min
    return x, y


def run_single_inference(
    image_path: str,
    model: torch.nn.Module,
    config: dict,
    device: torch.device
) -> Tuple[str, Tuple[float, float]]:
    """Perform localization inference on a single raw RGB image."""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Target image not found at: {img_path}")

    # Generate Attention Map on-the-fly
    alpha = float(config.get("attention", {}).get("alpha", 0.7))
    print("[Live] Running YOLOv8 on-the-fly human detector...")
    attention_map = build_attention_map_live(img_path, alpha)

    # Image transformations matching dataset setup
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

    # Convert to tensors and add batch dimension [1, C, H, W]
    image = Image.open(img_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    attention_tensor = attention_transform(attention_map).unsqueeze(0).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        topo_logits, coord_pred = model(image_tensor, attention_tensor)

    # Parse Topological Prediction
    topo_prob = torch.softmax(topo_logits, dim=1).squeeze(0).cpu().numpy()
    topo_class = int(np.argmax(topo_prob))
    topo_label = "Corridor" if topo_class == 1 else "Room"
    confidence = topo_prob[topo_class] * 100

    # Parse and Denormalize Coordinates
    coord_normalized = coord_pred.squeeze(0).cpu().numpy()
    norm_bounds = config["global_normalization"]
    x_min, x_max = norm_bounds["x_min"], norm_bounds["x_max"]
    y_min, y_max = norm_bounds["y_min"], norm_bounds["y_max"]

    pred_x, pred_y = denormalize_coordinates(
        coord_normalized[0], coord_normalized[1],
        x_min, x_max, y_min, y_max
    )

    return topo_label, (pred_x, pred_y), confidence


def evaluate_trajectory(
    model: torch.nn.Module,
    config: dict,
    config_path: str,
    device: torch.device
) -> None:
    """Evaluate full test set split trajectory and generate comparison plots."""
    print("\n--- Starting Full Test Set Trajectory Evaluation ---")

    # Load dataset
    dataset = RobotLocalizationDataset(
        total_csv=config["data"]["total_csv"],
        config_path=config_path
    )

    # Recreate the exact split from training
    train_ratio = float(config["data"]["train_ratio"])
    val_ratio = float(config["data"]["val_ratio"])
    
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    _, _, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config["project"]["seed"])
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=False
    )

    print(f"Loaded {len(test_dataset)} unseen samples for trajectory comparison.")

    # Global bounds for denormalization
    norm_bounds = config["global_normalization"]
    x_min, x_max = norm_bounds["x_min"], norm_bounds["x_max"]
    y_min, y_max = norm_bounds["y_min"], norm_bounds["y_max"]

    all_gt_coords = []
    all_pred_coords = []
    topo_correct = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, attention_maps, topo_labels, coord_labels = batch
            images = images.to(device)
            attention_maps = attention_maps.to(device)

            # Model prediction
            topo_logits, coord_preds = model(images, attention_maps)

            # Evaluate classification
            preds_class = torch.argmax(topo_logits, dim=1).cpu().numpy()
            topo_correct += np.sum(preds_class == topo_labels.numpy())
            total_samples += len(topo_labels)

            # Process Coordinates
            for gt_norm, pred_norm in zip(coord_labels.numpy(), coord_preds.cpu().numpy()):
                # Denormalize to physical meters
                gt_x, gt_y = denormalize_coordinates(gt_norm[0], gt_norm[1], x_min, x_max, y_min, y_max)
                pred_x, pred_y = denormalize_coordinates(pred_norm[0], pred_norm[1], x_min, x_max, y_min, y_max)
                
                all_gt_coords.append([gt_x, gt_y])
                all_pred_coords.append([pred_x, pred_y])

    all_gt_coords = np.array(all_gt_coords)
    all_pred_coords = np.array(all_pred_coords)

    # Calculate metrics in meters
    errors = np.sqrt(np.sum((all_gt_coords - all_pred_coords) ** 2, axis=1)) # Euclidean distances
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # Coordinates-wise MAE
    mae_x = np.mean(np.abs(all_gt_coords[:, 0] - all_pred_coords[:, 0]))
    mae_y = np.mean(np.abs(all_gt_coords[:, 1] - all_pred_coords[:, 1]))

    topo_acc = (topo_correct / total_samples) * 100

    print(f"\n==================================================")
    print(f"TRAJECTORY EVALUATION METRICS (Test Set):")
    print(f"  Topological Area Accuracy : {topo_acc:.2f}% ({topo_correct}/{total_samples})")
    print(f"  Mean Localization Error  : {mean_error:.4f} meters ({mean_error*100:.2f} cm)")
    print(f"  Maximum Error             : {max_error:.4f} meters ({max_error*100:.2f} cm)")
    print(f"  Root Mean Squared (RMSE)  : {rmse:.4f} meters ({rmse*100:.2f} cm)")
    print(f"  Mean Absolute Error (MAE):")
    print(f"    - X axis: {mae_x:.4f} m ({mae_x*100:.2f} cm)")
    print(f"    - Y axis: {mae_y:.4f} m ({mae_y*100:.2f} cm)")
    print(f"==================================================")

    # Create beautiful comparison plot
    plt.figure(figsize=(10, 8), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Plot ground truth and prediction
    plt.plot(all_gt_coords[:, 0], all_gt_coords[:, 1], "g-", label="Ground Truth Trajectory", alpha=0.7, linewidth=2)
    plt.scatter(all_gt_coords[:, 0], all_gt_coords[:, 1], c="green", s=15, alpha=0.6)
    
    plt.plot(all_pred_coords[:, 0], all_pred_coords[:, 1], "r--", label="Predicted Trajectory", alpha=0.8, linewidth=1.5)
    plt.scatter(all_pred_coords[:, 0], all_pred_coords[:, 1], c="red", s=15, marker="x", alpha=0.8)

    # Draw error lines between matching points (limit lines for visibility if too many points)
    step = max(1, len(all_gt_coords) // 100)
    for idx in range(0, len(all_gt_coords), step):
        plt.plot(
            [all_gt_coords[idx, 0], all_pred_coords[idx, 0]],
            [all_gt_coords[idx, 1], all_pred_coords[idx, 1]],
            "k:", alpha=0.3, linewidth=0.8
        )

    plt.title(
        f"Robot Trajectory Comparison (Test Set)\nMean Localization Error: {mean_error*100:.2f} cm | RMSE: {rmse*100:.2f} cm",
        fontsize=12, fontweight="bold", pad=15
    )
    plt.xlabel("X Coordinate (meters)", fontsize=10)
    plt.ylabel("Y Coordinate (meters)", fontsize=10)
    plt.legend(frameon=True, facecolor="white", edgecolor="gray")
    plt.axis("equal")
    plt.tight_layout()

    # Save visualization to checkpoints directory
    output_plot_path = Path(config["checkpoint"]["local_dir"]) / "trajectory_comparison.png"
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"\nSuccessfully generated and saved comparison trajectory plot to: {output_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Robot Localization Model Inference Utility")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json file")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to best_model.pth file")
    parser.add_argument("--image_path", type=str, default=None, help="Evaluate a single image (e.g. data/raw/room/images/frame_000000.png)")
    parser.add_argument("--trajectory", action="store_true", help="Compare full test trajectory and generate plot")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on hardware: {device}")

    # Load model
    model = FeatureAttentionHierarchicalNet(
        num_classes_topo=int(config["model"]["num_classes_topo"]),
        pretrained=False,
    ).to(device)

    # Load weights
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model checkpoint weights not found at: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model weights from: {args.model_path}")

    # Mode 1: Evaluate a single image
    if args.image_path is not None:
        try:
            topo, (x, y), conf = run_single_inference(args.image_path, model, config, device)
            print(f"\n==========================================")
            print(f"SINGLE IMAGE INFERENCE RESULT:")
            print(f"  Target Image : {args.image_path}")
            print(f"  Area Label   : {topo} ({conf:.2f}% Confidence)")
            print(f"  Coordinates  : X = {x:.4f} meters, Y = {y:.4f} meters")
            print(f"==========================================\n")
        except Exception as e:
            print(f"Single image inference failed: {e}")

    # Mode 2: Compare full test trajectory and generate comparison plot
    if args.trajectory or (args.image_path is None):
        evaluate_trajectory(model, config, args.config, device)


if __name__ == "__main__":
    main()
