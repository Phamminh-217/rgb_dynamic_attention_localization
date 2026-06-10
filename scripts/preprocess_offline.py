import argparse
import json
import os
from pathlib import Path
from PIL import Image
import math
import numpy as np
import pandas as pd

# Import YOLO dynamically to ensure clean failure if ultralytics is not fully installed
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def load_json(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(config: dict, config_path: str) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def generate_attention_maps(config: dict) -> None:
    """Generate grayscale dynamic-aware Attention Maps using YOLOv8n offline.

    Detects people (class 0) and damps their feature representation weight to 1 - alpha.
    """
    alpha = float(config.get("attention", {}).get("alpha", 0.7))
    damp_factor = 1.0 - alpha
    damp_value = int(255 * damp_factor)

    # Output directory for processed attention maps
    attention_maps_dir = Path(config["data"]["attention_maps_dir"])
    attention_maps_dir.mkdir(parents=True, exist_ok=True)

    # Define paths to raw image directories dynamically from config
    raw_root = Path(config["data"]["raw_root"])
    
    # We will gather all image files in all subdirectories of raw_root
    image_files = []
    for area_dir in sorted(raw_root.iterdir()):
        if area_dir.is_dir():
            for run_dir in sorted(area_dir.iterdir()):
                if run_dir.is_dir():
                    run_imgs = sorted([
                        f for f in run_dir.iterdir()
                        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
                    ])
                    image_files.extend(run_imgs)

    if not image_files:
        print(f"No image files found in {raw_root}. Skipping attention maps generation.")
        return

    # Filter out images that already have generated attention maps
    images_to_process = []
    for img_path in image_files:
        out_path = attention_maps_dir / img_path.name
        if not out_path.exists():
            images_to_process.append(img_path)

    if not images_to_process:
        print(f"All attention maps already exist. Skipping YOLOv8 inference.")
        return

    yolo_model = None

    if yolo_model is None:
        if YOLO is None:
            raise ImportError(
                "The 'ultralytics' library is required to run YOLOv8 offline preprocessing. "
                "Please run 'pip install ultralytics' first."
            )
        print("Loading pre-trained YOLOv8n model...")
        yolo_model = YOLO("yolov8n.pt")

    for idx, img_path in enumerate(images_to_process):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Create grayscale mask filled with 255 (representing static weight 1.0)
                mask_array = np.full((height, width), 255, dtype=np.uint8)

                # Run YOLOv8 offline inference on the frame
                results = yolo_model(img_path, verbose=False)
                
                # Inspect detection boxes
                for box in results[0].boxes:
                    class_id = int(box.cls[0].item())
                    
                    # Class 0 corresponds to 'person' in the COCO dataset
                    if class_id == 0:
                        # Extract pixel bounds of the detected person
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Clamp bounding boxes to image dimensions to prevent out-of-bound errors
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        # Dampen the feature representation weight in the detected bounding box
                        mask_array[y1:y2, x1:x2] = damp_value

                # Convert the modified numpy array back to PIL Image and save
                attention_map_img = Image.fromarray(mask_array, mode="L")
                
                # Save with same filename to output attention maps directory
                out_path = attention_maps_dir / img_path.name
                attention_map_img.save(out_path, format="PNG")

            if (idx + 1) % 100 == 0 or (idx + 1) == len(images_to_process):
                print(f"  Processed {idx + 1}/{len(images_to_process)} images...")
                
        except Exception as e:
            print(f"Error processing image {img_path.name}: {e}")

    print(f"Attention maps check completed for: {attention_maps_dir}")


def align_and_aggregate_poses(config: dict, config_path: str) -> None:
    """Align coordinate systems using Encoder Ground Truth odometry, compute global bounds, and save unified CSV dataset."""
    print("\n--- Running Encoder Odometry Ground Truth Mapping and Dataset Aggregation ---")

    # Read configuration parameters
    raw_root = Path(config["data"]["raw_root"])
    total_csv_path = Path(config["data"]["total_csv"])
    attention_maps_dir = Path(config["data"]["attention_maps_dir"])

    # Load merged encoder odometry data
    odometry_csv_path = Path("ground_truth_encoder/gt_hop_nhat/merged_odometry.csv")
    if not odometry_csv_path.exists():
        raise FileNotFoundError(f"Merged odometry file not found at: {odometry_csv_path}")

    print(f"Loading merged encoder odometry from: {odometry_csv_path}")
    odom_df = pd.read_csv(odometry_csv_path)

    datasets = []

    # Iterate over A-E areas configured in areas
    for area_id, area_cfg in config["areas"].items():
        topo_label = int(area_cfg["label"])
        print(f"\nProcessing Area {area_id} (label: {topo_label}): {area_cfg['description']}")

        for run in area_cfg["runs"]:
            dir_name = run["dir_name"]
            encoder_lan = int(run["encoder_lan"])
            run_dir = raw_root / area_id / dir_name

            if not run_dir.exists():
                print(f"  Warning: Image directory {run_dir} does not exist. Skipping.")
                continue

            # Load the matching subset of encoder odometry for this run_lan
            run_odom = odom_df[odom_df["lan"] == encoder_lan].reset_index(drop=True)
            if run_odom.empty:
                print(f"  Warning: No odometry data found for lan {encoder_lan}. Skipping.")
                continue

            # Get list of images in raw folder
            image_files = sorted(
                [f for f in run_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            )

            if not image_files:
                print(f"  Warning: No images found in {run_dir}. Skipping.")
                continue

            num_images = len(image_files)
            num_odom = len(run_odom)
            print(f"  Run '{dir_name}': {num_images} images, {num_odom} odometry entries. Interpolating...")

            # Linear interpolation of (x, y) poses based on image index mapping to odometry index
            rows = []
            for img_idx, img_file in enumerate(image_files):
                # Map image index linearily to odometry index range [0, num_odom - 1]
                if num_images > 1:
                    odom_pos = img_idx * (num_odom - 1) / (num_images - 1)
                else:
                    odom_pos = 0.0

                odom_idx_low = int(math.floor(odom_pos))
                odom_idx_high = min(odom_idx_low + 1, num_odom - 1)
                weight = odom_pos - odom_idx_low

                # Retrieve low and high values
                x_low = float(run_odom.loc[odom_idx_low, "x"])
                x_high = float(run_odom.loc[odom_idx_high, "x"])
                y_low = float(run_odom.loc[odom_idx_low, "y"])
                y_high = float(run_odom.loc[odom_idx_high, "y"])

                # Linearly interpolate
                global_x = x_low + weight * (x_high - x_low)
                global_y = y_low + weight * (y_high - y_low)

                # Relative paths
                rel_image_path = f"data/raw/{area_id}/{dir_name}/{img_file.name}"
                rel_attention_path = f"data/processed/attention_maps/{img_file.name}"

                rows.append({
                    "image_path": rel_image_path,
                    "attention_path": rel_attention_path,
                    "x": global_x,
                    "y": global_y,
                    "topo_label": topo_label
                })

            run_df = pd.DataFrame(rows)
            datasets.append(run_df)
            print(f"    Mapped {len(run_df)} frames.")

    if not datasets:
        raise RuntimeError("No datasets were successfully processed.")

    # Concatenate all datasets
    total_df = pd.concat(datasets, ignore_index=True)

    # Compute global Min-Max bounds on integrated global coordinates
    x_min = float(total_df["x"].min())
    x_max = float(total_df["x"].max())
    y_min = float(total_df["y"].min())
    y_max = float(total_df["y"].max())

    if x_max <= x_min:
        raise ValueError("Invalid calculated x range: x_max must be greater than x_min.")
    if y_max <= y_min:
        raise ValueError("Invalid calculated y range: y_max must be greater than y_min.")

    print(f"\nComputed Global Normalization Bounds from Encoder GT:")
    print(f"  X Bound: [{x_min:.4f}, {x_max:.4f}]")
    print(f"  Y Bound: [{y_min:.4f}, {y_max:.4f}]")

    # Update config.json with calculated global bounds
    config["global_normalization"]["x_min"] = x_min
    config["global_normalization"]["x_max"] = x_max
    config["global_normalization"]["y_min"] = y_min
    config["global_normalization"]["y_max"] = y_max

    save_json(config, config_path)
    print(f"Successfully updated 'global_normalization' bounds in: {config_path}")

    # Save aggregated dataframe index-free
    total_csv_path.parent.mkdir(parents=True, exist_ok=True)
    total_df.to_csv(total_csv_path, index=False)
    print(f"Successfully saved aggregated poses dataset to: {total_csv_path}")


def is_preprocessed_ready(config: dict) -> bool:
    """Validate if the offline preprocessing state is complete and correct."""
    total_csv = Path(config["data"]["total_csv"])
    attention_dir = Path(config["data"]["attention_maps_dir"])
    
    total_csv_exists = total_csv.exists()
    attention_dir_exists = attention_dir.exists()
    attention_dir_not_empty = attention_dir_exists and any(attention_dir.iterdir())
    
    # Check if bounds are already populated
    norm = config["global_normalization"]
    bounds_ready = (
        norm["x_min"] is not None and
        norm["x_max"] is not None and
        norm["y_min"] is not None and
        norm["y_max"] is not None
    )
    
    return total_csv_exists and attention_dir_not_empty and bounds_ready


def preprocess_offline(config_path: str) -> None:
    """Run full offline preprocessing pipeline."""
    config = load_json(config_path)
    generate_attention_maps(config)
    align_and_aggregate_poses(config, config_path)


def main():
    parser = argparse.ArgumentParser(description="Offline Preprocessing - YOLOv8 & Dataset Alignment")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json file")
    args = parser.parse_args()

    config = load_json(args.config)
    
    # Step 1: Run YOLOv8 offline to generate attention maps
    generate_attention_maps(config)

    # Step 2: Sync coordinate frames, calculate normalization parameters, and save aggregated dataset
    align_and_aggregate_poses(config, args.config)


if __name__ == "__main__":
    main()
