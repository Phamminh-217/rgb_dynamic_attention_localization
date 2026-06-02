import argparse
import json
import os
from pathlib import Path
from PIL import Image
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

    # Define paths to raw image directories
    raw_dirs = {
        "room": Path(config["data"]["room_csv"]).parent / "images",
        "corridor": Path(config["data"]["corridor_csv"]).parent / "images"
    }

    # Initialize YOLOv8 model only if there are maps to be generated
    yolo_model = None

    for section_name, raw_dir in raw_dirs.items():
        if not raw_dir.exists():
            print(f"Directory {raw_dir} does not exist. Skipping section '{section_name}'.")
            continue

        # Get list of all image frames in raw folder
        image_files = sorted(
            [f for f in raw_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )

        if not image_files:
            print(f"No image files found in {raw_dir}. Skipping section '{section_name}'.")
            continue

        # Filter out images that already have generated attention maps
        images_to_process = []
        for img_path in image_files:
            out_path = attention_maps_dir / img_path.name
            if not out_path.exists():
                images_to_process.append(img_path)

        if not images_to_process:
            print(f"All attention maps for section '{section_name}' already exist. Skipping YOLOv8 inference.")
            continue

        print(f"\nProcessing {len(images_to_process)}/{len(image_files)} images for section '{section_name}'...")
        
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
    """Align coordinate systems, compute global bounds, and save unified CSV dataset."""
    print("\n--- Running Coordinate Alignment and Dataset Aggregation ---")

    # Read configuration parameters
    room_csv_path = Path(config["data"]["room_csv"])
    corridor_csv_path = Path(config["data"]["corridor_csv"])
    total_csv_path = Path(config["data"]["total_csv"])
    attention_maps_dir = Path(config["data"]["attention_maps_dir"])

    labels = config["labels"]
    offsets = config["coordinate_alignment"]["corridor"]
    delta_x = float(offsets["delta_x"])
    delta_y = float(offsets["delta_y"])

    datasets = []

    # Map sections to their configuration
    sections = [
        {"name": "room", "csv": room_csv_path, "label": int(labels["room"]), "offset_x": 0.0, "offset_y": 0.0},
        {"name": "corridor", "csv": corridor_csv_path, "label": int(labels["corridor"]), "offset_x": delta_x, "offset_y": delta_y}
    ]

    for sec in sections:
        name = sec["name"]
        csv_path = sec["csv"]
        topo_label = sec["label"]
        offset_x = sec["offset_x"]
        offset_y = sec["offset_y"]

        if not csv_path.exists():
            print(f"Poses CSV file not found: {csv_path}. Skipping section '{name}'.")
            continue

        # Load raw poses CSV using Pandas
        print(f"Loading {name} poses from: {csv_path}")
        pose_df = pd.read_csv(csv_path)

        # Ensure poses CSV contains expected pos_x and pos_y columns
        if "pos_x" not in pose_df.columns or "pos_y" not in pose_df.columns:
            raise ValueError(f"CSV file {csv_path} must contain 'pos_x' and 'pos_y' columns.")

        # Find raw images directory for indexing mapping
        raw_images_dir = csv_path.parent / "images"
        if not raw_images_dir.exists():
            print(f"Images folder {raw_images_dir} does not exist. Skipping section '{name}'.")
            continue

        image_files = sorted(
            [f for f in raw_images_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )

        rows = []
        for img_file in image_files:
            # Extract index from filename suffix (e.g. prefix_000123.png -> index 123)
            try:
                idx = int(img_file.stem.split("_")[-1])
            except ValueError:
                print(f"Warning: Could not parse index suffix from {img_file.name}. Skipping file.")
                continue

            # Ensure index exists in the poses dataframe rows
            if idx < 0 or idx >= len(pose_df):
                print(f"Warning: Index {idx} out of range for CSV size {len(pose_df)}. Skipping file.")
                continue

            # Retrieve raw coordinates
            raw_x = float(pose_df.loc[idx, "pos_x"])
            raw_y = float(pose_df.loc[idx, "pos_y"])

            # Apply coordinate translation offset
            global_x = raw_x + offset_x
            global_y = raw_y + offset_y

            # Establish relative path contracts for DataLoader compatibility
            rel_image_path = f"data/raw/{name}/images/{img_file.name}"
            rel_attention_path = f"data/processed/attention_maps/{img_file.name}"

            rows.append({
                "image_path": rel_image_path,
                "attention_path": rel_attention_path,
                "x": global_x,
                "y": global_y,
                "topo_label": topo_label
            })

        sec_df = pd.DataFrame(rows)
        print(f"  Successfully mapped {len(sec_df)} frames for section '{name}'.")
        datasets.append(sec_df)

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

    print(f"\nComputed Global Normalization Bounds:")
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
