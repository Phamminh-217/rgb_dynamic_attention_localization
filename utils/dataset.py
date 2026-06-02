import json
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class RobotLocalizationDataset(Dataset):
    def __init__(self, total_csv: str, config_path: str) -> None:
        """Initialize RobotLocalizationDataset by loading CSV, config parameters, 

        and setting up transform pipelines.
        """
        self.total_csv = Path(total_csv)
        self.config_path = Path(config_path)

        # Load global configuration file
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Load poses CSV metadata
        self.records = pd.read_csv(self.total_csv)
        self._validate_columns()

        # Parse target resolution for images and attention maps
        image_size = self.config["data"]["image_size"]
        self.height = int(image_size[0])
        self.width = int(image_size[1])

        # Retrieve global normalization boundaries from config
        norm = self.config["global_normalization"]
        self.x_min = float(norm["x_min"])
        self.x_max = float(norm["x_max"])
        self.y_min = float(norm["y_min"])
        self.y_max = float(norm["y_max"])

        # Enforce range safety checks
        if self.x_max <= self.x_min:
            raise ValueError("Invalid x normalization range: x_max must be greater than x_min.")

        if self.y_max <= self.y_min:
            raise ValueError("Invalid y normalization range: y_max must be greater than y_min.")

        # Setup standard ImageNet normalization parameters for RGB backbone
        mean = self.config["image_normalization"]["mean"]
        std = self.config["image_normalization"]["std"]

        self.image_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Transforms for attention map (scaled into [0.0, 1.0])
        self.attention_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
        ])

    def _validate_columns(self) -> None:
        """Verify poses DataFrame contains the correct required metadata columns."""
        required_columns = {
            "image_path",
            "attention_path",
            "x",
            "y",
            "topo_label",
        }

        missing_columns = required_columns - set(self.records.columns)

        if missing_columns:
            raise ValueError(
                f"total_poses.csv is missing required columns: {sorted(missing_columns)}"
            )

    def __len__(self) -> int:
        """Return total size of the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch and prepare a single sample.

        Returns:
            image_tensor: Normalized RGB image [3, H, W]
            attention_tensor: Raw Attention Map [1, H, W] in [0, 1]
            topo_label: Topological label (scalar LongTensor)
            coord_label: Normalized global coordinate [2]
        """
        row = self.records.iloc[idx]

        # Retrieve paths
        image_path = Path(row["image_path"])
        attention_path = Path(row["attention_path"])

        # Load RGB image and grayscale Attention Map
        image = Image.open(image_path).convert("RGB")
        attention = Image.open(attention_path).convert("L")

        # Apply spatial resizing and normalization transformations
        image_tensor = self.image_transform(image)          # Shape: [3, H, W]
        attention_tensor = self.attention_transform(attention)  # Shape: [1, H, W]

        # Load physical coordinate pos_x and pos_y
        x = float(row["x"])
        y = float(row["y"])

        # Apply global Min-Max normalization to bounds [0.0, 1.0]
        x_n = (x - self.x_min) / (self.x_max - self.x_min)
        y_n = (y - self.y_min) / (self.y_max - self.y_min)

        # Build clean output tensors conforming to system contracts
        coord_label = torch.tensor([x_n, y_n], dtype=torch.float32)  # Shape: [2]
        topo_label = torch.tensor(int(row["topo_label"]), dtype=torch.long)

        return image_tensor, attention_tensor, topo_label, coord_label


if __name__ == "__main__":
    # Local verification block for testing Dataset integrity
    print("--- Running Local Dataset Verification ---")
    try:
        dataset = RobotLocalizationDataset(
            total_csv="data/processed/total_poses.csv",
            config_path="config.json"
        )
        print(f"Dataset loaded successfully with {len(dataset)} samples!")

        # Fetch first sample
        img_t, att_t, topo_l, coord_l = dataset[0]

        print("Single sample retrieval details:")
        print(f"  Image Tensor Shape    : {list(img_t.shape)} (Expected: [3, 224, 224])")
        print(f"  Attention Tensor Shape: {list(att_t.shape)} (Expected: [1, 224, 224])")
        print(f"  Topological Dtype     : {topo_l.dtype} (Expected: torch.int64)")
        print(f"  Coordinate Shape      : {list(coord_l.shape)} (Expected: [2])")
        print(f"  Coordinate Range      : X=[{coord_l[0]:.4f}], Y=[{coord_l[1]:.4f}] in [0, 1]")

        # Perform strict shape and value assertions
        assert img_t.shape == (3, 224, 224), "Image shape mismatch!"
        assert att_t.shape == (1, 224, 224), "Attention shape mismatch!"
        assert topo_l.dtype == torch.long, "Topological dtype mismatch!"
        assert coord_l.shape == (2,), "Coordinate shape mismatch!"
        assert att_t.min() >= 0.0 and att_t.max() <= 1.0, "Attention map value out of [0, 1] bounds!"
        assert coord_l.min() >= 0.0 and coord_l.max() <= 1.0, "Normalized coordinates out of [0, 1] bounds!"

        print("\nAll Dataset shape and value validation assertions passed successfully!")
    except Exception as e:
        print(f"Verification encountered error: {e}")
