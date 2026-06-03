import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils.dataset import RobotLocalizationDataset
from models.architecture import FeatureAttentionHierarchicalNet
from scripts.preprocess_offline import is_preprocessed_ready, preprocess_offline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training coordinator."""
    parser = argparse.ArgumentParser(description="Master Training Controller")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to configuration config.json file"
    )
    parser.add_argument(
        "--colab", 
        action="store_true", 
        help="Enable automated Google Colab runtime storage optimization"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None, 
        help="Path to checkpoint .pth file to resume training"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="Override device selection (e.g. 'cuda' or 'cpu')"
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    """Safely load JSON file contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    """Enforce deterministic seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(config: dict, override_device: str = None) -> torch.device:
    """Determine best available hardware accelerator."""
    if override_device is not None:
        return torch.device(override_device)

    requested_device = config["project"].get("device", "cuda")
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_colab_dataset(config: dict) -> None:
    """Decompress dataset zip on high-performance local runtime SSD in Colab environment."""
    drive_zip_path = Path(config["colab"]["drive_zip_path"])
    local_dataset_dir = Path(config["colab"]["local_dataset_dir"])

    if not local_dataset_dir.exists() or not any(local_dataset_dir.iterdir()):
        if not drive_zip_path.exists():
            raise FileNotFoundError(f"Google Drive data.zip archive not found at: {drive_zip_path}")

        local_dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Colab] Unpacking dataset from Drive: {drive_zip_path} -> Local: {local_dataset_dir}...")
        
        with zipfile.ZipFile(drive_zip_path, "r") as zip_ref:
            zip_ref.extractall(local_dataset_dir)
        print("[Colab] Dataset extraction completed successfully.")
    else:
        print("[Colab] Dataset already exists on local SSD runtime. Skipping extraction.")

    # Dynamically create symbolic link to local './data' folder to sync relative paths
    local_data_link = Path("data")
    
    # Remove existing link or directory if it is a symlink
    if local_data_link.is_symlink():
        local_data_link.unlink()
    elif local_data_link.exists() and not any(local_data_link.iterdir()):
        # If it's an empty directory, we can safely remove it to replace with symlink
        local_data_link.rmdir()

    # Determine source directory (handle both zipping data/ folder directly or its contents)
    if (local_dataset_dir / "data").exists():
        src_dir = local_dataset_dir / "data"
    else:
        src_dir = local_dataset_dir

    import os
    try:
        os.symlink(src_dir.absolute(), local_data_link.absolute())
        print(f"[Colab] Successfully linked relative paths: {local_data_link} -> {src_dir}")
    except FileExistsError:
        print(f"[Colab] Local '{local_data_link}' already exists. Please ensure it points to the correct dataset.")
    except Exception as e:
        print(f"[Colab] Warning: Could not create symbolic link: {e}")


def ensure_preprocessed_data(config: dict, config_path: str) -> dict:
    """Ensure offline preprocessed assets exist before initiating dataset loader."""
    if is_preprocessed_ready(config):
        print("[Data] Preprocessed pose CSV, attention maps, and bounds are ready.")
        return config

    print("[Data] Missing preprocessed assets. Running preprocess_offline.py automatically...")
    preprocess_offline(config_path)

    updated_config = load_json(config_path)
    if not is_preprocessed_ready(updated_config):
        raise RuntimeError(
            "Preprocessing automation failed: total_poses.csv or attention maps are still missing."
        )
    return updated_config


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_topo: nn.Module,
    criterion_coord: nn.Module,
    gamma: float,
    device: torch.device
) -> Dict[str, float]:
    """Execute training logic for a single epoch across all batches."""
    model.train()

    total_loss_sum = 0.0
    topo_loss_sum = 0.0
    coord_loss_sum = 0.0

    for batch in train_loader:
        # Unpack tensors conforming to dataset shape contracts
        images, attention_maps, topo_labels, coord_labels = batch

        # Push tensors to hardware device
        images = images.to(device)                      # Shape: [B, 3, H, W]
        attention_maps = attention_maps.to(device)      # Shape: [B, 1, H, W]
        topo_labels = topo_labels.to(device)            # Shape: [B]
        coord_labels = coord_labels.to(device)          # Shape: [B, 2]

        # Forward pass
        topo_logits, coord_pred = model(images, attention_maps)

        # Compute multi-task losses
        loss_topo = criterion_topo(topo_logits, topo_labels)
        loss_coord = criterion_coord(coord_pred, coord_labels)
        loss_total = loss_topo + gamma * loss_coord

        # Backpropagation and parameters update
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # Aggregate statistics
        total_loss_sum += loss_total.item()
        topo_loss_sum += loss_topo.item()
        coord_loss_sum += loss_coord.item()

    num_batches = len(train_loader)
    return {
        "train_total_loss": total_loss_sum / num_batches,
        "train_topo_loss": topo_loss_sum / num_batches,
        "train_coord_loss": coord_loss_sum / num_batches,
    }


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion_topo: nn.Module,
    criterion_coord: nn.Module,
    gamma: float,
    device: torch.device
) -> Dict[str, float]:
    """Execute validation logic for a single epoch across all batches."""
    model.eval()

    total_loss_sum = 0.0
    topo_loss_sum = 0.0
    coord_loss_sum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images, attention_maps, topo_labels, coord_labels = batch

            images = images.to(device)
            attention_maps = attention_maps.to(device)
            topo_labels = topo_labels.to(device)
            coord_labels = coord_labels.to(device)

            topo_logits, coord_pred = model(images, attention_maps)

            loss_topo = criterion_topo(topo_logits, topo_labels)
            loss_coord = criterion_coord(coord_pred, coord_labels)
            loss_total = loss_topo + gamma * loss_coord

            total_loss_sum += loss_total.item()
            topo_loss_sum += loss_topo.item()
            coord_loss_sum += loss_coord.item()

    num_batches = len(val_loader)
    return {
        "val_total_loss": total_loss_sum / num_batches,
        "val_topo_loss": topo_loss_sum / num_batches,
        "val_coord_loss": coord_loss_sum / num_batches,
    }


def save_checkpoint_safely(
    checkpoint: dict,
    epoch: int,
    local_checkpoint_dir: str,
    drive_checkpoint_dir: str = None,
    is_best: bool = False
) -> None:
    """Save training state locally and optionally copy to Google Drive for safety."""
    local_checkpoint_dir = Path(local_checkpoint_dir)
    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epoch_name = f"epoch_{epoch:03d}.pth"
    local_path = local_checkpoint_dir / epoch_name

    # Save state dicts locally
    torch.save(checkpoint, local_path)

    if is_best:
        best_path = local_checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)

    # Backup to safe Google Drive space in Colab mode
    if drive_checkpoint_dir is not None:
        drive_checkpoint_dir = Path(drive_checkpoint_dir)
        drive_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(local_path, drive_checkpoint_dir / epoch_name)
        if is_best:
            shutil.copy2(
                local_checkpoint_dir / "best_model.pth",
                drive_checkpoint_dir / "best_model.pth"
            )


def load_checkpoint(
    resume_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[int, float]:
    """Restore network weights, optimizer parameters, and epoch state from checkpoint."""
    print(f"Resuming training state from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    return start_epoch, best_val_loss


def print_epoch_summary(epoch: int, train_metrics: dict, val_metrics: dict, best_val_loss: float) -> None:
    """Display clean validation and loss metrics logs on console."""
    print(f"\nEpoch {epoch:03d} Summary:")
    print(
        f"  Train Total Loss: {train_metrics['train_total_loss']:.5f} "
        f"[Topo CE: {train_metrics['train_topo_loss']:.5f}, Coord L1: {train_metrics['train_coord_loss']:.5f}]"
    )
    print(
        f"  Val Total Loss:   {val_metrics['val_total_loss']:.5f} "
        f"[Topo CE: {val_metrics['val_topo_loss']:.5f}, Coord L1: {val_metrics['val_coord_loss']:.5f}]"
    )
    print(f"  Best Val Loss:    {best_val_loss:.5f}")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    # Set seed for reproducible run
    set_seed(config["project"]["seed"])

    # Setup hardware device
    device = resolve_device(config, args.device)
    print(f"Training initialized on hardware: {device}")

    # Step 1: Pre-training environment preparation
    if args.colab or config.get("colab", {}).get("enabled", False):
        ensure_colab_dataset(config)

    # Step 2: Validate preprocessing and synchronize coordinates
    config = ensure_preprocessed_data(config, args.config)

    # Step 3: Instantiate localized dataset
    dataset = RobotLocalizationDataset(
        total_csv=config["data"]["total_csv"],
        config_path=args.config
    )

    # Step 4: Random split into train, validation, and test sets
    train_ratio = float(config["data"]["train_ratio"])
    val_ratio = float(config["data"]["val_ratio"])
    test_ratio = float(config["data"].get("test_ratio", 0.1))

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config["project"]["seed"])
    )
    print(
        f"Dataset Split: {len(train_dataset)} train frames, "
        f"{len(val_dataset)} val frames, {len(test_dataset)} test frames."
    )

    # Step 5: Setup DataLoaders
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    pin_memory = bool(config["data"]["pin_memory"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Step 6: Initialize model using the correct architecture module
    model = FeatureAttentionHierarchicalNet(
        num_classes_topo=int(config["model"]["num_classes_topo"]),
        pretrained=bool(config["model"]["pretrained"]),
        dropout_p=float(config["model"]["dropout_p"])
    ).to(device)

    # Step 7: Setup Loss Criteria
    criterion_topo = nn.CrossEntropyLoss()
    criterion_coord = nn.SmoothL1Loss()

    # Step 8: Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    start_epoch = 1
    best_val_loss = float("inf")

    # Step 9: Supports checkpoint resume
    if args.resume is not None:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)

    # Step 10: Training Loop
    epochs = int(config["training"]["epochs"])
    gamma = float(config["loss"]["gamma"])
    checkpoint_interval = int(config["training"].get("checkpoint_interval", 10))

    print("\n--- Initiating Multi-task Training Loop ---")
    for epoch in range(start_epoch, epochs + 1):
        # Train epoch
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion_topo=criterion_topo,
            criterion_coord=criterion_coord,
            gamma=gamma,
            device=device
        )

        # Validate epoch
        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion_topo=criterion_topo,
            criterion_coord=criterion_coord,
            gamma=gamma,
            device=device
        )

        # Check validation loss improvement
        is_best = val_metrics["val_total_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_total_loss"]

        # Build checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "config": config,
        }

        # Safe checkpoint backup
        if epoch % checkpoint_interval == 0 or is_best:
            save_checkpoint_safely(
                checkpoint=checkpoint,
                epoch=epoch,
                local_checkpoint_dir=config["checkpoint"]["local_dir"],
                drive_checkpoint_dir=config["colab"].get("drive_checkpoint_dir") if (args.colab or config["colab"]["enabled"]) else None,
                is_best=is_best
            )

        print_epoch_summary(epoch, train_metrics, val_metrics, best_val_loss)

    print("\nTraining completed successfully.")

    # Step 11: Final evaluation on the Test set using the best model weights
    best_checkpoint_path = Path(config["checkpoint"]["local_dir"]) / "best_model.pth"
    if best_checkpoint_path.exists():
        print(f"\n--- Loading best model weights from {best_checkpoint_path} for Test Set Evaluation ---")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        test_metrics = validate_one_epoch(
            model=model,
            val_loader=test_loader,
            criterion_topo=criterion_topo,
            criterion_coord=criterion_coord,
            gamma=gamma,
            device=device
        )
        print(f"\n==========================================")
        print(f"TEST SET FINAL EVALUATION METRICS:")
        print(f"  Test Total Loss: {test_metrics['val_total_loss']:.5f}")
        print(f"  [Topo CE Loss: {test_metrics['val_topo_loss']:.5f}]")
        print(f"  [Coord L1 Loss: {test_metrics['val_coord_loss']:.5f}]")
        print(f"==========================================\n")


if __name__ == "__main__":
    main()
