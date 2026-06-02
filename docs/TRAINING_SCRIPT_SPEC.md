# TRAINING_SCRIPT_SPEC.md

# Đặc tả luồng điều phối huấn luyện cho `train.py`

> Tài liệu này là đặc tả kỹ thuật bắt buộc cho AI agent khi lập trình file quản lý trung tâm `train.py`.
>
> `train.py` chỉ đóng vai trò **Master Controller**: cấu hình môi trường, kiểm tra dữ liệu, gọi tiền xử lý nếu cần, khởi tạo DataLoader, khởi tạo model, tính loss, huấn luyện, đánh giá và sao lưu checkpoint. File này không được định nghĩa kiến trúc mạng, không xử lý CSV thô cấp thấp và không tự tạo Attention Map.

---

## 1. Vai trò của `train.py`

`train.py` là điểm vào chính của toàn bộ pipeline huấn luyện. File này kết nối hai module độc lập:

```text
Data Pipeline Module
    ├── utils/dataset.py
    └── preprocess_offline.py

Model Architecture Module
    └── models/architecture.py
```

Luồng tổng quát của `train.py`:

```text
config.json
    ↓
train.py
    ├── setup Google Colab environment
    ├── check / extract dataset
    ├── check preprocessing status
    ├── call preprocess_offline.py if needed
    ├── create Dataset / DataLoader
    ├── create Model
    ├── compute multi-task loss
    ├── optimize model
    └── save checkpoints to Google Drive
```

Ràng buộc bắt buộc:

```text
- train.py không định nghĩa ResNet50.
- train.py không định nghĩa MLP heads.
- train.py không xử lý Feature-level Attention.
- train.py không đọc từng ảnh trực tiếp.
- train.py không tự xử lý CSV thô.
- train.py không tự chạy YOLOv8 trong training loop.
- train.py chỉ gọi các interface đã định nghĩa ở module dữ liệu và module mạng.
```

---

## 2. Command-line Interface

`train.py` phải nhận file cấu hình thông qua command line:

```bash
python train.py --config config.json
```

Khi chạy trên Google Colab:

```bash
python train.py --config config.json --colab
```

Các argument đề xuất:

| Argument   | Kiểu   | Bắt buộc | Ý nghĩa                                             |
| ---------- | ------ | -------: | --------------------------------------------------- |
| `--config` | string |       Có | Đường dẫn tới `config.json`                         |
| `--colab`  | flag   |    Không | Bật chế độ tự động hóa Google Colab                 |
| `--resume` | string |    Không | Đường dẫn checkpoint `.pth` để tiếp tục huấn luyện  |
| `--device` | string |    Không | Ghi đè device trong config, ví dụ `cuda` hoặc `cpu` |

Interface tối thiểu:

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()
```

---

## 3. Cấu hình yêu cầu trong `config.json`

`train.py` phải đọc các trường sau từ `config.json`.

```json
{
  "project": {
    "seed": 42,
    "device": "cuda"
  },

  "colab": {
    "enabled": true,
    "drive_zip_path": "/content/drive/MyDrive/REGRESSION_MODEL/data.zip",
    "local_dataset_dir": "/content/dataset",
    "drive_checkpoint_dir": "/content/drive/MyDrive/REGRESSION_MODEL/checkpoints"
  },

  "data": {
    "total_csv": "/content/dataset/processed/total_poses.csv",
    "processed_root": "/content/dataset/processed",
    "attention_maps_dir": "/content/dataset/processed/attention_maps",
    "batch_size": 16,
    "num_workers": 2,
    "pin_memory": true,
    "train_ratio": 0.8,
    "val_ratio": 0.2
  },

  "model": {
    "num_classes_topo": 2,
    "pretrained": true,
    "dropout_p": 0.3
  },

  "training": {
    "epochs": 200,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "checkpoint_interval": 10
  },

  "loss": {
    "gamma": 0.5
  },

  "checkpoint": {
    "local_dir": "/content/checkpoints",
    "save_best": true,
    "monitor": "val_total_loss"
  }
}
```

Trong đó:

| Trường                         | Vai trò                                       |
| ------------------------------ | --------------------------------------------- |
| `colab.drive_zip_path`         | Đường dẫn tới `data.zip` trên Google Drive    |
| `colab.local_dataset_dir`      | Thư mục SSD nội bộ để giải nén dataset        |
| `colab.drive_checkpoint_dir`   | Thư mục checkpoint an toàn trên Google Drive  |
| `data.total_csv`               | File CSV tổng sau tiền xử lý                  |
| `data.attention_maps_dir`      | Thư mục chứa Attention Maps                   |
| `training.checkpoint_interval` | Chu kỳ sao lưu checkpoint, mặc định 10 epochs |
| `loss.gamma`                   | Hệ số cân bằng loss hồi quy tọa độ            |

---

## 4. Tự động cấu hình môi trường Google Colab

Khi chạy ở chế độ Colab, `train.py` phải kiểm tra xem file dữ liệu nén `data.zip` trên Google Drive đã được copy và giải nén vào SSD nội bộ `/content/dataset` hay chưa.

### 4.1. Điều kiện dataset đã sẵn sàng

Dataset được xem là đã sẵn sàng nếu thư mục sau tồn tại và không rỗng:

```text
/content/dataset
```

Nếu `/content/dataset` chưa tồn tại hoặc đang rỗng, `train.py` phải tự động giải nén:

```text
/content/drive/MyDrive/.../data.zip
        ↓
/content/dataset
```

### 4.2. Quy tắc giải nén bằng `zipfile`

Không dùng lệnh shell kiểu:

```bash
!unzip data.zip
```

`train.py` phải dùng Python chuẩn:

```python
import zipfile

with zipfile.ZipFile(drive_zip_path, "r") as zip_ref:
    zip_ref.extractall(local_dataset_dir)
```

Hàm bắt buộc:

```python
def ensure_colab_dataset(config):
    drive_zip_path = Path(config["colab"]["drive_zip_path"])
    local_dataset_dir = Path(config["colab"]["local_dataset_dir"])

    if local_dataset_dir.exists() and any(local_dataset_dir.iterdir()):
        print("[Colab] Dataset already exists on local SSD.")
        return

    if not drive_zip_path.exists():
        raise FileNotFoundError(f"data.zip not found: {drive_zip_path}")

    local_dataset_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(drive_zip_path, "r") as zip_ref:
        zip_ref.extractall(local_dataset_dir)

    print(f"[Colab] Dataset extracted to: {local_dataset_dir}")
```

> Quy tắc hiệu năng: không huấn luyện trực tiếp trên Google Drive. Dataset phải được giải nén vào SSD nội bộ `/content/dataset`.

---

## 5. Master Logic kiểm tra tiền xử lý dữ liệu

Trước khi tạo Dataset, `train.py` phải kiểm tra dữ liệu đã được tiền xử lý đầy đủ hay chưa.

Dữ liệu được xem là hợp lệ khi có đủ:

```text
total_poses.csv
attention_maps/
```

Trong đó:

| Thành phần        | Vai trò                                                         |
| ----------------- | --------------------------------------------------------------- |
| `total_poses.csv` | File tổng chứa ảnh, Attention Map, nhãn topo, tọa độ đã đồng bộ |
| `attention_maps/` | Thư mục chứa ảnh xám Attention Map tạo offline từ YOLOv8        |

### 5.1. Điều kiện cần kiểm tra

`train.py` phải kiểm tra:

```python
total_csv_exists = Path(config["data"]["total_csv"]).exists()

attention_dir = Path(config["data"]["attention_maps_dir"])
attention_dir_exists = attention_dir.exists()
attention_dir_not_empty = attention_dir_exists and any(attention_dir.iterdir())
```

Nếu thiếu `total_poses.csv` hoặc thiếu thư mục `attention_maps`, `train.py` phải kích hoạt tiền xử lý offline.

---

## 6. Interface yêu cầu từ `preprocess_offline.py`

`train.py` nên gọi trực tiếp hàm từ module tiền xử lý:

```python
from preprocess_offline import is_preprocessed_ready, preprocess_offline
```

Interface yêu cầu:

```python
def is_preprocessed_ready(config: dict) -> bool:
    ...
```

```python
def preprocess_offline(config_path: str) -> None:
    ...
```

Trong đó, `preprocess_offline(config_path)` chịu trách nhiệm:

```text
- Đọc room_poses.csv và corridor_poses.csv.
- Đồng bộ hệ quy chiếu.
- Gán topo_label.
- Tạo hoặc kiểm tra attention_maps bằng YOLOv8 offline.
- Tạo total_poses.csv.
- Cập nhật global_normalization trong config.json.
```

Logic bắt buộc trong `train.py`:

```python
def ensure_preprocessed_data(config, config_path):
    if is_preprocessed_ready(config):
        print("[Data] Preprocessed data is ready.")
        return config

    print("[Data] Missing total_poses.csv or attention_maps. Running offline preprocessing...")
    preprocess_offline(config_path)

    updated_config = load_json(config_path)

    if not is_preprocessed_ready(updated_config):
        raise RuntimeError(
            "Preprocessing failed: total_poses.csv or attention_maps is still missing."
        )

    return updated_config
```

> `train.py` không tự tạo Attention Map. Nó chỉ kích hoạt script offline. Logic YOLOv8 thuộc `preprocess_offline.py`.

---

## 7. Khởi tạo Dataset và DataLoader

Sau khi dữ liệu đã sẵn sàng, `train.py` tạo Dataset từ module dữ liệu:

```python
from utils.dataset import RobotLocalizationDataset
```

Dataset được khởi tạo bằng:

```python
dataset = RobotLocalizationDataset(
    total_csv=config["data"]["total_csv"],
    config_path=config_path
)
```

Nếu `total_poses.csv` là file tổng duy nhất, `train.py` được phép chia dataset bằng `random_split`.

```python
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(seed)
)
```

Khởi tạo DataLoader:

```python
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
```

Batch trả về từ Dataset phải có dạng:

```python
images, attention_maps, topo_labels, coord_labels = batch
```

Shape bắt buộc:

| Tensor           |          Shape |
| ---------------- | -------------: |
| `images`         | `[B, 3, H, W]` |
| `attention_maps` | `[B, 1, H, W]` |
| `topo_labels`    |          `[B]` |
| `coord_labels`   |       `[B, 2]` |

---

## 8. Khởi tạo Model

`train.py` phải khởi tạo model từ module mạng:

```python
from models.architecture import DynamicAwareLocalizationNet
```

Code bắt buộc:

```python
model = DynamicAwareLocalizationNet(
    num_classes_topo=config["model"]["num_classes_topo"],
    pretrained=config["model"]["pretrained"],
    dropout_p=config["model"]["dropout_p"]
)

model = model.to(device)
```

Ràng buộc:

```text
- train.py không được tự định nghĩa ResNet50.
- train.py không được tự định nghĩa MLP heads.
- train.py không được tự xử lý Feature-level Attention.
```

---

## 9. Multi-task Loss Function

Mô hình trả về:

```python
topo_logits, coord_pred = model(images, attention_maps)
```

Trong đó:

| Output        | Ý nghĩa                                    |
| ------------- | ------------------------------------------ |
| `topo_logits` | Raw logits từ nhánh phân loại topo         |
| `coord_pred`  | Tọa độ normalized dự đoán từ nhánh hồi quy |

### 9.1. Loss phân loại topo

Nhánh phân loại sử dụng:

```python
criterion_topo = nn.CrossEntropyLoss()
```

Công thức:

[
\mathcal{L}_{CrossEntropy}
==========================

CE(topo_logits, topo_labels)
]

Ràng buộc:

```text
- topo_logits phải là raw logits.
- Không Softmax trước CrossEntropyLoss.
- topo_labels phải có dtype torch.long.
```

### 9.2. Loss hồi quy tọa độ

Nhánh hồi quy sử dụng:

```python
criterion_coord = nn.SmoothL1Loss()
```

Công thức:

[
\mathcal{L}_{SmoothL1}
======================

SmoothL1(coord_pred, coord_labels)
]

Ràng buộc:

```text
- coord_pred có shape [B, 2].
- coord_labels có shape [B, 2].
- coord_labels phải là tọa độ đã chuẩn hóa Min-Max.
```

### 9.3. Loss tổng hợp

Loss tổng thể:

[
\mathcal{L}_{total}
===================

\mathcal{L}*{CrossEntropy}
+
\gamma \cdot \mathcal{L}*{SmoothL1}
]

Trong code:

```python
loss_topo = criterion_topo(topo_logits, topo_labels)
loss_coord = criterion_coord(coord_pred, coord_labels)
loss_total = loss_topo + gamma * loss_coord
```

Trong đó:

| Ký hiệu                      | Ý nghĩa                       |
| ---------------------------- | ----------------------------- |
| (\mathcal{L}_{CrossEntropy}) | Loss phân loại topo           |
| (\mathcal{L}_{SmoothL1})     | Loss hồi quy tọa độ           |
| (\gamma)                     | Hệ số cân bằng giữa hai nhánh |

`gamma` được đọc từ:

```python
gamma = float(config["loss"]["gamma"])
```

---

## 10. Optimizer

Optimizer mặc định:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"]
)
```

`train.py` được phép bổ sung scheduler nếu cấu hình có khai báo, nhưng scheduler không phải thành phần bắt buộc trong đặc tả này.

---

## 11. Training Epoch Logic

Một epoch huấn luyện phải thực hiện đúng thứ tự:

```text
1. model.train()
2. Lặp qua train_loader
3. Chuyển batch Tensor sang device
4. Forward model
5. Tính loss_topo
6. Tính loss_coord
7. Tính loss_total
8. optimizer.zero_grad()
9. loss_total.backward()
10. optimizer.step()
11. Cộng dồn thống kê loss
```

Pseudo-code:

```python
def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion_topo,
    criterion_coord,
    gamma,
    device
):
    model.train()

    total_loss_sum = 0.0
    topo_loss_sum = 0.0
    coord_loss_sum = 0.0

    for images, attention_maps, topo_labels, coord_labels in train_loader:
        images = images.to(device)
        attention_maps = attention_maps.to(device)
        topo_labels = topo_labels.to(device)
        coord_labels = coord_labels.to(device)

        topo_logits, coord_pred = model(images, attention_maps)

        loss_topo = criterion_topo(topo_logits, topo_labels)
        loss_coord = criterion_coord(coord_pred, coord_labels)
        loss_total = loss_topo + gamma * loss_coord

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        total_loss_sum += loss_total.item()
        topo_loss_sum += loss_topo.item()
        coord_loss_sum += loss_coord.item()

    num_batches = len(train_loader)

    return {
        "train_total_loss": total_loss_sum / num_batches,
        "train_topo_loss": topo_loss_sum / num_batches,
        "train_coord_loss": coord_loss_sum / num_batches,
    }
```

---

## 12. Validation Logic

Validation không được cập nhật trọng số.

```text
1. model.eval()
2. torch.no_grad()
3. Lặp qua val_loader
4. Forward model
5. Tính loss_topo, loss_coord, loss_total
6. Không gọi backward
7. Không gọi optimizer.step
```

Pseudo-code:

```python
def validate_one_epoch(
    model,
    val_loader,
    criterion_topo,
    criterion_coord,
    gamma,
    device
):
    model.eval()

    total_loss_sum = 0.0
    topo_loss_sum = 0.0
    coord_loss_sum = 0.0

    with torch.no_grad():
        for images, attention_maps, topo_labels, coord_labels in val_loader:
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
```

---

## 13. Cơ chế checkpoint an toàn

Checkpoint phải được lưu theo hai lớp:

```text
1. Lưu cục bộ vào SSD runtime.
2. Sao chép sang Google Drive để tránh mất dữ liệu khi Colab ngắt kết nối.
```

### 13.1. Nội dung checkpoint

Checkpoint `.pth` phải chứa:

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_metrics": train_metrics,
    "val_metrics": val_metrics,
    "best_val_loss": best_val_loss,
    "config": config,
}
```

### 13.2. Lưu định kỳ sau mỗi 10 epochs

Sau mỗi 10 epochs, `train.py` phải tự động lưu checkpoint:

```python
if epoch % 10 == 0:
    save_checkpoint_safely(...)
```

Chu kỳ này được đọc từ config:

```python
checkpoint_interval = int(config["training"].get("checkpoint_interval", 10))
```

### 13.3. Đường dẫn checkpoint

Checkpoint cục bộ:

```text
/content/checkpoints/epoch_010.pth
/content/checkpoints/epoch_020.pth
```

Checkpoint trên Google Drive:

```text
/content/drive/MyDrive/REGRESSION_MODEL/checkpoints/epoch_010.pth
/content/drive/MyDrive/REGRESSION_MODEL/checkpoints/epoch_020.pth
```

### 13.4. Hàm lưu checkpoint an toàn

```python
def save_checkpoint_safely(
    checkpoint,
    epoch,
    local_checkpoint_dir,
    drive_checkpoint_dir=None,
    is_best=False
):
    local_checkpoint_dir = Path(local_checkpoint_dir)
    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epoch_name = f"epoch_{epoch:03d}.pth"
    local_path = local_checkpoint_dir / epoch_name

    torch.save(checkpoint, local_path)

    if is_best:
        best_path = local_checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)

    if drive_checkpoint_dir is not None:
        drive_checkpoint_dir = Path(drive_checkpoint_dir)
        drive_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(local_path, drive_checkpoint_dir / epoch_name)

        if is_best:
            shutil.copy2(
                local_checkpoint_dir / "best_model.pth",
                drive_checkpoint_dir / "best_model.pth"
            )
```

> Không chỉ lưu checkpoint vào `/content`. Runtime Colab có thể bị reset. Bản sao an toàn phải được đẩy về Google Drive định kỳ.

---

## 14. Best Checkpoint Logic

Ngoài checkpoint định kỳ, `train.py` nên lưu `best_model.pth` khi validation loss tốt hơn trước.

```python
if val_metrics["val_total_loss"] < best_val_loss:
    best_val_loss = val_metrics["val_total_loss"]
    is_best = True
else:
    is_best = False
```

Nếu `is_best=True`, lưu:

```text
best_model.pth
```

vào cả local checkpoint dir và Google Drive checkpoint dir.

---

## 15. Resume Training

`train.py` có thể hỗ trợ tiếp tục huấn luyện từ checkpoint.

Command:

```bash
python train.py \
    --config config.json \
    --resume /content/drive/MyDrive/REGRESSION_MODEL/checkpoints/epoch_050.pth
```

Logic:

```python
checkpoint = torch.load(resume_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] + 1
best_val_loss = checkpoint["best_val_loss"]
```

Nếu không có `--resume`:

```python
start_epoch = 1
best_val_loss = float("inf")
```

---

## 16. Main Execution Flow bắt buộc

Pseudo-code tổng thể:

```python
def main():
    args = parse_args()
    config = load_json(args.config)

    set_seed(config["project"]["seed"])

    device = resolve_device(config, args.device)

    if args.colab or config.get("colab", {}).get("enabled", False):
        ensure_colab_dataset(config)

    config = ensure_preprocessed_data(config, args.config)

    dataset = RobotLocalizationDataset(
        total_csv=config["data"]["total_csv"],
        config_path=args.config
    )

    train_dataset, val_dataset = split_dataset(dataset, config)

    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        config
    )

    model = DynamicAwareLocalizationNet(
        num_classes_topo=config["model"]["num_classes_topo"],
        pretrained=config["model"]["pretrained"],
        dropout_p=config["model"]["dropout_p"]
    ).to(device)

    criterion_topo = nn.CrossEntropyLoss()
    criterion_coord = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume is not None:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume,
            model,
            optimizer,
            device
        )

    epochs = config["training"]["epochs"]
    gamma = config["loss"]["gamma"]
    checkpoint_interval = config["training"].get("checkpoint_interval", 10)

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion_topo,
            criterion_coord,
            gamma,
            device
        )

        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion_topo,
            criterion_coord,
            gamma,
            device
        )

        is_best = val_metrics["val_total_loss"] < best_val_loss

        if is_best:
            best_val_loss = val_metrics["val_total_loss"]

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "config": config,
        }

        if epoch % checkpoint_interval == 0 or is_best:
            save_checkpoint_safely(
                checkpoint=checkpoint,
                epoch=epoch,
                local_checkpoint_dir=config["checkpoint"]["local_dir"],
                drive_checkpoint_dir=config["colab"].get("drive_checkpoint_dir"),
                is_best=is_best
            )

        print_epoch_summary(epoch, train_metrics, val_metrics, best_val_loss)
```

---

## 17. Code skeleton tối thiểu cho `train.py`

```python
import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils.dataset import RobotLocalizationDataset
from models.architecture import DynamicAwareLocalizationNet
from preprocess_offline import is_preprocessed_ready, preprocess_offline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(config, override_device=None):
    if override_device is not None:
        return torch.device(override_device)

    requested_device = config["project"].get("device", "cuda")

    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def ensure_colab_dataset(config):
    drive_zip_path = Path(config["colab"]["drive_zip_path"])
    local_dataset_dir = Path(config["colab"]["local_dataset_dir"])

    if local_dataset_dir.exists() and any(local_dataset_dir.iterdir()):
        print("[Colab] Dataset already exists on local SSD.")
        return

    if not drive_zip_path.exists():
        raise FileNotFoundError(f"data.zip not found: {drive_zip_path}")

    local_dataset_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(drive_zip_path, "r") as zip_ref:
        zip_ref.extractall(local_dataset_dir)

    print(f"[Colab] Dataset extracted to: {local_dataset_dir}")


def ensure_preprocessed_data(config, config_path):
    if is_preprocessed_ready(config):
        print("[Data] Preprocessed data is ready.")
        return config

    print("[Data] Missing total_poses.csv or attention_maps. Running offline preprocessing...")
    preprocess_offline(config_path)

    updated_config = load_json(config_path)

    if not is_preprocessed_ready(updated_config):
        raise RuntimeError(
            "Preprocessing failed: total_poses.csv or attention_maps is still missing."
        )

    return updated_config
```

Phần còn lại của `train.py` phải bám theo các hàm `train_one_epoch`, `validate_one_epoch`, `save_checkpoint_safely`, `load_checkpoint` và `main()` đã quy định ở các mục trên.

---

## 18. Các lỗi triển khai phải tránh

| Lỗi                                                        | Trạng thái               |
| ---------------------------------------------------------- | ------------------------ |
| Huấn luyện trực tiếp trên file zip trong Google Drive      | Không hợp lệ             |
| Không giải nén dataset vào `/content/dataset`              | Không đúng yêu cầu Colab |
| Không kiểm tra `total_poses.csv` trước huấn luyện          | Không hợp lệ             |
| Không kiểm tra thư mục `attention_maps`                    | Không hợp lệ             |
| Tự tạo Attention Map trong `train.py`                      | Sai phân tách module     |
| Định nghĩa model trong `train.py`                          | Không hợp lệ             |
| Dùng Softmax trước `CrossEntropyLoss`                      | Không hợp lệ             |
| Dùng `MSELoss` thay cho `SmoothL1Loss`                     | Không đúng đặc tả        |
| Không nhân gamma vào loss hồi quy                          | Không hợp lệ             |
| Chỉ lưu checkpoint trong `/content`                        | Không an toàn trên Colab |
| Không sao lưu checkpoint về Google Drive sau mỗi 10 epochs | Không đúng yêu cầu       |
| Dataset trả path thay vì Tensor rồi `train.py` tự đọc ảnh  | Không hợp lệ             |

---

## 19. Tiêu chí hoàn thành `train.py`

`train.py` được xem là hoàn thành khi thỏa mãn toàn bộ tiêu chí sau:

```text
1. Đọc được config.json từ command line.
2. Tự xác định device cuda/cpu.
3. Ở chế độ Colab, kiểm tra và giải nén data.zip từ Google Drive vào /content/dataset bằng zipfile.
4. Kiểm tra sự tồn tại của total_poses.csv.
5. Kiểm tra sự tồn tại và trạng thái không rỗng của attention_maps.
6. Nếu thiếu dữ liệu tiền xử lý, tự động gọi preprocess_offline.py.
7. Khởi tạo RobotLocalizationDataset từ utils/dataset.py.
8. Khởi tạo DataLoader cho train và validation.
9. Khởi tạo DynamicAwareLocalizationNet từ models/architecture.py.
10. Tính CrossEntropyLoss trên raw logits.
11. Tính SmoothL1Loss trên tọa độ normalized.
12. Tính loss_total = loss_topo + gamma * loss_coord.
13. Huấn luyện bằng optimizer AdamW.
14. Có validation loop dùng torch.no_grad().
15. Lưu checkpoint định kỳ sau mỗi 10 epochs.
16. Sao lưu checkpoint lên Google Drive.
17. Lưu best_model.pth khi validation loss tốt hơn.
18. Hỗ trợ resume training từ checkpoint nếu có --resume.
19. Không chứa logic kiến trúc mạng hoặc xử lý dữ liệu cấp thấp.
```

---

## 20. Kết luận kỹ thuật

`train.py` là bộ điều phối trung tâm của hệ thống. Nó biểu diễn ánh xạ vận hành:

```text
config.json
    ↓
Colab dataset preparation
    ↓
Offline preprocessing check
    ↓
Dataset / DataLoader
    ↓
Model
    ↓
Multi-task loss
    ↓
Optimization
    ↓
Safe checkpoint backup
```

File này phải giữ đúng nguyên lý phân tách module. Mọi xử lý CSV, offset, tạo Attention Map và chuẩn hóa dữ liệu thuộc về Data Pipeline. Mọi định nghĩa ResNet50, Feature-level Attention và MLP heads thuộc về Model Architecture. `train.py` chỉ kết nối, huấn luyện, đánh giá và sao lưu.
