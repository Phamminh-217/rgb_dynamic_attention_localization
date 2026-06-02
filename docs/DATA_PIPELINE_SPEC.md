# DATA_PIPELINE_SPEC.md

# Đặc tả kỹ thuật Data Pipeline cho `preprocess_offline.py` và `utils/dataset.py`

> Tài liệu này là đặc tả bắt buộc cho AI agent khi lập trình module dữ liệu của hệ thống định vị robot di động trong nhà.
>
> Module dữ liệu chỉ được phép xử lý I/O, CSV, ảnh RGB, Attention Map, nhãn topo và tọa độ ground truth. Module này tuyệt đối không được định nghĩa hoặc gọi bất kỳ thành phần nào liên quan đến mạng nơ-ron như ResNet, MLP, forward pass, loss hoặc optimizer.

---

## 1. Phạm vi của Data Pipeline Module

Data Pipeline Module gồm hai file chính:

```text
preprocess_offline.py
utils/dataset.py
```

Trong đó, `preprocess_offline.py` chịu trách nhiệm xử lý dữ liệu thô trước huấn luyện. File `utils/dataset.py` chịu trách nhiệm định nghĩa PyTorch Dataset để nạp dữ liệu đã xử lý cho training loop.

| File                     | Vai trò                                                                                                                                   | Không được chứa                                             |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `preprocess_offline.py`  | Đọc CSV thô, đồng bộ hệ quy chiếu, gán nhãn topo, gộp dữ liệu, tạo `total_poses.csv`, cập nhật `global_normalization` trong `config.json` | Không định nghĩa Dataset, không định nghĩa model            |
| `utils/dataset.py`       | Đọc `total_poses.csv`, đọc ảnh RGB, đọc Attention Map, chuẩn hóa ảnh, chuẩn hóa tọa độ, trả Tensor cho DataLoader                         | Không xử lý offset, không gọi YOLOv8, không định nghĩa mạng |
| `models/architecture.py` | Không thuộc module dữ liệu                                                                                                                | Không được xuất hiện trong module dữ liệu                   |

> Nguyên tắc bắt buộc: Data Pipeline không biết kiến trúc mạng là gì. Nó chỉ tạo ra Tensor sạch để đưa vào model.

---

## 2. Cấu trúc dữ liệu đầu vào

Dữ liệu thô ban đầu gồm hai file CSV độc lập:

```text
room_poses.csv
corridor_poses.csv
```

Hai file này tương ứng với dữ liệu thu trong phòng và dữ liệu thu ở hành lang.

### 2.1. Schema tối thiểu của CSV thô

Mỗi file CSV thô bắt buộc phải có tối thiểu các cột sau:

```csv
image_path,attention_path,x,y
```

Ý nghĩa từng cột:

| Cột              | Kiểu dữ liệu | Ý nghĩa                                       |
| ---------------- | ------------ | --------------------------------------------- |
| `image_path`     | string       | Đường dẫn tới ảnh RGB gốc                     |
| `attention_path` | string       | Đường dẫn tới ảnh xám Attention Map tương ứng |
| `x`              | float        | Tọa độ (x) ban đầu                            |
| `y`              | float        | Tọa độ (y) ban đầu                            |

Các cột bổ sung như `timestamp`, `frame_id`, `sequence_id` có thể được giữ lại trong `total_poses.csv`, nhưng không bắt buộc dùng trong Dataset.

---

## 3. Cấu hình bắt buộc trong `config.json`

`config.json` phải chứa các trường phục vụ tiền xử lý và Dataset.

```json
{
  "data": {
    "room_csv": "data/raw/room_poses.csv",
    "corridor_csv": "data/raw/corridor_poses.csv",
    "total_csv": "data/processed/total_poses.csv",
    "image_size": [224, 224]
  },

  "coordinate_alignment": {
    "corridor": {
      "delta_x": 14.45,
      "delta_y": 2.45
    }
  },

  "global_normalization": {
    "x_min": null,
    "x_max": null,
    "y_min": null,
    "y_max": null
  },

  "labels": {
    "room": 0,
    "corridor": 1
  },

  "image_normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

Trong đó:

| Trường                                  | Ý nghĩa                                      |
| --------------------------------------- | -------------------------------------------- |
| `data.room_csv`                         | Đường dẫn tới CSV thô của phòng              |
| `data.corridor_csv`                     | Đường dẫn tới CSV thô của hành lang          |
| `data.total_csv`                        | Đường dẫn file CSV tổng sau tiền xử lý       |
| `coordinate_alignment.corridor.delta_x` | Offset theo trục (x) cho dữ liệu hành lang   |
| `coordinate_alignment.corridor.delta_y` | Offset theo trục (y) cho dữ liệu hành lang   |
| `global_normalization`                  | Cực biên toàn cục dùng cho chuẩn hóa Min-Max |
| `labels.room`                           | Nhãn topo cho dữ liệu phòng                  |
| `labels.corridor`                       | Nhãn topo cho dữ liệu hành lang              |
| `image_normalization`                   | Tham số chuẩn hóa ảnh RGB theo ImageNet      |

---

## 4. Quy trình tiền xử lý tọa độ trong `preprocess_offline.py`

`preprocess_offline.py` là script chạy offline trước khi huấn luyện. File này có nhiệm vụ đọc hai file CSV thô, đồng bộ hệ quy chiếu, gán nhãn topo, gộp dữ liệu và cập nhật cực biên tọa độ vào `config.json`.

### 4.1. Đọc hai file CSV thô

Script phải đọc độc lập hai file:

```python
room_df = pd.read_csv(room_csv)
corridor_df = pd.read_csv(corridor_csv)
```

Trước khi xử lý, script phải kiểm tra các cột bắt buộc:

```text
image_path
attention_path
x
y
```

Nếu thiếu một trong các cột trên, script phải dừng và báo lỗi rõ ràng.

---

### 4.2. Gán nhãn topo tự động

Dữ liệu phòng được gán:

[
topo_label = 0
]

Dữ liệu hành lang được gán:

[
topo_label = 1
]

Trong code:

```python
room_df["topo_label"] = 0
corridor_df["topo_label"] = 1
```

Quy ước này phải thống nhất với `config.json`:

```json
{
  "labels": {
    "room": 0,
    "corridor": 1
  }
}
```

---

### 4.3. Đồng bộ hệ quy chiếu cho dữ liệu hành lang

Dữ liệu phòng được xem là nằm trong hệ quy chiếu chuẩn. Dữ liệu hành lang phải được dịch chuyển về cùng hệ quy chiếu bằng offset cấu hình.

Với mỗi mẫu trong `corridor_df`, tọa độ được cập nhật theo công thức:

[
x^{global} = x^{corridor} + \Delta x
]

[
y^{global} = y^{corridor} + \Delta y
]

Trong đó:

| Ký hiệu                      | Ý nghĩa                                            |
| ---------------------------- | -------------------------------------------------- |
| (x^{corridor}, y^{corridor}) | Tọa độ hành lang ban đầu trong CSV thô             |
| (\Delta x, \Delta y)         | Offset đọc từ `config.json`                        |
| (x^{global}, y^{global})     | Tọa độ hành lang sau khi đưa về hệ quy chiếu chung |

Code bắt buộc:

```python
delta_x = config["coordinate_alignment"]["corridor"]["delta_x"]
delta_y = config["coordinate_alignment"]["corridor"]["delta_y"]

corridor_df["x"] = corridor_df["x"] + delta_x
corridor_df["y"] = corridor_df["y"] + delta_y
```

> Offset chỉ áp dụng cho dữ liệu hành lang. Không được cộng offset cho dữ liệu phòng.

---

### 4.4. Gộp hai tập dữ liệu

Sau khi gán nhãn và đồng bộ tọa độ, hai DataFrame được gộp lại:

```python
total_df = pd.concat([room_df, corridor_df], ignore_index=True)
```

File tổng được ghi ra:

```text
data/processed/total_poses.csv
```

Schema tối thiểu của `total_poses.csv`:

```csv
image_path,attention_path,x,y,topo_label
```

Ví dụ:

```csv
image_path,attention_path,x,y,topo_label
data/raw/room/images/room_000001.jpg,data/processed/attention_maps/room_000001.png,1.25,2.40,0
data/raw/corridor/images/corridor_000001.jpg,data/processed/attention_maps/corridor_000001.png,15.80,3.10,1
```

---

### 4.5. Quét cực biên tọa độ toàn cục

Sau khi tạo `total_df`, script phải quét toàn bộ dữ liệu tổng để tìm cực biên:

[
x_{min} = \min(x_i)
]

[
x_{max} = \max(x_i)
]

[
y_{min} = \min(y_i)
]

[
y_{max} = \max(y_i)
]

Code bắt buộc:

```python
x_min = float(total_df["x"].min())
x_max = float(total_df["x"].max())
y_min = float(total_df["y"].min())
y_max = float(total_df["y"].max())
```

Sau đó cập nhật vào `config.json` tại trường:

```json
"global_normalization": {
  "x_min": x_min,
  "x_max": x_max,
  "y_min": y_min,
  "y_max": y_max
}
```

> Cực biên phải được tính sau khi đã cộng offset cho dữ liệu hành lang. Không được tính cực biên từ hai hệ quy chiếu chưa đồng bộ.

---

### 4.6. Điều kiện hợp lệ của cực biên

Sau khi tính cực biên, script phải kiểm tra:

[
x_{max} > x_{min}
]

[
y_{max} > y_{min}
]

Nếu một trong hai điều kiện sai, script phải dừng vì không thể chuẩn hóa Min-Max.

---

### 4.7. Pseudo-code cho `preprocess_offline.py`

```python
def preprocess_offline(config_path):
    config = load_json(config_path)

    room_csv = config["data"]["room_csv"]
    corridor_csv = config["data"]["corridor_csv"]
    total_csv = config["data"]["total_csv"]

    room_df = pd.read_csv(room_csv)
    corridor_df = pd.read_csv(corridor_csv)

    validate_required_columns(room_df)
    validate_required_columns(corridor_df)

    room_df["topo_label"] = config["labels"]["room"]
    corridor_df["topo_label"] = config["labels"]["corridor"]

    delta_x = config["coordinate_alignment"]["corridor"]["delta_x"]
    delta_y = config["coordinate_alignment"]["corridor"]["delta_y"]

    corridor_df["x"] = corridor_df["x"] + delta_x
    corridor_df["y"] = corridor_df["y"] + delta_y

    total_df = pd.concat([room_df, corridor_df], ignore_index=True)

    x_min = float(total_df["x"].min())
    x_max = float(total_df["x"].max())
    y_min = float(total_df["y"].min())
    y_max = float(total_df["y"].max())

    assert x_max > x_min
    assert y_max > y_min

    config["global_normalization"]["x_min"] = x_min
    config["global_normalization"]["x_max"] = x_max
    config["global_normalization"]["y_min"] = y_min
    config["global_normalization"]["y_max"] = y_max

    save_csv(total_df, total_csv)
    save_json(config, config_path)
```

---

## 5. Quy chuẩn thiết kế `RobotLocalizationDataset`

`RobotLocalizationDataset` được định nghĩa trong:

```text
utils/dataset.py
```

Class này kế thừa:

```python
torch.utils.data.Dataset
```

Tên class bắt buộc:

```python
RobotLocalizationDataset
```

---

### 5.1. Hàm khởi tạo

Interface bắt buộc:

```python
class RobotLocalizationDataset(Dataset):
    def __init__(self, total_csv: str, config_path: str):
        ...
```

Trong đó:

| Tham số       | Ý nghĩa                              |
| ------------- | ------------------------------------ |
| `total_csv`   | Đường dẫn tới file `total_poses.csv` |
| `config_path` | Đường dẫn tới file `config.json`     |

Trong `__init__`, Dataset phải thực hiện:

```text
1. Đọc config.json.
2. Đọc total_poses.csv.
3. Kiểm tra các cột bắt buộc.
4. Đọc image_size.
5. Đọc global_normalization.
6. Tạo transform cho ảnh RGB.
7. Tạo transform cho Attention Map.
```

Dataset không được xử lý offset. Offset đã phải được xử lý trước trong `preprocess_offline.py`.

---

### 5.2. Hàm `__len__`

Hàm `__len__` trả về tổng số frame trong file tổng:

```python
def __len__(self):
    return len(self.records)
```

Trong đó `self.records` là DataFrame đọc từ `total_poses.csv`.

---

### 5.3. Hàm `__getitem__`

Hàm `__getitem__(self, idx)` bắt buộc đọc đồng thời và trả về đúng 4 thành phần:

```python
image_tensor, attention_tensor, topo_label, coord_label
```

Trong đó:

| Output             | Kiểu                |       Shape | Ý nghĩa                                       |
| ------------------ | ------------------- | ----------: | --------------------------------------------- |
| `image_tensor`     | `torch.FloatTensor` | `[3, H, W]` | Ảnh RGB gốc đã resize và chuẩn hóa ImageNet   |
| `attention_tensor` | `torch.FloatTensor` | `[1, H, W]` | Ảnh xám Attention Map, giá trị trong `[0, 1]` |
| `topo_label`       | `torch.LongTensor`  |      scalar | Nhãn topo                                     |
| `coord_label`      | `torch.FloatTensor` |       `[2]` | Tọa độ normalized `[x_n, y_n]`                |

---

## 6. Xử lý ảnh RGB trong Dataset

Ảnh RGB được đọc từ `image_path` trong `total_poses.csv`.

Quy trình bắt buộc:

```text
1. Đọc ảnh bằng PIL.
2. Chuyển sang RGB.
3. Resize về [H, W] từ config.
4. Chuyển sang Tensor.
5. Chuẩn hóa theo ImageNet mean/std.
```

Thông số chuẩn hóa ImageNet:

[
mean = [0.485, 0.456, 0.406]
]

[
std = [0.229, 0.224, 0.225]
]

Transform đề xuất:

```python
self.image_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=config["image_normalization"]["mean"],
        std=config["image_normalization"]["std"],
    ),
])
```

Output:

```text
image_tensor: [3, H, W]
```

---

## 7. Xử lý Attention Map trong Dataset

Attention Map được đọc từ `attention_path` trong `total_poses.csv`.

Quy trình bắt buộc:

```text
1. Đọc ảnh bằng PIL.
2. Chuyển sang grayscale bằng convert("L").
3. Resize về [H, W].
4. Chuyển sang Tensor.
5. Đảm bảo giá trị pixel nằm trong [0, 1].
6. Trả về Tensor shape [1, H, W].
```

Transform đề xuất:

```python
self.attention_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
])
```

`transforms.ToTensor()` tự động chuyển ảnh grayscale từ giá trị pixel `[0, 255]` về `[0, 1]`.

Output:

```text
attention_tensor: [1, H, W]
```

> Không chuẩn hóa Attention Map bằng ImageNet mean/std. Attention Map không phải ảnh RGB.

---

## 8. Chuẩn hóa tọa độ trong Dataset

Dataset đọc tọa độ thực tế đã đồng bộ hệ quy chiếu từ `total_poses.csv`:

[
x, y
]

Sau đó chuẩn hóa bằng Min-Max sử dụng `global_normalization` trong `config.json`.

Công thức:

[
x_n = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
]

[
y_n = \frac{y - y_{\min}}{y_{\max} - y_{\min}}
]

Trong đó:

| Ký hiệu            | Ý nghĩa                                 |
| ------------------ | --------------------------------------- |
| (x, y)             | Tọa độ thực tế sau coordinate alignment |
| (x_n, y_n)         | Tọa độ chuẩn hóa                        |
| (x_{min}, x_{max}) | Cực biên toàn cục trục (x)              |
| (y_{min}, y_{max}) | Cực biên toàn cục trục (y)              |

Code bắt buộc:

```python
x_n = (x - x_min) / (x_max - x_min)
y_n = (y - y_min) / (y_max - y_min)
coord_label = torch.tensor([x_n, y_n], dtype=torch.float32)
```

Output:

```text
coord_label: [2]
```

---

## 9. Xử lý nhãn topo

Dataset đọc `topo_label` từ `total_poses.csv`.

Code bắt buộc:

```python
topo_label = torch.tensor(int(row["topo_label"]), dtype=torch.long)
```

Output:

```text
topo_label: scalar LongTensor
```

Quy ước:

| Khu vực   | Giá trị |
| --------- | ------: |
| Phòng     |       0 |
| Hành lang |       1 |

---

## 10. Code skeleton bắt buộc cho `utils/dataset.py`

```python
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class RobotLocalizationDataset(Dataset):
    def __init__(self, total_csv: str, config_path: str):
        self.total_csv = Path(total_csv)
        self.config_path = Path(config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.records = pd.read_csv(self.total_csv)
        self._validate_columns()

        image_size = self.config["data"]["image_size"]
        self.height = int(image_size[0])
        self.width = int(image_size[1])

        norm = self.config["global_normalization"]
        self.x_min = float(norm["x_min"])
        self.x_max = float(norm["x_max"])
        self.y_min = float(norm["y_min"])
        self.y_max = float(norm["y_max"])

        if self.x_max <= self.x_min:
            raise ValueError("Invalid x normalization range: x_max must be greater than x_min.")

        if self.y_max <= self.y_min:
            raise ValueError("Invalid y normalization range: y_max must be greater than y_min.")

        mean = self.config["image_normalization"]["mean"]
        std = self.config["image_normalization"]["std"]

        self.image_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.attention_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
        ])

    def _validate_columns(self):
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

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]

        image_path = Path(row["image_path"])
        attention_path = Path(row["attention_path"])

        image = Image.open(image_path).convert("RGB")
        attention = Image.open(attention_path).convert("L")

        image_tensor = self.image_transform(image)
        attention_tensor = self.attention_transform(attention)

        x = float(row["x"])
        y = float(row["y"])

        x_n = (x - self.x_min) / (self.x_max - self.x_min)
        y_n = (y - self.y_min) / (self.y_max - self.y_min)

        coord_label = torch.tensor([x_n, y_n], dtype=torch.float32)
        topo_label = torch.tensor(int(row["topo_label"]), dtype=torch.long)

        return image_tensor, attention_tensor, topo_label, coord_label
```

---

## 11. Code skeleton bắt buộc cho `preprocess_offline.py`

```python
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"image_path", "attention_path", "x", "y"}


def load_json(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(config, config_path):
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def validate_required_columns(df, name):
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing_columns)}"
        )


def preprocess_offline(config_path):
    config_path = Path(config_path)
    config = load_json(config_path)

    room_csv = Path(config["data"]["room_csv"])
    corridor_csv = Path(config["data"]["corridor_csv"])
    total_csv = Path(config["data"]["total_csv"])

    room_df = pd.read_csv(room_csv)
    corridor_df = pd.read_csv(corridor_csv)

    validate_required_columns(room_df, "room_poses.csv")
    validate_required_columns(corridor_df, "corridor_poses.csv")

    room_df = room_df.copy()
    corridor_df = corridor_df.copy()

    room_df["topo_label"] = int(config["labels"]["room"])
    corridor_df["topo_label"] = int(config["labels"]["corridor"])

    delta_x = float(config["coordinate_alignment"]["corridor"]["delta_x"])
    delta_y = float(config["coordinate_alignment"]["corridor"]["delta_y"])

    corridor_df["x"] = corridor_df["x"].astype(float) + delta_x
    corridor_df["y"] = corridor_df["y"].astype(float) + delta_y

    total_df = pd.concat([room_df, corridor_df], ignore_index=True)

    x_min = float(total_df["x"].min())
    x_max = float(total_df["x"].max())
    y_min = float(total_df["y"].min())
    y_max = float(total_df["y"].max())

    if x_max <= x_min:
        raise ValueError("Invalid x range: x_max must be greater than x_min.")

    if y_max <= y_min:
        raise ValueError("Invalid y range: y_max must be greater than y_min.")

    config["global_normalization"]["x_min"] = x_min
    config["global_normalization"]["x_max"] = x_max
    config["global_normalization"]["y_min"] = y_min
    config["global_normalization"]["y_max"] = y_max

    total_csv.parent.mkdir(parents=True, exist_ok=True)
    total_df.to_csv(total_csv, index=False)

    save_json(config, config_path)

    print(f"Saved total poses to: {total_csv}")
    print(f"Updated global_normalization in: {config_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    preprocess_offline(args.config)
```

---

## 12. Contract giữa Dataset và Model

Dataset trả về batch:

```python
image_tensor, attention_tensor, topo_label, coord_label
```

Sau khi qua DataLoader, batch có dạng:

```python
images, attention_maps, topo_labels, coord_labels = batch
```

Shape bắt buộc:

| Batch Tensor     |          Shape |
| ---------------- | -------------: |
| `images`         | `[B, 3, H, W]` |
| `attention_maps` | `[B, 1, H, W]` |
| `topo_labels`    |          `[B]` |
| `coord_labels`   |       `[B, 2]` |

Model sẽ được gọi trong `train.py` như sau:

```python
topo_logits, coord_pred = model(images, attention_maps)
```

> Dataset không được gọi model. Model không được đọc Dataset. `train.py` là nơi duy nhất kết nối hai module này.

---

## 13. Kiểm tra nhanh Dataset

AI agent phải đảm bảo đoạn kiểm tra sau chạy được:

```python
dataset = RobotLocalizationDataset(
    total_csv="data/processed/total_poses.csv",
    config_path="config.json"
)

image_tensor, attention_tensor, topo_label, coord_label = dataset[0]

assert image_tensor.shape[0] == 3
assert attention_tensor.shape[0] == 1
assert topo_label.dtype == torch.long
assert coord_label.shape == torch.Size([2])
assert coord_label.dtype == torch.float32
```

Kiểm tra giá trị Attention Map:

```python
assert attention_tensor.min() >= 0.0
assert attention_tensor.max() <= 1.0
```

Kiểm tra tọa độ normalized:

```python
assert coord_label[0] >= 0.0 and coord_label[0] <= 1.0
assert coord_label[1] >= 0.0 and coord_label[1] <= 1.0
```

---

## 14. Các lỗi triển khai phải tránh

| Lỗi                                                | Trạng thái        |
| -------------------------------------------------- | ----------------- |
| Cộng offset cho dữ liệu phòng                      | Không hợp lệ      |
| Không cộng offset cho dữ liệu hành lang            | Không hợp lệ      |
| Tính `global_normalization` trước khi cộng offset  | Không hợp lệ      |
| Không cập nhật `config.json` sau khi tính cực biên | Không hợp lệ      |
| Dataset đọc hai CSV thô thay vì `total_poses.csv`  | Không đúng đặc tả |
| Dataset trả về đường dẫn ảnh thay vì Tensor        | Không hợp lệ      |
| Attention Map bị normalize bằng ImageNet mean/std  | Không hợp lệ      |
| `topo_label` dùng FloatTensor                      | Không hợp lệ      |
| `coord_label` dùng tọa độ mét chưa normalized      | Không hợp lệ      |
| `utils/dataset.py` import ResNet hoặc MLP          | Không hợp lệ      |
| `preprocess_offline.py` định nghĩa model           | Không hợp lệ      |
| Dataset gọi YOLOv8 trong `__getitem__`             | Không hợp lệ      |

---

## 15. Tiêu chí hoàn thành module dữ liệu

Module dữ liệu được xem là hoàn thành khi thỏa mãn toàn bộ tiêu chí sau:

```text
1. preprocess_offline.py đọc được room_poses.csv và corridor_poses.csv.
2. Dữ liệu phòng được gán topo_label = 0.
3. Dữ liệu hành lang được gán topo_label = 1.
4. Dữ liệu hành lang được cộng delta_x và delta_y từ config.json.
5. Hai tập dữ liệu được gộp thành total_poses.csv.
6. Cực biên x_min, x_max, y_min, y_max được tính từ total_poses.csv sau khi alignment.
7. global_normalization trong config.json được cập nhật tự động.
8. RobotLocalizationDataset đọc total_poses.csv và config.json.
9. __len__ trả về đúng số lượng frame.
10. __getitem__ trả đúng 4 Tensor: image_tensor, attention_tensor, topo_label, coord_label.
11. image_tensor có shape [3, H, W] và được chuẩn hóa ImageNet.
12. attention_tensor có shape [1, H, W] và giá trị trong [0, 1].
13. topo_label là LongTensor.
14. coord_label là FloatTensor [2] đã chuẩn hóa Min-Max.
15. Module dữ liệu không chứa bất kỳ logic mạng nơ-ron nào.
```

---

## 16. Kết luận kỹ thuật

Data Pipeline Module biểu diễn ánh xạ:

```text
room_poses.csv + corridor_poses.csv + config.json
        ↓
coordinate alignment + topo labeling + global normalization
        ↓
total_poses.csv + updated config.json
        ↓
RobotLocalizationDataset
        ↓
image_tensor, attention_tensor, topo_label, coord_label
```

Module này chỉ xử lý dữ liệu. Nó không được biết model sử dụng ResNet50, Feature-level Attention hay MLP heads. Toàn bộ trách nhiệm kiến trúc mạng thuộc `models/architecture.py`, còn trách nhiệm kết nối Dataset với Model thuộc `train.py`.

Đặc tả này là nguồn tham chiếu bắt buộc cho AI agent khi lập trình `preprocess_offline.py` và `utils/dataset.py`.
