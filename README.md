# Hệ thống định vị phân cấp RGB với Feature-level Attention nhận thức vùng động

> **Mục tiêu kỹ thuật:** Xây dựng hệ thống định vị robot di động trong nhà từ ảnh RGB, có khả năng ước lượng tọa độ thực tế ((x, y)) và phân loại nhãn khu vực topo trong môi trường có người di chuyển.

Dự án này được thiết kế nghiêm ngặt theo nguyên lý **Separation of Concerns - Phân tách mối quan tâm**. Toàn bộ hệ thống được chia thành hai sub-module độc lập hoàn toàn về mặt vật lý và logic:

| Sub-module                    | Vai trò chính                                                                                    | Không được chứa                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| **Data Pipeline Module**      | Chuẩn bị dữ liệu, đồng bộ hệ quy chiếu, chuẩn hóa tọa độ, nạp RGB + Attention Map + Ground Truth | Không định nghĩa kiến trúc mạng                            |
| **Model Architecture Module** | Định nghĩa ResNet50 backbone, Feature-level Attention và Multi-task MLP Heads                    | Không đọc file, không xử lý CSV, không xử lý offset tọa độ |

File `train.py` đóng vai trò **Master Controller**, chịu trách nhiệm kết nối hai module trên trong quá trình huấn luyện.

---

## 1. System Overview

Hệ thống giải quyết bài toán định vị robot di động trong nhà bằng ảnh RGB đầu vào. Với mỗi ảnh tại thời điểm $t$, mô hình cần dự đoán đồng thời tọa độ vị trí thực tế của robot:

$$
\hat{\mathbf{p}}_t = [\hat{x}_t, \hat{y}_t]^T
$$

và nhãn khu vực topo:

$$
\hat{c}_t \in \{\text{Room}, \text{Corridor}\}
$$

Trong đó, $\hat{x}_t$ và $\hat{y}_t$ là tọa độ robot trong hệ quy chiếu bản đồ, còn $\hat{c}_t$ biểu diễn robot đang ở **Phòng** hay **Hành lang**.

### Giải pháp cốt lõi

Hệ thống sử dụng YOLOv8 ở giai đoạn offline để phát hiện vùng ảnh chứa đối tượng động, đặc biệt là người di chuyển. Kết quả phát hiện này được chuyển thành **Attention Map** dạng ảnh xám một kênh. Trong giai đoạn huấn luyện chính, ảnh RGB gốc được đưa vào ResNet50 để trích xuất Feature Map sâu. Attention Map sau đó được nội suy về cùng kích thước với Feature Map và được nhân trực tiếp tại cấp độ đặc trưng.

Pipeline tổng quát:

```text
RGB Image
   │
   ├── Offline YOLOv8 Preprocessing
   │        └── Attention Map
   │
   └── ResNet50 Backbone
            │
            └── Feature Map [B, 2048, H', W']
                    │
                    └── Feature-level Attention Fusion
                            │
                            └── Multi-task MLP Heads
                                    ├── Topological Classification Head
                                    └── Coordinate Regression Head
```

> YOLOv8 không nằm trong vòng lặp huấn luyện chính của mạng định vị. YOLOv8 chỉ được sử dụng offline để tạo Attention Map, giúp giảm chi phí tính toán và giữ pipeline huấn luyện ổn định.

---

## 2. Project Directory Structure

```text
rgb_dynamic_attention_localization/
│
├── README.md
├── config.json
├── train.py
├── preprocess_offline.py
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── room/
│   │   │   ├── images/
│   │   │   │   ├── room_000001.jpg
│   │   │   │   ├── room_000002.jpg
│   │   │   │   └── ...
│   │   │   └── poses.csv
│   │   │
│   │   └── corridor/
│   │       ├── images/
│   │       │   ├── corridor_000001.jpg
│   │       │   ├── corridor_000002.jpg
│   │       │   └── ...
│   │       └── poses.csv
│   │
│   ├── processed/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   │   ├── sample_000001.jpg
│   │   │   │   └── ...
│   │   │   ├── attention_maps/
│   │   │   │   ├── sample_000001.png
│   │   │   │   └── ...
│   │   │   └── poses.csv
│   │   │
│   │   ├── val/
│   │   │   ├── images/
│   │   │   ├── attention_maps/
│   │   │   └── poses.csv
│   │   │
│   │   └── test/
│   │       ├── images/
│   │       ├── attention_maps/
│   │       └── poses.csv
│   │
│   └── metadata/
│       ├── normalization.json
│       ├── label_map.json
│       └── split_info.json
│
├── models/
│   ├── __init__.py
│   └── architecture.py
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── losses.py
│   ├── metrics.py
│   └── io_utils.py
│
├── checkpoints/
│   ├── best_model.pth
│   ├── last_model.pth
│   └── logs/
│       ├── train_log.csv
│       └── metrics.json
│
└── notebooks/
    ├── colab_train.ipynb
    └── visualization.ipynb
```

### Vai trò từng thành phần

| Đường dẫn                          | Vai trò kỹ thuật                                                                                                      |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `config.json`                      | Lưu toàn bộ tham số cấu hình: đường dẫn dữ liệu, batch size, learning rate, hệ số loss, offset tọa độ, kích thước ảnh |
| `preprocess_offline.py`            | Chạy YOLOv8 offline để tạo Attention Map; đồng bộ hệ quy chiếu; tạo dữ liệu processed                                 |
| `train.py`                         | Master Controller: đọc config, tạo Dataset/DataLoader, khởi tạo model, tính loss, tối ưu và lưu checkpoint            |
| `data/raw/`                        | Dữ liệu gốc từ quá trình thu thập, tách riêng phòng và hành lang                                                      |
| `data/processed/`                  | Dữ liệu đã chuẩn hóa cấu trúc, đã có ảnh RGB, Attention Map và `poses.csv` tương ứng                                  |
| `data/metadata/normalization.json` | Lưu (x_{min}, x_{max}, y_{min}, y_{max}) để chuẩn hóa và giải chuẩn hóa tọa độ                                        |
| `data/metadata/label_map.json`     | Ánh xạ nhãn topo, ví dụ `Room -> 0`, `Corridor -> 1`                                                                  |
| `models/architecture.py`           | Định nghĩa toàn bộ mạng nơ-ron; không chứa logic đọc file                                                             |
| `utils/dataset.py`                 | Định nghĩa PyTorch Dataset; chịu trách nhiệm nạp RGB, Attention Map và ground truth                                   |
| `utils/transforms.py`              | Chứa các phép resize, normalize ảnh RGB và Attention Map                                                              |
| `utils/losses.py`                  | Định nghĩa hàm loss đa nhiệm nếu muốn tách khỏi `train.py`                                                            |
| `utils/metrics.py`                 | Tính accuracy topo, MAE, RMSE, sai số Euclidean                                                                       |
| `checkpoints/`                     | Lưu trọng số `.pth` và log huấn luyện                                                                                 |

---

## 3. Data Pipeline Specification

Data Pipeline Module chịu trách nhiệm biến dữ liệu thô thành dữ liệu có thể đưa trực tiếp vào mô hình. Module này xử lý ảnh RGB, Attention Map, nhãn topo và tọa độ ground truth. Module này không được định nghĩa bất kỳ lớp mạng nơ-ron nào.

### 3.1. Input / Output của Data Pipeline

Đầu vào của module là dữ liệu gốc được thu từ robot:

```text
RGB image + pose ground truth + scene label
```

Đầu ra của module là một sample huấn luyện có cấu trúc:

```python
{
    "image": Tensor[3, H, W],
    "attention": Tensor[1, H, W],
    "topo_label": LongTensor[],
    "coord": Tensor[2]
}
```

| Thành phần   |       Shape | Ý nghĩa                               |
| ------------ | ----------: | ------------------------------------- |
| `image`      | `[3, H, W]` | Ảnh RGB gốc đã resize và normalize    |
| `attention`  | `[1, H, W]` | Ảnh xám Attention Map tạo từ YOLOv8   |
| `topo_label` |      scalar | Nhãn topo: `0 = Room`, `1 = Corridor` |
| `coord`      |       `[2]` | Tọa độ normalized ([x_n, y_n])        |

---

### 3.2. Đồng bộ hệ quy chiếu giữa phòng và hành lang

Trong quá trình thu thập dữ liệu, dữ liệu phòng và dữ liệu hành lang có thể được ghi nhận trong các hệ quy chiếu cục bộ khác nhau. Để huấn luyện một mô hình hồi quy thống nhất, toàn bộ tọa độ phải được đưa về cùng một hệ quy chiếu bản đồ.

Với dữ liệu hành lang, ta áp dụng phép cộng offset:

$$
x^{global} = x^{corridor} + \Delta x
$$

$$
y^{global} = y^{corridor} + \Delta y
$$

Trong đó:

| Ký hiệu                      | Ý nghĩa                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| $x^{corridor}, y^{corridor}$ | Tọa độ ban đầu của mẫu hành lang                              |
| $\Delta x, \Delta y$         | Khoảng lệch giữa hệ quy chiếu hành lang và hệ quy chiếu phòng |
| $x^{global}, y^{global}$     | Tọa độ sau khi đưa về hệ quy chiếu chung                      |

Thông số offset được lưu trong `config.json`:

```json
{
  "coordinate_offset": {
    "corridor": {
      "dx": 14.45,
      "dy": 2.45
    }
  }
}
```

> Quy tắc bắt buộc: offset chỉ được xử lý trong `preprocess_offline.py` hoặc `utils/dataset.py`. File `models/architecture.py` không được biết đến khái niệm offset.

---

### 3.3. Chuẩn hóa tọa độ hình học

Nhánh hồi quy không trực tiếp học tọa độ mét ban đầu. Tọa độ thực tế được chuẩn hóa về đoạn $[0,1]$ bằng Min-Max Normalization:

$$
x_n = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

$$
y_n = \frac{y - y_{min}}{y_{max} - y_{min}}
$$

Trong đó:

| Ký hiệu            | Ý nghĩa                                                |
| ------------------ | ------------------------------------------------------ |
| $x, y$             | Tọa độ thực tế sau khi đã đồng bộ hệ quy chiếu         |
| $x_n, y_n$         | Tọa độ normalized dùng để huấn luyện                   |
| $x_{min}, x_{max}$ | Biên nhỏ nhất và lớn nhất của trục $x$ trong tập train |
| $y_{min}, y_{max}$ | Biên nhỏ nhất và lớn nhất của trục $y$ trong tập train |

Giải chuẩn hóa khi đánh giá:

$$
\hat{x} = \hat{x}_n(x_{max} - x_{min}) + x_{min}
$$

$$
\hat{y} = \hat{y}_n(y_{max} - y_{min}) + y_{min}
$$

Các tham số chuẩn hóa được lưu trong:

```text
data/metadata/normalization.json
```

Ví dụ:

```json
{
  "x_min": 0.0,
  "x_max": 20.0,
  "y_min": 0.0,
  "y_max": 8.0
}
```

---

### 3.4. Tạo Attention Map offline bằng YOLOv8

YOLOv8 được sử dụng offline để xác định vùng ảnh có đối tượng động. Kết quả phát hiện được chuyển thành ảnh xám một kênh, gọi là **Attention Map**.

Với bản đồ vùng động $M_t$, Attention Map $A_t$ được tính bởi:

$$
A_t(u,v)=1-\alpha M_t(u,v)
$$

Trong đó:

| Ký hiệu    | Ý nghĩa                             |
| ---------- | ----------------------------------- |
| $M_t(u,v)$ | Bản đồ vùng động tại pixel $(u,v)$  |
| $A_t(u,v)$ | Giá trị attention tại pixel $(u,v)$ |
| $\alpha$   | Hệ số suy giảm vùng động            |

Quy ước giá trị:

| Pixel trong ảnh | Giá trị trên Attention Map | Ý nghĩa                   |
| --------------- | -------------------------: | ------------------------- |
| Vùng tĩnh       |                      `1.0` | Giữ nguyên đặc trưng      |
| Vùng động       |                `1 - alpha` | Suy giảm đặc trưng        |
| Không xác định  |                      `1.0` | Mặc định xem là vùng tĩnh |

Attention Map được lưu dưới dạng ảnh xám `.png`:

```text
data/processed/train/attention_maps/sample_000001.png
```

> YOLOv8 chỉ chạy trong `preprocess_offline.py`. Trong vòng lặp huấn luyện chính, `train.py` chỉ đọc Attention Map đã có sẵn từ ổ đĩa.

---

### 3.5. Cấu trúc `poses.csv`

Mỗi thư mục `train/`, `val/`, `test/` có một file `poses.csv` chứa mapping giữa ảnh, Attention Map và nhãn ground truth.

```csv
image_path,attention_path,topo_label,x,y,x_norm,y_norm
images/sample_000001.jpg,attention_maps/sample_000001.png,0,1.25,2.40,0.0625,0.3000
images/sample_000002.jpg,attention_maps/sample_000002.png,1,15.80,3.10,0.7900,0.3875
```

| Cột                | Ý nghĩa                                     |
| ------------------ | ------------------------------------------- |
| `image_path`       | Đường dẫn tương đối tới ảnh RGB             |
| `attention_path`   | Đường dẫn tương đối tới Attention Map       |
| `topo_label`       | Nhãn topo dạng số                           |
| `x`, `y`           | Tọa độ thực tế sau khi đồng bộ hệ quy chiếu |
| `x_norm`, `y_norm` | Tọa độ normalized dùng cho huấn luyện       |

---

### 3.6. PyTorch Dataset & DataLoader

`utils/dataset.py` chỉ làm ba nhiệm vụ:

```text
1. Đọc poses.csv
2. Nạp ảnh RGB và Attention Map
3. Trả về tensor đúng định dạng cho model
```

Giao diện đề xuất:

```python
class RGBAttentionLocalizationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, image_transform=None, attention_transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
```

Output của `__getitem__()`:

```python
image, attention, topo_label, coord
```

Trong đó:

```python
image.shape     == [3, H, W]
attention.shape == [1, H, W]
topo_label      == torch.long
coord.shape     == [2]
```

> Dataset không được gọi model. Dataset không được tính loss. Dataset không được chứa logic kiến trúc mạng.

---

## 4. Model Architecture Specification

Model Architecture Module chỉ chịu trách nhiệm định nghĩa phép tính trên tensor. Module này không được đọc ảnh, không đọc CSV, không xử lý offset và không biết dữ liệu nằm ở thư mục nào.

File trung tâm:

```text
models/architecture.py
```

---

### 4.1. Input / Output của model

Model nhận hai tensor:

$$
\mathbf{I} \in \mathbb{R}^{B \times 3 \times H \times W}
$$

$$
\mathbf{A} \in \mathbb{R}^{B \times 1 \times H \times W}
$$

Trong đó:

| Tensor      |          Shape | Ý nghĩa             |
| ----------- | -------------: | ------------------- |
| `image`     | `[B, 3, H, W]` | Batch ảnh RGB       |
| `attention` | `[B, 1, H, W]` | Batch Attention Map |

Model trả về:

```python
topo_logits, coord_pred
```

| Output        |    Shape | Ý nghĩa                                           |
| ------------- | -------: | ------------------------------------------------- |
| `topo_logits` | `[B, 2]` | Logits cho hai lớp topo: Room/Corridor            |
| `coord_pred`  | `[B, 2]` | Tọa độ normalized dự đoán $[\hat{x}_n,\hat{y}_n]$ |

---

### 4.2. ResNet50 Backbone

Ảnh RGB được đưa qua ResNet50 đến hết `layer4`. Phần fully-connected gốc của ResNet50 bị loại bỏ.

$$
F = \phi_{ResNet50}(I)
$$

Với input batch $I$, Feature Map đầu ra có dạng:

$$
F \in \mathbb{R}^{B \times 2048 \times H' \times W'}
$$

Trong đó:

| Ký hiệu  | Ý nghĩa                                      |
| -------- | -------------------------------------------- |
| $B$      | Batch size                                   |
| $2048$   | Số kênh đặc trưng đầu ra của ResNet50 layer4 |
| $H', W'$ | Kích thước không gian sau downsampling       |

Ví dụ với ảnh đầu vào (224 \times 224), Feature Map thường có kích thước:

```text
[B, 2048, 7, 7]
```

---

### 4.3. Feature-level Attention Fusion

Attention Map ban đầu có shape:

$$
A \in \mathbb{R}^{B \times 1 \times H \times W}
$$

Trong khi Feature Map từ ResNet50 có shape:

$$
F \in \mathbb{R}^{B \times 2048 \times H' \times W'}
$$

Do đó, Attention Map phải được nội suy về cùng kích thước không gian với Feature Map:

$$
A' = \text{Interpolate}(A, H', W')
$$

với:

$$
A' \in \mathbb{R}^{B \times 1 \times H' \times W'}
$$

Phép nội suy sử dụng Bilinear Interpolation:

```python
attention_resized = F.interpolate(
    attention,
    size=feature_map.shape[-2:],
    mode="bilinear",
    align_corners=False
)
```

Sau đó thực hiện phép nhân từng phần tử:

$$
F_{att} = F \odot A'
$$

Nhờ broadcasting, $A'$ được nhân lên toàn bộ 2048 kênh của Feature Map.

| Thành phần          |               Shape |
| ------------------- | ------------------: |
| `feature_map`       | `[B, 2048, H', W']` |
| `attention_resized` |    `[B, 1, H', W']` |
| `attended_feature`  | `[B, 2048, H', W']` |

> Cơ chế này được gọi là **Feature-level Attention**, vì Attention Map không được nhân trực tiếp vào ảnh RGB mà được nhân vào đặc trưng sâu sau ResNet50.

---

### 4.4. Global Average Pooling

Sau khi áp dụng attention, Feature Map được đưa qua Global Average Pooling:

$$
z = GAP(F_{att})
$$

Kết quả:

$$
z \in \mathbb{R}^{B \times 2048}
$$

Vector $z$ là biểu diễn đặc trưng chung cho hai nhiệm vụ: phân loại topo và hồi quy tọa độ.

---

### 4.5. Multi-task MLP Heads

Mạng sử dụng hai nhánh MLP chạy song song.

#### 4.5.1. Topological Classification Head

Nhánh phân loại topo nhận vector $z$ và trả về logits:

$$
\hat{q} = h_{topo}(z)
$$

Cấu trúc đề xuất:

```text
Linear(2048 → 512)
BatchNorm1d(512)
ReLU
Dropout(p=0.3)
Linear(512 → 128)
BatchNorm1d(128)
ReLU
Dropout(p=0.2)
Linear(128 → 2)
```

Output:

```text
topo_logits: [B, 2]
```

Loss sử dụng:

$$
\mathcal{L}_{topo} = CE(c, \hat{q})
$$

Trong đó $CE$ là Cross-Entropy Loss.

---

#### 4.5.2. Coordinate Regression Head

Nhánh hồi quy tọa độ nhận cùng vector $z$ và trả về tọa độ normalized:

$$
\hat{\mathbf{p}}_n = h_{coord}(z)
$$

Cấu trúc đề xuất:

```text
Linear(2048 → 512)
BatchNorm1d(512)
ReLU
Dropout(p=0.3)
Linear(512 → 128)
BatchNorm1d(128)
ReLU
Dropout(p=0.2)
Linear(128 → 2)
Sigmoid
```

Output:

```text
coord_pred: [B, 2]
```

Loss sử dụng:

$$
\mathcal{L}_{coord} = SmoothL1(\mathbf{p}_n, \hat{\mathbf{p}}_n)
$$

Trong đó $\mathbf{p}_n=[x_n,y_n]^T$ là tọa độ normalized ground truth.

> Lớp `Sigmoid` ở cuối nhánh hồi quy được dùng vì tọa độ ground truth đã được chuẩn hóa về đoạn $[0,1]$.

---

### 4.6. Quy tắc bắt buộc của `models/architecture.py`

File `models/architecture.py` chỉ được chứa:

```text
Tensor input
Tensor operation
Neural network layers
Forward pass
```

File này không được chứa:

```text
CSV parsing
Image path handling
Coordinate offset logic
Min-Max fitting
Data split logic
YOLOv8 inference
Checkpoint saving
Training loop
```

Ranh giới module:

| Logic                   | File được phép chứa                             |
| ----------------------- | ----------------------------------------------- |
| Đọc CSV                 | `utils/dataset.py`                              |
| Đọc ảnh RGB             | `utils/dataset.py`                              |
| Đọc Attention Map       | `utils/dataset.py`                              |
| Offset tọa độ           | `preprocess_offline.py` hoặc `utils/dataset.py` |
| Chuẩn hóa tọa độ        | `preprocess_offline.py` hoặc `utils/dataset.py` |
| ResNet50 backbone       | `models/architecture.py`                        |
| Feature-level Attention | `models/architecture.py`                        |
| MLP heads               | `models/architecture.py`                        |
| Loss và optimizer       | `train.py` hoặc `utils/losses.py`               |
| Checkpoint              | `train.py`                                      |

---

## 5. Unified Execution Flow

File `train.py` đóng vai trò điều phối trung tâm. Nó không xử lý chi tiết dữ liệu ở mức thấp và cũng không định nghĩa kiến trúc mạng. Nhiệm vụ của `train.py` là kết nối Data Pipeline Module với Model Architecture Module.

```text
config.json
    │
    ├── utils/dataset.py
    │       └── DataLoader
    │
    ├── models/architecture.py
    │       └── DynamicAwareLocalizationNet
    │
    └── train.py
            ├── forward pass
            ├── multi-task loss
            ├── backward
            ├── optimizer step
            └── checkpoint saving
```

---

### 5.1. Luồng huấn luyện

```text
Start train.py
    │
    ├── Load config.json
    │
    ├── Check processed dataset
    │       ├── If missing Attention Maps → call preprocess_offline.py
    │       └── If existing → continue
    │
    ├── Create Dataset and DataLoader
    │
    ├── Initialize model from models/architecture.py
    │
    ├── For each epoch:
    │       ├── Load batch: image, attention, topo_label, coord
    │       ├── Forward: topo_logits, coord_pred = model(image, attention)
    │       ├── Compute topo loss
    │       ├── Compute coordinate loss
    │       ├── Compute total loss
    │       ├── Backpropagation
    │       └── Update parameters
    │
    ├── Validate model
    │
    ├── Save best checkpoint
    │
    └── End
```

---

### 5.2. Hàm lỗi đa nhiệm

Tổng loss được định nghĩa:

$$
\mathcal{L}_{total} = \mathcal{L}_{topo} + \gamma \mathcal{L}_{coord}
$$

Trong đó:

| Ký hiệu               | Ý nghĩa                                        |
| --------------------- | ---------------------------------------------- |
| $\mathcal{L}_{topo}$  | Cross-Entropy Loss cho phân loại Room/Corridor |
| $\mathcal{L}_{coord}$ | Smooth L1 Loss cho hồi quy tọa độ normalized   |
| $\gamma$              | Hệ số cân bằng giữa hai nhiệm vụ               |

Cài đặt trong `config.json`:

```json
{
  "loss": {
    "gamma": 0.5
  }
}
```

Nếu muốn ưu tiên phân loại topo hơn hồi quy tọa độ, chọn:

$$
0 < \gamma < 1
$$

---

### 5.3. Quy trình tự động hóa trên Google Colab

Khi triển khai trên Google Colab, `train.py` nên tự động hóa các bước sau:

```text
Google Drive
    │
    ├── raw_dataset.zip
    ├── processed_dataset.zip
    └── checkpoints/
            │
            └── best_model.pth
```

Quy trình đề xuất:

```text
1. Mount Google Drive
2. Copy dataset zip từ Drive vào SSD nội bộ của Colab
3. Giải nén dataset vào /content/project/data/
4. Kiểm tra data/processed/
5. Nếu thiếu Attention Maps:
       gọi preprocess_offline.py
6. Khởi tạo Dataset/DataLoader
7. Huấn luyện model
8. Lưu checkpoint tạm vào /content/project/checkpoints/
9. Đồng bộ checkpoint ngược về Google Drive
```

> Quy tắc hiệu năng: không huấn luyện trực tiếp trên file zip hoặc thư mục Google Drive. Dữ liệu phải được copy và giải nén vào SSD nội bộ của Colab trước khi huấn luyện.

---

## 6. Configuration File

`config.json` là điểm cấu hình duy nhất của hệ thống.

```json
{
  "project": {
    "name": "rgb_dynamic_attention_localization",
    "seed": 42,
    "device": "cuda"
  },

  "data": {
    "root": "data/processed",
    "train_csv": "data/processed/train/poses.csv",
    "val_csv": "data/processed/val/poses.csv",
    "test_csv": "data/processed/test/poses.csv",
    "image_size": [224, 224],
    "num_workers": 4
  },

  "coordinate_offset": {
    "corridor": {
      "dx": 14.45,
      "dy": 2.45
    }
  },

  "normalization": {
    "metadata_path": "data/metadata/normalization.json"
  },

  "labels": {
    "Room": 0,
    "Corridor": 1,
    "num_classes": 2
  },

  "attention": {
    "alpha": 0.7,
    "map_mode": "grayscale",
    "apply_level": "feature"
  },

  "model": {
    "backbone": "resnet50",
    "pretrained": true,
    "feature_dim": 2048,
    "dropout_topo": 0.3,
    "dropout_coord": 0.3
  },

  "training": {
    "epochs": 200,
    "batch_size": 16,
    "learning_rate": 0.001,
    "weight_decay": 0.0001
  },

  "loss": {
    "topo_loss": "cross_entropy",
    "coord_loss": "smooth_l1",
    "gamma": 0.5
  },

  "checkpoint": {
    "save_dir": "checkpoints",
    "save_best": true,
    "monitor": "val_total_loss"
  }
}
```

---

## 7. Interface Contract giữa hai sub-modules

Để đảm bảo hai module độc lập, cần duy trì contract cố định giữa Data Pipeline và Model Architecture.

### 7.1. Contract đầu ra của Dataset

`utils/dataset.py` phải trả về:

```python
image, attention, topo_label, coord
```

với kiểu dữ liệu:

```python
image: torch.FloatTensor       # [B, 3, H, W]
attention: torch.FloatTensor   # [B, 1, H, W]
topo_label: torch.LongTensor   # [B]
coord: torch.FloatTensor       # [B, 2]
```

### 7.2. Contract đầu vào của Model

`models/architecture.py` phải nhận:

```python
topo_logits, coord_pred = model(image, attention)
```

và trả về:

```python
topo_logits: torch.FloatTensor # [B, 2]
coord_pred: torch.FloatTensor  # [B, 2]
```

### 7.3. Contract tính loss trong `train.py`

`train.py` tính loss theo đúng output contract:

```python
loss_topo = criterion_topo(topo_logits, topo_label)
loss_coord = criterion_coord(coord_pred, coord)
loss_total = loss_topo + gamma * loss_coord
```

---

## 8. Pseudo-code kiến trúc mạng

```python
class DynamicAwareLocalizationNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = ResNet50_without_fc()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.topo_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.coord_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, image, attention):
        feature = self.backbone(image)

        attention_resized = F.interpolate(
            attention,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        feature_att = feature * attention_resized

        z = self.gap(feature_att)
        z = torch.flatten(z, 1)

        topo_logits = self.topo_head(z)
        coord_pred = self.coord_head(z)

        return topo_logits, coord_pred
```

---

## 9. Pseudo-code DataLoader

```python
class RGBAttentionLocalizationDataset(Dataset):
    def __init__(self, root_dir, csv_file, image_transform=None, attention_transform=None):
        self.root_dir = root_dir
        self.records = pd.read_csv(csv_file)
        self.image_transform = image_transform
        self.attention_transform = attention_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]

        image_path = os.path.join(self.root_dir, row["image_path"])
        attention_path = os.path.join(self.root_dir, row["attention_path"])

        image = load_rgb_image(image_path)
        attention = load_grayscale_image(attention_path)

        if self.image_transform:
            image = self.image_transform(image)

        if self.attention_transform:
            attention = self.attention_transform(attention)

        topo_label = torch.tensor(row["topo_label"], dtype=torch.long)
        coord = torch.tensor([row["x_norm"], row["y_norm"]], dtype=torch.float32)

        return image, attention, topo_label, coord
```

---

## 10. Training Command

Huấn luyện trên máy local:

```bash
python train.py --config config.json
```

Chạy tiền xử lý offline:

```bash
python preprocess_offline.py --config config.json
```

Huấn luyện trên Google Colab sau khi mount Drive:

```bash
python train.py \
    --config config.json \
    --colab_mode true \
    --drive_checkpoint_dir /content/drive/MyDrive/rgb_localization/checkpoints
```

---

## 11. Evaluation Metrics

Các chỉ số đánh giá chính:

| Nhóm đánh giá  | Metric           | Công thức / Ý nghĩa                       |
| -------------- | ---------------- | ----------------------------------------- |
| Phân loại topo | Accuracy         | Tỷ lệ dự đoán đúng Room/Corridor          |
| Phân loại topo | Confusion Matrix | Kiểm tra nhầm lẫn giữa Phòng và Hành lang |
| Hồi quy tọa độ | MAE-x            | Sai số tuyệt đối trung bình theo trục $x$ |
| Hồi quy tọa độ | MAE-y            | Sai số tuyệt đối trung bình theo trục $y$ |
| Hồi quy tọa độ | Euclidean Error  | $\sqrt{(\hat{x}-x)^2+(\hat{y}-y)^2}$      |
| Hồi quy tọa độ | RMSE             | Sai số bình phương trung bình căn bậc hai |

Sai số Euclidean:

$$
e_i=\sqrt{(\hat{x}_i-x_i)^2+(\hat{y}_i-y_i)^2}
$$

RMSE:

$$
RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}
$$

---

## 12. Architectural Constraints

Các ràng buộc sau phải được giữ trong toàn bộ quá trình phát triển:

| Ràng buộc                                   | Lý do                                         |
| ------------------------------------------- | --------------------------------------------- |
| `models/architecture.py` không đọc file     | Đảm bảo model độc lập với dữ liệu             |
| `utils/dataset.py` không chứa layer mạng    | Đảm bảo data module không phụ thuộc kiến trúc |
| YOLOv8 chỉ chạy offline                     | Giảm tải cho training loop                    |
| Attention áp dụng ở feature-level           | Giữ ảnh RGB gốc, chỉ suy giảm đặc trưng sâu   |
| Tọa độ hồi quy phải normalized              | Ổn định quá trình học                         |
| Checkpoint phải lưu được cả model và config | Đảm bảo tái lập thực nghiệm                   |

---

## 13. Expected Checkpoint Format

Checkpoint `.pth` nên có cấu trúc:

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_loss": best_val_loss,
    "config": config,
    "normalization": normalization_metadata,
    "label_map": label_map
}
```

Việc lưu cả `normalization` và `label_map` trong checkpoint giúp mô hình có thể suy luận độc lập sau huấn luyện.

---

## 14. Inference Flow

Trong suy luận, hệ thống nhận ảnh RGB và Attention Map đã được tạo tương ứng.

```text
RGB image + Attention Map
        │
        └── model(image, attention)
                ├── topo_logits
                └── coord_pred
```

Nhãn topo:

$$
\hat{c} = \arg\max(\hat{q})
$$

Tọa độ normalized được giải chuẩn hóa:

$$
\hat{x} = \hat{x}_n(x_{max} - x_{min}) + x_{min}
$$

$$
\hat{y} = \hat{y}_n(y_{max} - y_{min}) + y_{min}
$$

Output cuối cùng:

```python
{
    "topo_label": "Room",
    "x": 1.25,
    "y": 2.40
}
```

---

## 15. Development Rule

> Mọi thay đổi mã nguồn phải giữ nguyên nguyên lý phân tách module. Nếu một đoạn code vừa xử lý dữ liệu vừa định nghĩa model, đoạn code đó phải được tách lại trước khi merge.

Quy tắc kiểm tra nhanh:

| Câu hỏi                                                   | Nếu câu trả lời là “có” thì xử lý          |
| --------------------------------------------------------- | ------------------------------------------ |
| `architecture.py` có đọc CSV không?                       | Sai kiến trúc                              |
| `dataset.py` có khai báo Linear/Conv/ResNet không?        | Sai kiến trúc                              |
| `train.py` có xử lý từng pixel attention không?           | Nên chuyển sang preprocessing hoặc dataset |
| YOLOv8 có chạy trong mỗi training step không?             | Sai thiết kế                               |
| Coordinate loss có dùng tọa độ mét chưa normalized không? | Cần kiểm tra lại chuẩn hóa                 |

---

## 16. Summary

Dự án này triển khai một hệ thống định vị phân cấp dựa trên ảnh RGB cho robot di động trong nhà. Thiết kế chính của hệ thống là kết hợp ảnh RGB với Attention Map nhận thức vùng động, sau đó áp dụng attention tại cấp độ Feature Map của ResNet50. Mạng định vị sử dụng hai nhánh MLP chạy song song để thực hiện đồng thời phân loại khu vực topo và hồi quy tọa độ.

Kiến trúc phần mềm được chia thành hai module độc lập: **Data Pipeline Module** và **Model Architecture Module**. File `train.py` chỉ đóng vai trò điều phối, giúp hệ thống dễ mở rộng, dễ kiểm thử và phù hợp với yêu cầu triển khai thực nghiệm cho bài báo.
