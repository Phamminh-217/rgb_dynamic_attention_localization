# MODEL_SPECIFICATION.md

# Đặc tả kiến trúc mạng sâu cho `models/architecture.py`

> Tài liệu này là đặc tả kỹ thuật bắt buộc cho AI agent khi lập trình module mạng nơ-ron `models/architecture.py`.
>
> Module này chỉ được phép định nghĩa phép tính trên Tensor. Không được chứa bất kỳ logic đọc file, xử lý CSV, xử lý thư mục, chia tập dữ liệu, xử lý offset, chuẩn hóa từ metadata, gọi YOLOv8, tính loss hoặc training loop.

---

## 1. Mục tiêu của module

`models/architecture.py` định nghĩa mạng định vị thị giác phân cấp dựa trên ảnh RGB kết hợp cơ chế **Feature-level Attention** nhận thức vùng động.

Mạng nhận hai Tensor độc lập:

[
images \in \mathbb{R}^{B \times 3 \times H \times W}
]

[
attention_maps \in \mathbb{R}^{B \times 1 \times H \times W}
]

và trả về hai Tensor dự đoán:

[
topo_logits \in \mathbb{R}^{B \times C}
]

[
coord_pred \in \mathbb{R}^{B \times 2}
]

Trong đó:

| Ký hiệu       | Ý nghĩa                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------- |
| (B)           | Batch size                                                                                   |
| (H, W)        | Kích thước không gian của ảnh đầu vào                                                        |
| (C)           | Số lớp topo, ký hiệu trong code là `num_classes_topo`                                        |
| `topo_logits` | Raw logits cho bài toán phân loại topo                                                       |
| `coord_pred`  | Tensor dự đoán tọa độ liên tục `[x, y]` hoặc `[x_norm, y_norm]`, tùy theo nhãn từ DataLoader |

---

## 2. Ràng buộc phạm vi bắt buộc

`models/architecture.py` chỉ được chứa:

```text
- Import torch, torch.nn, torch.nn.functional
- Import torchvision.models nếu dùng ResNet50 pretrained
- Định nghĩa class mạng
- Định nghĩa ResNet50 backbone đã bỏ avgpool và fc
- Định nghĩa Feature-level Attention Fusion
- Định nghĩa Global Average Pooling
- Định nghĩa hai MLP heads
- Định nghĩa forward(images, attention_maps)
```

`models/architecture.py` không được chứa:

```text
- Đọc CSV
- Đọc ảnh RGB từ ổ cứng
- Đọc Attention Map từ ổ cứng
- Xử lý đường dẫn thư mục
- Xử lý offset tọa độ
- Chuẩn hóa Min-Max
- Giải chuẩn hóa tọa độ
- Tạo Dataset hoặc DataLoader
- Gọi YOLOv8
- Tính loss
- Training loop
- Optimizer
- Scheduler
- Lưu hoặc tải checkpoint
```

> Quy tắc cốt lõi: model không biết dữ liệu đến từ đâu. Model chỉ nhận Tensor sạch và trả về Tensor dự đoán.

---

## 3. Interface bắt buộc của model

### 3.1. Tên class chuẩn

```python
DynamicAwareLocalizationNet
```

### 3.2. Hàm khởi tạo

```python
def __init__(
    self,
    num_classes_topo: int,
    pretrained: bool = True,
    dropout_p: float = 0.3,
):
    ...
```

Trong đó:

| Tham số            | Kiểu    | Ý nghĩa                                  |
| ------------------ | ------- | ---------------------------------------- |
| `num_classes_topo` | `int`   | Số lớp topo                              |
| `pretrained`       | `bool`  | Có sử dụng ResNet50 pretrained hay không |
| `dropout_p`        | `float` | Xác suất dropout trong MLP heads         |

### 3.3. Hàm forward

```python
def forward(self, images, attention_maps):
    ...
```

Input bắt buộc:

| Tensor           |          Shape | Dtype mong đợi  |
| ---------------- | -------------: | --------------- |
| `images`         | `[B, 3, H, W]` | `torch.float32` |
| `attention_maps` | `[B, 1, H, W]` | `torch.float32` |

Output bắt buộc:

```python
return topo_logits, coord_pred
```

| Tensor        |                   Shape | Ghi chú                   |
| ------------- | ----------------------: | ------------------------- |
| `topo_logits` | `[B, num_classes_topo]` | Raw logits, không Softmax |
| `coord_pred`  |                `[B, 2]` | Tọa độ liên tục dự đoán   |

---

## 4. ResNet50 Backbone

### 4.1. Nguyên tắc sử dụng ResNet50

Backbone sử dụng ResNet50 pretrained. Tuy nhiên, phải loại bỏ:

```text
- avgpool
- fc
```

Chỉ giữ từ đầu mạng đến hết `layer4`.

Các thành phần được giữ lại:

```text
conv1
bn1
relu
maxpool
layer1
layer2
layer3
layer4
```

Cấu trúc backbone tương đương:

```python
self.backbone = nn.Sequential(
    resnet.conv1,
    resnet.bn1,
    resnet.relu,
    resnet.maxpool,
    resnet.layer1,
    resnet.layer2,
    resnet.layer3,
    resnet.layer4,
)
```

### 4.2. Tensor đầu ra của backbone

Với đầu vào:

[
images \in \mathbb{R}^{B \times 3 \times H \times W}
]

ResNet50 backbone tạo Feature Map:

[
F_t = \phi_{ResNet50}(images)
]

Trong đó:

[
F_t \in \mathbb{R}^{B \times 2048 \times H' \times W'}
]

`2048` là số kênh đầu ra của `layer4`.

Với ảnh đầu vào `224 x 224`, shape thường gặp là:

```text
F_t: [B, 2048, 7, 7]
```

Tuy nhiên, không được hard-code `7 x 7`. Kích thước (H') và (W') phải được lấy động từ:

```python
features.shape[-2:]
```

---

## 5. Feature-Level Attention Fusion

### 5.1. Mục tiêu

Attention Map được tạo trước bởi Data Pipeline từ vùng động trong ảnh. Trong `models/architecture.py`, Attention Map được sử dụng để suy giảm đặc trưng vùng động tại cấp độ Feature Map sâu.

Không nhân Attention Map trực tiếp với ảnh RGB trong model.

Cơ chế bắt buộc:

[
F_{att} = F_t \odot A'_t
]

Trong đó:

| Ký hiệu | Ý nghĩa                                                    |
| ------- | ---------------------------------------------------------- |
| (F_t)   | Feature Map từ ResNet50                                    |
| (A'_t)  | Attention Map sau khi resize về kích thước của Feature Map |
| (\odot) | Phép nhân từng phần tử                                     |

---

### 5.2. Đồng bộ kích thước Tensor

Attention Map đầu vào có shape:

[
A_t \in \mathbb{R}^{B \times 1 \times H \times W}
]

Feature Map có shape:

[
F_t \in \mathbb{R}^{B \times 2048 \times H' \times W'}
]

Do đó, phải resize Attention Map:

[
A'_t = \text{Interpolate}(A_t, H', W')
]

với:

[
A'_t \in \mathbb{R}^{B \times 1 \times H' \times W'}
]

Code bắt buộc:

```python
attention_resized = F.interpolate(
    attention_maps,
    size=features.shape[-2:],
    mode="bilinear",
    align_corners=False,
)
```

Trong đó:

| Thành phần            | Ý nghĩa                                          |
| --------------------- | ------------------------------------------------ |
| `attention_maps`      | Tensor đầu vào `[B, 1, H, W]`                    |
| `features.shape[-2:]` | Kích thước không gian `[H', W']` của Feature Map |
| `mode="bilinear"`     | Nội suy song tuyến tính                          |
| `align_corners=False` | Thiết lập ổn định khi resize ảnh hoặc feature    |

---

### 5.3. Phép nhân Element-wise

Sau khi resize Attention Map:

```python
attended_features = features * attention_resized
```

Shape tương ứng:

| Tensor              |               Shape |
| ------------------- | ------------------: |
| `features`          | `[B, 2048, H', W']` |
| `attention_resized` |    `[B, 1, H', W']` |
| `attended_features` | `[B, 2048, H', W']` |

PyTorch sẽ broadcast kênh đơn của `attention_resized` lên toàn bộ 2048 kênh của Feature Map.

Ràng buộc:

```text
- Không dùng phép cộng thay cho phép nhân.
- Không dùng concatenate giữa Feature Map và Attention Map.
- Không áp dụng Attention ở cấp ảnh RGB trong model.
```

---

## 6. Global Average Pooling

Sau Feature-level Attention, Tensor được đưa qua Global Average Pooling:

[
z = GAP(F_{att})
]

Trong code:

```python
self.gap = nn.AdaptiveAvgPool2d((1, 1))
pooled = self.gap(attended_features)
```

Shape sau GAP:

```text
pooled: [B, 2048, 1, 1]
```

Sau đó flatten:

```python
z = torch.flatten(pooled, 1)
```

Shape cuối cùng:

```text
z: [B, 2048]
```

Vector (z) là đặc trưng chung được đưa vào hai nhánh MLP chạy song song.

---

## 7. Multi-task MLP Heads

Hai nhánh MLP nhận chung vector:

[
z \in \mathbb{R}^{B \times 2048}
]

Hai nhánh phải chạy song song và độc lập về đầu ra.

---

### 7.1. Nhánh phân loại topo

Nhánh phân loại topo dự đoán nhãn khu vực topo của robot.

Cấu trúc bắt buộc:

```text
Linear(2048 -> 512)
BatchNorm1d(512)
ReLU
Dropout(0.3)
Linear(512 -> 256)
ReLU
Linear(256 -> num_classes_topo)
```

Code tương ứng:

```python
self.topo_head = nn.Sequential(
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, num_classes_topo),
)
```

Đầu ra:

[
topo_logits \in \mathbb{R}^{B \times num_classes_topo}
]

Ràng buộc:

```text
- Không dùng Softmax trong model.
- Không dùng Argmax trong model.
- Không tính Cross-Entropy Loss trong model.
```

Lý do: `torch.nn.CrossEntropyLoss` nhận trực tiếp raw logits. Softmax hoặc argmax chỉ được dùng ở bước đánh giá hoặc suy luận.

---

### 7.2. Nhánh hồi quy tọa độ

Nhánh hồi quy tọa độ dự đoán hai giá trị tọa độ liên tục.

Cấu trúc bắt buộc:

```text
Linear(2048 -> 512)
BatchNorm1d(512)
ReLU
Dropout(0.3)
Linear(512 -> 256)
ReLU
Linear(256 -> 2)
```

Code tương ứng:

```python
self.coord_head = nn.Sequential(
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 2),
)
```

Đầu ra:

[
coord_pred \in \mathbb{R}^{B \times 2}
]

Ràng buộc:

```text
- Không đọc min/max tọa độ trong model.
- Không giải chuẩn hóa tọa độ trong model.
- Không tính Smooth L1 Loss trong model.
- Không áp dụng hậu xử lý tọa độ trong model.
- Không tự ý thêm Sigmoid, Tanh hoặc Clamp nếu không được yêu cầu bởi file cấu hình bên ngoài.
```

---

## 8. Forward Pass chuẩn

Forward pass bắt buộc thực hiện theo đúng thứ tự:

```text
1. Nhận images và attention_maps.
2. Đưa images qua ResNet50 backbone để lấy features.
3. Resize attention_maps về đúng kích thước không gian của features.
4. Nhân features với attention_resized.
5. Đưa attended_features qua Global Average Pooling.
6. Flatten thành vector [B, 2048].
7. Đưa vector vào topo_head để lấy topo_logits.
8. Đưa vector vào coord_head để lấy coord_pred.
9. Trả về topo_logits, coord_pred.
```

Pseudo-code chuẩn:

```python
def forward(self, images, attention_maps):
    features = self.backbone(images)

    attention_resized = F.interpolate(
        attention_maps,
        size=features.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    attended_features = features * attention_resized

    pooled = self.gap(attended_features)
    z = torch.flatten(pooled, 1)

    topo_logits = self.topo_head(z)
    coord_pred = self.coord_head(z)

    return topo_logits, coord_pred
```

---

## 9. Code skeleton bắt buộc cho `models/architecture.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DynamicAwareLocalizationNet(nn.Module):
    def __init__(
        self,
        num_classes_topo: int,
        pretrained: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        if num_classes_topo <= 0:
            raise ValueError("num_classes_topo must be a positive integer.")

        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None

        resnet = models.resnet50(weights=weights)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.topo_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes_topo),
        )

        self.coord_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, images: torch.Tensor, attention_maps: torch.Tensor):
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [B, 3, H, W], got {tuple(images.shape)}"
            )

        if attention_maps.ndim != 4:
            raise ValueError(
                "attention_maps must have shape [B, 1, H, W], "
                f"got {tuple(attention_maps.shape)}"
            )

        if images.shape[0] != attention_maps.shape[0]:
            raise ValueError(
                "images and attention_maps must have the same batch size."
            )

        if images.shape[1] != 3:
            raise ValueError(
                f"images channel dimension must be 3, got {images.shape[1]}"
            )

        if attention_maps.shape[1] != 1:
            raise ValueError(
                "attention_maps channel dimension must be 1, "
                f"got {attention_maps.shape[1]}"
            )

        features = self.backbone(images)

        attention_resized = F.interpolate(
            attention_maps,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        attended_features = features * attention_resized

        pooled = self.gap(attended_features)
        z = torch.flatten(pooled, 1)

        topo_logits = self.topo_head(z)
        coord_pred = self.coord_head(z)

        return topo_logits, coord_pred
```

---

## 10. Kiểm tra shape bắt buộc

AI agent phải kiểm tra module bằng input giả lập:

```python
import torch

B = 4
images = torch.randn(B, 3, 224, 224)
attention_maps = torch.ones(B, 1, 224, 224)

model = DynamicAwareLocalizationNet(num_classes_topo=2)
topo_logits, coord_pred = model(images, attention_maps)
```

Output mong đợi:

```python
assert topo_logits.shape == torch.Size([4, 2])
assert coord_pred.shape == torch.Size([4, 2])
```

Shape nội bộ mong đợi khi input là `224 x 224`:

```text
features: [4, 2048, 7, 7]
attention_resized: [4, 1, 7, 7]
attended_features: [4, 2048, 7, 7]
z: [4, 2048]
```

> Không được hard-code kích thước `7 x 7`. Đây chỉ là trường hợp cụ thể khi input là `224 x 224`.

---

## 11. Loss không nằm trong model

Model không tính loss. Output của model được `train.py` sử dụng như sau:

```python
topo_logits, coord_pred = model(images, attention_maps)

loss_topo = criterion_topo(topo_logits, topo_labels)
loss_coord = criterion_coord(coord_pred, coords)

loss_total = loss_topo + gamma * loss_coord
```

Công thức tổng quát:

[
\mathcal{L}_{total}
===================

\mathcal{L}*{topo}
+
\gamma \mathcal{L}*{coord}
]

Trong đó:

[
\mathcal{L}_{topo} = CE(topo_logits, topo_labels)
]

[
\mathcal{L}_{coord} = SmoothL1(coord_pred, coords)
]

`gamma` là hệ số cân bằng giữa hai nhiệm vụ và phải được xử lý ở `train.py`, không xử lý trong `models/architecture.py`.

---

## 12. Contract với DataLoader

DataLoader cung cấp batch theo dạng:

```python
images, attention_maps, topo_labels, coords = batch
```

Model chỉ nhận:

```python
topo_logits, coord_pred = model(images, attention_maps)
```

Model không được nhận hoặc xử lý:

```text
image_path
attention_path
csv_row
x_min
x_max
y_min
y_max
scene_name
sequence_id
folder_name
```

Những thông tin trên thuộc Data Pipeline hoặc Evaluation.

---

## 13. Các lỗi triển khai phải tránh

| Lỗi                                                    | Trạng thái        |
| ------------------------------------------------------ | ----------------- |
| Dùng ResNet50 đầy đủ gồm `avgpool` và `fc`             | Không hợp lệ      |
| Nhân Attention Map trực tiếp với ảnh RGB trong model   | Không đúng đặc tả |
| Dùng `torch.cat([features, attention_resized], dim=1)` | Không đúng đặc tả |
| Dùng Softmax trong `forward()`                         | Không hợp lệ      |
| Dùng Argmax trong `forward()`                          | Không hợp lệ      |
| Tính loss trong `forward()`                            | Không hợp lệ      |
| Đọc CSV trong `architecture.py`                        | Không hợp lệ      |
| Đọc ảnh trong `architecture.py`                        | Không hợp lệ      |
| Giải chuẩn hóa tọa độ trong model                      | Không hợp lệ      |
| Hard-code Feature Map là `7 x 7`                       | Không hợp lệ      |
| Gọi YOLOv8 trong model                                 | Không hợp lệ      |

---

## 14. Tiêu chí hoàn thành module

`models/architecture.py` được xem là hoàn thành khi thỏa mãn toàn bộ tiêu chí sau:

```text
1. Có class DynamicAwareLocalizationNet.
2. Nhận đúng hai input Tensor: images và attention_maps.
3. ResNet50 đã loại bỏ avgpool và fc.
4. Feature Map đầu ra backbone có 2048 channels.
5. Attention Map được resize bằng F.interpolate với mode="bilinear".
6. Attention được áp dụng bằng phép nhân element-wise ở feature-level.
7. GAP tạo vector [B, 2048].
8. Có hai MLP heads chạy song song.
9. Topo head trả raw logits [B, num_classes_topo].
10. Coord head trả Tensor [B, 2].
11. Không có logic đọc file, CSV, thư mục, YOLOv8, loss, optimizer hoặc checkpoint.
12. Test shape với input [B, 3, 224, 224] và [B, 1, 224, 224] chạy thành công.
```

---

## 15. Kết luận kỹ thuật

`models/architecture.py` là module thuần kiến trúc mạng. Nó chỉ biểu diễn ánh xạ:

[
(images, attention_maps)
\rightarrow
(topo_logits, coord_pred)
]

Toàn bộ logic dữ liệu, đồng bộ hệ quy chiếu, chuẩn hóa tọa độ, tạo Attention Map bằng YOLOv8, chia train/val/test, tính loss và lưu checkpoint phải nằm ngoài module này.

Đặc tả này là nguồn tham chiếu bắt buộc cho AI agent khi lập trình mạng sâu cho hệ thống định vị phân cấp RGB kết hợp Feature-level Attention nhận thức vùng động.
