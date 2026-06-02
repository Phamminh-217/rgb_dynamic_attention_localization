# AI_STATE.md

# Trí nhớ trạng thái dự án cho AI Agent

> **Critical Instruction — Bắt buộc đọc trước khi làm việc**
>
> Mỗi AI Agent khi bắt đầu một session mới phải đọc file `AI_STATE.md` này trước tiên. File này là **State Ledger** của dự án, dùng để ghi nhớ cấu trúc hệ thống, trạng thái tiến độ, các ràng buộc kiến trúc bất biến và nhiệm vụ tiếp theo.
>
> Không được viết code mới trước khi kiểm tra các mục trong file này. Nếu có xung đột giữa yêu cầu mới và các ràng buộc trong file này, phải ưu tiên giữ đúng kiến trúc đã khóa.

---

## 1. Tên dự án

**Hệ thống định vị phân cấp robot di động dựa trên ảnh RGB, kết hợp cơ chế Feature-level Attention nhận thức vùng động**

Tên ngắn đề xuất trong repository:

```text
rgb_dynamic_attention_localization
```

---

## 2. Mục tiêu hệ thống

Hệ thống định vị robot di động trong nhà từ ảnh RGB đầu vào. Với mỗi frame ảnh, mô hình cần dự đoán đồng thời:

```text
1. Nhãn khu vực topo
2. Tọa độ robot (x, y)
```

Trong đó:

| Thành phần    | Ý nghĩa                                             |
| ------------- | --------------------------------------------------- |
| `topo_label`  | Nhãn khu vực topo, ví dụ `0 = Room`, `1 = Corridor` |
| `coord_label` | Tọa độ chuẩn hóa `[x_n, y_n]` trong khoảng `[0, 1]` |
| `coord_pred`  | Tọa độ dự đoán bởi nhánh hồi quy                    |
| `topo_logits` | Raw logits dự đoán bởi nhánh phân loại topo         |

---

## 3. Nguyên lý kiến trúc cốt lõi

Hệ thống tuân thủ nghiêm ngặt nguyên lý:

```text
Separation of Concerns
```

Tức là các module phải tách biệt hoàn toàn về trách nhiệm.

```text
Data Pipeline Module
    ├── preprocess_offline.py
    └── utils/dataset.py

Model Architecture Module
    └── models/architecture.py

Training Controller
    └── train.py
```

Không module nào được lấn trách nhiệm của module khác.

---

## 4. Pipeline kỹ thuật đã khóa

Pipeline cuối cùng của hệ thống:

```text
RGB Image
    │
    ├── Offline YOLOv8
    │       └── Attention Map
    │
    └── ResNet50 Backbone
            │
            └── Feature Map Layer 4: [B, 2048, H', W']
                    │
                    └── Feature-level Attention
                            │
                            └── Global Average Pooling
                                    │
                                    └── Shared Feature Vector [B, 2048]
                                            │
                                            ├── Topological Classification Head
                                            └── Coordinate Regression Head
```

Cơ chế attention đã khóa:

```text
Attention Map được sinh offline từ YOLOv8.
Attention Map không nhân trực tiếp vào ảnh RGB.
Attention Map được resize bằng Bilinear Interpolate.
Attention Map được nhân Element-wise với Feature Map Layer 4 của ResNet50.
```

Công thức:

[
F_{att} = F_t \odot A'_t
]

Trong đó:

| Ký hiệu   | Ý nghĩa                                                |
| --------- | ------------------------------------------------------ |
| (F_t)     | Feature Map đầu ra Layer 4 của ResNet50                |
| (A'_t)    | Attention Map đã resize về kích thước `[B, 1, H', W']` |
| (\odot)   | Phép nhân từng phần tử                                 |
| (F_{att}) | Feature Map sau khi suy giảm đặc trưng vùng động       |

---

# 5. Nhật ký tiến độ dự án

## Giai đoạn 1: Khởi tạo dự án & Đặc tả kỹ thuật

**Trạng thái:** Hoàn thành thiết kế đặc tả ban đầu.

* [x] Xác định bài toán định vị phân cấp từ ảnh RGB.
* [x] Chốt kiến trúc tổng thể gồm Data Pipeline, Model Architecture và Training Controller.
* [x] Chốt nguyên lý tách module theo Separation of Concerns.
* [x] Thiết kế layout thư mục dự án.
* [x] Thiết kế cấu trúc `config.json`.
* [x] Viết đặc tả tổng quan `README.md`.
* [x] Viết đặc tả mạng sâu `MODEL_SPECIFICATION.md`.
* [x] Viết đặc tả dữ liệu `DATA_PIPELINE_SPEC.md`.
* [x] Viết đặc tả huấn luyện `TRAINING_SCRIPT_SPEC.md`.
* [x] Khởi tạo file trạng thái `AI_STATE.md`.

**Kết luận giai đoạn 1:**
Dự án đã có đầy đủ tài liệu thiết kế để AI Agent triển khai code theo từng module mà không phá vỡ kiến trúc.

---

## Giai đoạn 2: Module Xử lý dữ liệu đầu vào

**Trạng thái:** Hoàn thành.

Mục tiêu của giai đoạn này là lập trình:

```text
preprocess_offline.py
utils/dataset.py
```

Checklist chi tiết:

* [x] Viết hàm đọc `room_poses.csv` (pose.csv).
* [x] Viết hàm đọc `corridor_poses.csv` (pose.csv).
* [x] Kiểm tra schema tối thiểu của CSV thô gồm cột tọa độ và khớp frame 1-1 bằng index.
* [x] Tự động gán `topo_label = 0` cho dữ liệu phòng.
* [x] Tự động gán `topo_label = 1` cho dữ liệu hành lang.
* [x] Đọc `delta_x` và `delta_y` từ `config.json`.
* [x] Cộng offset hành lang theo công thức:

[
x^{global} = x^{corridor} + \Delta x
]

[
y^{global} = y^{corridor} + \Delta y
]

* [x] Gộp dữ liệu phòng và hành lang thành `total_poses.csv`.
* [x] Quét toàn bộ `total_poses.csv` để tính:

[
x_{min}, x_{max}, y_{min}, y_{max}
]

* [x] Cập nhật các cực biên vào trường `global_normalization` trong `config.json`.
* [x] Viết class `RobotLocalizationDataset(Dataset)`.
* [x] Dataset phải trả đúng 4 thành phần:

```python
image_tensor, attention_tensor, topo_label, coord_label
```

* [x] `image_tensor` có shape `[3, H, W]`.
* [x] `attention_tensor` có shape `[1, H, W]`.
* [x] `topo_label` là `torch.LongTensor`.
* [x] `coord_label` là `torch.FloatTensor` shape `[2]`.
* [x] Tọa độ phải được chuẩn hóa Min-Max về `[0, 1]`.
* [x] Module dữ liệu không được import hoặc định nghĩa ResNet, MLP, forward pass.

**Nhiệm vụ tiếp theo ưu tiên:**
Huấn luyện và lập trình bộ điều phối huấn luyện `train.py` (Giai đoạn 4).

---

## Giai đoạn 3: Module Kiến trúc mạng

**Trạng thái:** Hoàn thành.

Mục tiêu của giai đoạn này là lập trình:

```text
models/architecture.py
```

Checklist chi tiết:

* [x] Tạo class `FeatureAttentionHierarchicalNet` (đã override lớp `DynamicAwareLocalizationNet` theo yêu cầu trực tiếp).
* [x] Model nhận đúng 2 input Tensor:

```python
images, attention_maps
```

* [x] `images` có shape `[B, 3, H, W]`.
* [x] `attention_maps` có shape `[B, 1, H, W]`.
* [x] Load ResNet50 pretrained từ `torchvision.models`.
* [x] Cắt bỏ `avgpool` và `fc` của ResNet50.
* [x] Giữ ResNet50 từ đầu đến hết `layer4`.
* [x] Đảm bảo Feature Map đầu ra có shape:

```text
[B, 2048, H', W']
```

* [x] Resize Attention Map bằng:

```python
torch.nn.functional.interpolate(
    attention_maps,
    size=features.shape[-2:],
    mode="bilinear",
    align_corners=False
)
```

* [x] Nhân Element-wise:

```python
attended_features = features * attention_resized
```

* [x] Dùng `AdaptiveAvgPool2d((1, 1))`.
* [x] Flatten thành vector `[B, 2048]`.
* [x] Thiết kế nhánh phân loại topo:

```text
Linear(2048 -> 512)
BatchNorm1d(512)
ReLU
Dropout(0.3)
Linear(512 -> 256)
ReLU
Linear(256 -> num_classes_topo)
```

* [x] Nhánh phân loại topo trả về raw logits, không Softmax.
* [x] Thiết kế nhánh hồi quy tọa độ:

```text
Linear(2048 -> 512)
BatchNorm1d(512)
ReLU
Dropout(0.3)
Linear(512 -> 256)
ReLU
Linear(256 -> 2)
```

* [x] Nhánh hồi quy trả về Tensor `[B, 2]`.
* [x] Model không được đọc CSV.
* [x] Model không được xử lý file ảnh.
* [x] Model không được gọi YOLOv8.
* [x] Model không được tính loss.
* [x] Model không được lưu checkpoint.

**Nhiệm vụ tiếp theo ưu tiên:**
Triển khai Data Pipeline (Giai đoạn 2) bao gồm `preprocess_offline.py` và `utils/dataset.py`.

---

## Giai đoạn 4: Điều phối & Huấn luyện

**Trạng thái:** Chưa làm.

Mục tiêu của giai đoạn này là lập trình:

```text
train.py
```

Checklist chi tiết:

* [ ] Đọc `config.json` từ command line.
* [ ] Tự xác định device `cuda` hoặc `cpu`.
* [ ] Hỗ trợ chế độ Google Colab.
* [ ] Kiểm tra file `data.zip` trên Google Drive.
* [ ] Tự giải nén `data.zip` vào SSD nội bộ `/content/dataset` bằng `zipfile`.
* [ ] Không huấn luyện trực tiếp trên Google Drive.
* [ ] Kiểm tra sự tồn tại của `total_poses.csv`.
* [ ] Kiểm tra thư mục `attention_maps`.
* [ ] Nếu thiếu dữ liệu tiền xử lý, tự động gọi `preprocess_offline.py`.
* [ ] Khởi tạo `RobotLocalizationDataset`.
* [ ] Khởi tạo `DataLoader`.
* [ ] Khởi tạo `DynamicAwareLocalizationNet`.
* [ ] Tính loss phân loại bằng:

```python
nn.CrossEntropyLoss()
```

* [ ] Tính loss hồi quy bằng:

```python
nn.SmoothL1Loss()
```

* [ ] Tính tổng loss:

[
\mathcal{L}_{total}
===================

\mathcal{L}*{CrossEntropy}
+
\gamma \cdot \mathcal{L}*{SmoothL1}
]

* [ ] Huấn luyện bằng optimizer `AdamW`.
* [ ] Có validation loop với `torch.no_grad()`.
* [ ] Lưu checkpoint định kỳ sau mỗi 10 epochs.
* [ ] Sao lưu checkpoint về Google Drive.
* [ ] Lưu `best_model.pth` khi validation loss tốt hơn.
* [ ] Hỗ trợ resume training từ checkpoint.

**Nhiệm vụ tiếp theo ưu tiên:**
Sau khi Data Pipeline và Model Architecture chạy đúng shape, triển khai `train.py`.

---

# 6. Ràng buộc kỹ thuật bất biến

Các ràng buộc dưới đây là **Sticky Constraints**. Không được thay đổi nếu không có chỉ thị rõ ràng.

## 6.1. Ràng buộc tách module

| Ràng buộc                                              | Trạng thái |
| ------------------------------------------------------ | ---------- |
| `models/architecture.py` không được đọc CSV            | Bắt buộc   |
| `models/architecture.py` không được đọc ảnh từ ổ cứng  | Bắt buộc   |
| `models/architecture.py` không được xử lý path thư mục | Bắt buộc   |
| `models/architecture.py` không được gọi YOLOv8         | Bắt buộc   |
| `models/architecture.py` không được tính loss          | Bắt buộc   |
| `utils/dataset.py` không được định nghĩa ResNet        | Bắt buộc   |
| `utils/dataset.py` không được định nghĩa MLP           | Bắt buộc   |
| `utils/dataset.py` không được chứa forward pass        | Bắt buộc   |
| `train.py` không được tự định nghĩa kiến trúc mạng     | Bắt buộc   |
| `train.py` không được tự đọc từng ảnh thay Dataset     | Bắt buộc   |

---

## 6.2. Ràng buộc Tensor

Model nhận đầu vào:

```python
images: torch.FloatTensor          # [B, 3, H, W]
attention_maps: torch.FloatTensor  # [B, 1, H, W]
```

Backbone ResNet50 đến hết `layer4` phải tạo:

```python
features: torch.FloatTensor        # [B, 2048, H', W']
```

Attention Map sau resize:

```python
attention_resized: torch.FloatTensor  # [B, 1, H', W']
```

Sau nhân Element-wise:

```python
attended_features: torch.FloatTensor  # [B, 2048, H', W']
```

Sau Global Average Pooling và flatten:

```python
z: torch.FloatTensor  # [B, 2048]
```

Output của model:

```python
topo_logits: torch.FloatTensor  # [B, num_classes_topo]
coord_pred: torch.FloatTensor   # [B, 2]
```

---

## 6.3. Ràng buộc Attention

Cơ chế Attention bắt buộc là:

```text
Feature-level Attention
```

Không được thay bằng:

```text
Image-level Attention
Concatenation Fusion
Additive Fusion
Self-Attention Transformer
Temporal Attention
```

Trừ khi người dùng yêu cầu mở rộng kiến trúc trong một giai đoạn nghiên cứu mới.

Công thức bắt buộc:

[
F_{att} = F_t \odot A'_t
]

---

## 6.4. Ràng buộc tọa độ

Tọa độ hồi quy bắt buộc phải chuẩn hóa Min-Max về đoạn `[0, 1]`.

Công thức chuẩn hóa:

[
x_n = \frac{x - x_{min}}{x_{max} - x_{min}}
]

[
y_n = \frac{y - y_{min}}{y_{max} - y_{min}}
]

Không được huấn luyện trực tiếp bằng tọa độ mét nếu chưa có yêu cầu thay đổi đặc tả.

Dataset phải trả:

```python
coord_label = torch.tensor([x_n, y_n], dtype=torch.float32)
```

---

## 6.5. Ràng buộc loss

Loss phân loại topo:

```python
nn.CrossEntropyLoss()
```

Loss hồi quy tọa độ:

```python
nn.SmoothL1Loss()
```

Loss tổng:

[
\mathcal{L}_{total}
===================

\mathcal{L}*{CrossEntropy}
+
\gamma \cdot \mathcal{L}*{SmoothL1}
]

Không được dùng Softmax trước `CrossEntropyLoss`.

---

# 7. Các file đặc tả phải đọc trước khi code

AI Agent phải đọc các file theo thứ tự sau:

```text
1. AI_STATE.md
2. README.md
3. DATA_PIPELINE_SPEC.md
4. MODEL_SPECIFICATION.md
5. TRAINING_SCRIPT_SPEC.md
6. config.json
```

Quy tắc:

```text
- Khi viết Data Pipeline, ưu tiên DATA_PIPELINE_SPEC.md.
- Khi viết Model Architecture, ưu tiên MODEL_SPECIFICATION.md.
- Khi viết train.py, ưu tiên TRAINING_SCRIPT_SPEC.md.
- Khi có xung đột, giữ nguyên Sticky Constraints trong AI_STATE.md.
```

---

# 8. Trạng thái hiện tại của dự án

Tại thời điểm khởi tạo file này:

```text
README.md                  : Đã thiết kế
MODEL_SPECIFICATION.md     : Đã thiết kế
DATA_PIPELINE_SPEC.md      : Đã thiết kế
TRAINING_SCRIPT_SPEC.md    : Đã thiết kế
AI_STATE.md                : Đã khởi tạo
preprocess_offline.py      : Chưa lập trình
utils/dataset.py           : Chưa lập trình
models/architecture.py     : Chưa lập trình
train.py                   : Chưa lập trình
```

---

# 9. Nhiệm vụ tiếp theo cho AI Agent

Nhiệm vụ tiếp theo nên thực hiện theo đúng thứ tự:

```text
1. Tạo cấu trúc thư mục dự án.
2. Tạo config.json theo README.md và các file đặc tả.
3. Lập trình preprocess_offline.py.
4. Lập trình utils/dataset.py.
5. Test Dataset bằng một sample giả hoặc dữ liệu thật.
6. Lập trình models/architecture.py.
7. Test shape model bằng Tensor giả.
8. Lập trình train.py.
9. Chạy thử một epoch nhỏ.
10. Kiểm tra checkpoint và log.
```

Không được nhảy sang `train.py` trước khi Dataset và Model đã qua kiểm tra shape.

---

# 10. Quy tắc cập nhật file này

Sau mỗi phiên làm việc, AI Agent phải cập nhật `AI_STATE.md` nếu có thay đổi trạng thái.

Ví dụ:

```text
- Khi hoàn thành preprocess_offline.py, đánh dấu checklist tương ứng ở Giai đoạn 2.
- Khi hoàn thành utils/dataset.py, đánh dấu checklist tương ứng ở Giai đoạn 2.
- Khi hoàn thành models/architecture.py, đánh dấu checklist tương ứng ở Giai đoạn 3.
- Khi hoàn thành train.py, đánh dấu checklist tương ứng ở Giai đoạn 4.
```

Không được đánh dấu `[x]` cho nhiệm vụ chưa được lập trình hoặc chưa được kiểm tra.

---

# 11. Quy tắc kiểm tra nhanh trước khi merge code

Trước khi coi một module là hoàn thành, phải kiểm tra:

| Module             | Kiểm tra bắt buộc                                 |
| ------------------ | ------------------------------------------------- |
| Data Pipeline      | Dataset trả đúng 4 Tensor                         |
| Model Architecture | Forward pass chạy với input giả                   |
| Training Script    | Chạy được ít nhất 1 epoch nhỏ                     |
| Checkpoint         | Lưu được `.pth` vào local và Drive nếu dùng Colab |
| Config             | Không thiếu trường cần thiết                      |

Shape test tối thiểu cho model:

```python
B = 4
images = torch.randn(B, 3, 224, 224)
attention_maps = torch.ones(B, 1, 224, 224)

topo_logits, coord_pred = model(images, attention_maps)

assert topo_logits.shape == torch.Size([4, 2])
assert coord_pred.shape == torch.Size([4, 2])
```

---

# 12. Cảnh báo chống sai kiến trúc

Các hành vi sau được xem là sai thiết kế:

```text
- Đưa YOLOv8 vào forward của model.
- Nhân Attention Map trực tiếp với ảnh RGB trong model.
- Dùng torch.cat để ghép Attention Map với Feature Map.
- Để architecture.py đọc total_poses.csv.
- Để dataset.py import ResNet50.
- Để train.py tự đọc ảnh bằng PIL thay Dataset.
- Tính global_normalization trong train.py.
- Dùng tọa độ mét chưa chuẩn hóa làm nhãn hồi quy.
- Dùng Softmax trước CrossEntropyLoss.
- Chỉ lưu checkpoint trong /content mà không sao lưu về Google Drive khi chạy Colab.
```

Nếu phát hiện một trong các lỗi trên, phải dừng triển khai và sửa lại theo đặc tả.

---

# 13. Tóm tắt trạng thái ngắn

```text
Dự án đang ở cuối Giai đoạn 1.
Tài liệu đặc tả đã hoàn thành.
Code chính chưa triển khai.
Nhiệm vụ tiếp theo: lập trình Data Pipeline.
Ràng buộc quan trọng nhất: giữ tách biệt Data Module và Model Module.
```
