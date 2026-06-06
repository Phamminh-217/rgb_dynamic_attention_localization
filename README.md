# Hierarchical RGB Robot Localization with Feature-level Attention

[👉 Click here for English version (README_EN.md)](README_EN.md)

Dự án này triển khai hệ thống định vị robot di động phân cấp trong nhà từ ảnh RGB và bản đồ chú ý động ở cấp độ đặc trưng (Feature-level Attention). Hệ thống dự đoán đồng thời nhãn khu vực topo (Room/Corridor) và tọa độ thực tế $(x, y)$ của robot trong môi trường có vật thể động di chuyển (người).

---

## 🚀 Kết Quả Đạt Được (Test Set Evaluation)

Dưới đây là các chỉ số đánh giá chi tiết trên tập kiểm thử độc lập:

### 1. Chỉ số đánh giá tổng quát (Global Metrics)

| Chỉ số (Metric) | Kết quả (Value) | Mô tả (Description) |
| :--- | :---: | :--- |
| **Topological Area Accuracy** | **100.00%** | Phân loại chính xác 100% khu vực Room / Corridor |
| **Mean Localization Error (MAE)** | **3.20 cm** | Sai số khoảng cách định vị trung bình |
| **Root Mean Squared Error (RMSE)** | **3.93 cm** | Sai số trung bình bình phương (đo mức độ ổn định) |
| **MAE - Trục X** | **1.98 cm** | Sai số trung bình trên trục hoành |
| **MAE - Trục Y** | **1.97 cm** | Sai số trung bình trên trục tung |
| **Maximum Localization Error** | **15.48 cm** | Sai số lớn nhất ở góc cua ngắt bản đồ |

*Biểu đồ so sánh quỹ đạo thực tế và dự đoán Room được lưu tại: [results/room_trajectory_comparison.png](results/room_trajectory_comparison.png)*
*Đồ thị biểu diễn lịch sử tổn thất (loss) và độ chính xác (accuracy) huấn luyện: [checkpoints/training_curves.png](checkpoints/training_curves.png)* (được sao lưu đồng bộ về Drive khi train trên Colab)

---

## 🛠️ Kiến Trúc Hệ Thống (System Pipeline)

Hệ thống tuân thủ nghiêm ngặt nguyên lý **Phân tách mối quan tâm (Separation of Concerns)**:

```text
       Ảnh RGB thô ──► [ YOLOv8 (Offline) ] ──► Attention Map (Bản đồ chú ý)
            │                                           │
            ▼                                           ▼
    [ ResNet50 Layer 4 ]                         [ Nội suy Bilinear ]
     (Feature Map 2048)                                 │
            │                                           │
            ▼                                           ▼
            └──────────────► [ Phép nhân ⊙ ] ◄──────────┘
                             (Lọc bỏ nhiễu người)
                                     │
                                     ▼
                            [ Global Avg Pool ]
                                     │
                                     ▼
                          ┌──────────┴──────────┐
                          ▼                     ▼
                  [ MLP Phân Loại ]      [ MLP Hồi Quy ]
                    (Room/Corridor)          (Tọa độ x,y)
```

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── data/                      # Thư mục dữ liệu (Được ignore khi push Git)
│   ├── raw/                   # Dữ liệu ảnh thô và pose.csv gốc
│   └── processed/             # CSV gộp và thư mục Attention Maps
├── models/
│   └── architecture.py        # Định nghĩa mạng nơ-ron FeatureAttentionHierarchicalNet
├── utils/
│   └── dataset.py             # Dataloader, chuẩn hóa tọa độ Min-Max
├── COLAB_TRAINING_GUIDE.md    # Hướng dẫn chi tiết chạy train trên Google Colab
├── checkpoints/               # Nơi lưu trữ weights (.pth) và đồ thị kết quả
├── config.json                # File cấu hình siêu tham số huấn luyện đầy đủ
├── train.py                   # Script huấn luyện & đánh giá chính
├── results/                   # Thư mục xuất các kết quả báo cáo quỹ đạo, visual report
├── scripts/                   # Thư mục chứa các script tiền xử lý chính
│   └── preprocess_offline.py  # Tiền xử lý YOLOv8 tạo attention map offline
└── scratch/                   # Các script chạy test nhanh, đánh giá và sinh báo cáo visual
    ├── inference_custom.py    # Test 1 ảnh bất kỳ -> xuất báo cáo visual 3-panel vào results/
    ├── evaluate_directory_trajectory.py # Chạy trên thư mục ảnh -> xuất đồ thị quỹ đạo so sánh
    ├── evaluate_room.py       # Đánh giá và vẽ quỹ đạo riêng của Room
    └── evaluate_corridor.py   # Đánh giá và vẽ quỹ đạo riêng của Corridor
```

---

## 💻 Hướng Dẫn Sử Dụng Chi Tiết

### 1. Huấn luyện Model (Training)

#### Huấn luyện trên máy cá nhân (Local GPU):
```bash
python3 train.py --config config.json
```

#### Huấn luyện trên Google Colab (Có kết nối Google Drive):
*Xem chi tiết hướng dẫn tại [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)*
```bash
python3 train.py --config config.json --colab
```

---

### 2. Suy luận trên 1 ảnh bất kỳ (Single Image Inference)
Chạy script `inference_custom.py` để phân tích 1 hoặc nhiều ảnh cụ thể. Script sẽ tự động chạy YOLOv8 live để sinh bản đồ chú ý và tạo **ảnh so sánh trực quan (gồm ảnh gốc, attention map, tọa độ 2D so với Ground Truth)** lưu trong thư mục `results/`.

```bash
# Test 1 ảnh bất kỳ (Ví dụ ảnh Room)
PYTHONPATH=. python3 scratch/inference_custom.py --paths data/raw/room/images/20260514_202335_361623_000100.png --output_dir results

# Test nhiều ảnh cùng lúc
PYTHONPATH=. python3 scratch/inference_custom.py --paths <đường-dẫn-ảnh-1> <đường-dẫn-ảnh-2> --output_dir results
```

---

### 3. Đánh giá quỹ đạo trên Folder dữ liệu (Folder Trajectory Inference)
Chạy script `evaluate_directory_trajectory.py` để thực hiện suy luận trên toàn bộ ảnh trong một thư mục bất kỳ. Hệ thống sẽ tự động ghép nối thứ tự các ảnh theo thời gian, vẽ **đường quỹ đạo dự đoán** đối chiếu đè lên **đường quỹ đạo thực tế (Ground Truth)** để so sánh trực quan, có đánh dấu điểm **START** và **END**.

```bash
# Đánh giá và vẽ quỹ đạo so sánh cho toàn bộ thư mục Room
PYTHONPATH=. python3 scratch/evaluate_directory_trajectory.py --images_dir data/raw/room/images/ --output_plot results/room_trajectory_comparison.png

# Đánh giá và vẽ quỹ đạo so sánh cho toàn bộ thư mục Corridor
PYTHONPATH=. python3 scratch/evaluate_directory_trajectory.py --images_dir data/raw/corridor/images/ --output_plot results/corridor_trajectory_comparison.png
```
