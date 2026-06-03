# Hierarchical RGB Robot Localization with Dynamic-aware Feature-level Attention

Dự án này triển khai hệ thống định vị robot di động phân cấp trong nhà từ ảnh RGB và bản đồ chú ý động ở cấp độ đặc trưng (Feature-level Attention). Hệ thống dự đoán đồng thời nhãn khu vực topo (Room/Corridor) và tọa độ thực tế $(x, y)$ của robot trong môi trường có vật thể động di chuyển (người).

---

## 🚀 Kết Quả Đạt Được (Test Set Evaluation)

Dưới đây là các chỉ số đánh giá chi tiết trên tập kiểm thử độc lập **Test Set (381 ảnh chưa từng thấy khi huấn luyện)**:

### 1. Chỉ số đánh giá tổng quát (Global Metrics)

| Chỉ số (Metric) | Kết quả (Value) | Mô tả (Description) |
| :--- | :---: | :--- |
| **Topological Area Accuracy** | **100.00%** | Phân loại chính xác 100% khu vực Room / Corridor |
| **Mean Localization Error (MAE)** | **3.20 cm** | Sai số khoảng cách định vị trung bình |
| **Root Mean Squared Error (RMSE)** | **3.93 cm** | Sai số trung bình bình phương (đo mức độ ổn định) |
| **MAE - Trục X** | **1.98 cm** | Sai số trung bình trên trục hoành |
| **MAE - Trục Y** | **1.97 cm** | Sai số trung bình trên trục tung |
| **Maximum Localization Error** | **15.48 cm** | Sai số lớn nhất ở góc cua ngắt bản đồ |

### 2. Kết quả kiểm thử trên các mẫu ảnh ngẫu nhiên (Individual Samples)

| Tên ảnh (Frame) | Nhãn thật (True Area) | Dự đoán (Pred Area) | Tọa độ thật (True Pose) | Tọa độ dự đoán (Pred Pose) | Sai số (Error) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `frame_001425.png` | Corridor | Corridor (100% conf) | (-0.4957m, +1.3617m) | (-0.4988m, +1.3780m) | **1.66 cm** |
| `frame_001332.png` | Corridor | Corridor (100% conf) | (-0.4948m, +1.2459m) | (-0.4922m, +1.2264m) | **1.96 cm** |
| `frame_001285.png` | Corridor | Corridor (100% conf) | (-0.4763m, +1.2041m) | (-0.5027m, +1.1942m) | **2.82 cm** |
| `frame_000599.png` | Room | Room (100% conf) | (+0.2976m, +0.0421m) | (+0.2569m, +0.0529m) | **4.21 cm** |
| `frame_001201.png` | Room | Room (100% conf) | (-0.6310m, +0.1302m) | (-0.6886m, +0.1178m) | **5.89 cm** |
| `frame_000628.png` | Room | Room (100% conf) | (+0.3060m, +0.0523m) | (+0.2267m, +0.0791m) | **8.37 cm** |

*Biểu đồ so sánh quỹ đạo thực tế và dự đoán được lưu tại: [checkpoints/trajectory_comparison.png](checkpoints/trajectory_comparison.png)*

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
├── docs/
│   └── COLAB_TRAINING_GUIDE.md# Hướng dẫn chi tiết chạy train trên Google Colab
├── checkpoints/               # Nơi lưu trữ weights (.pth) và đồ thị kết quả
├── config.json                # File cấu hình
├── scripts/                   # Thư mục chứa các script chính chạy offline/suy luận
│   ├── preprocess_offline.py  # Tiền xử lý YOLOv8 tạo attention map offline
│   └── inference.py           # Đánh giá quỹ đạo và chạy suy luận robot siêu tham số huấn luyện đầy đủ
├── config_mini.json           # File cấu hình huấn luyện thử nghiệm nhanh
├── create_mini_dataset.py     # Script trích xuất nhanh tập dữ liệu mini để test local
│   scripts/preprocess_offline.py  # Script tiền xử lý YOLOv8 tạo attention map offline
├── train.py                   # Script huấn luyện & đánh giá chính
│   scripts/inference.py           # Công cụ chạy suy luận và so sánh quỹ đạo thực tế
```

---

## 💻 Hướng Dẫn Sử Dụng Nhanh (Quick Start)

### 1. Tạo tập mini và test nhanh local (5 giây)
```bash
# Trích xuất 200 ảnh ngẫu nhiên cân bằng lớp và sinh cấu hình mini
python3 create_mini_dataset.py

# Huấn luyện thử nghiệm 10 epochs
python3 train.py --config config_mini.json
```

### 2. Huấn luyện đầy đủ trên máy local hoặc Google Colab
Đọc hướng dẫn chi tiết tại [COLAB_TRAINING_GUIDE.md](docs/COLAB_TRAINING_GUIDE.md). Lệnh chạy chính thức:
```bash
python3 train.py --config config.json --colab
```

### 3. Đánh giá quỹ đạo và chạy suy luận trực tiếp
```bash
# Vẽ biểu đồ so sánh quỹ đạo test và tính toán sai số bằng centimet
PYTHONPATH=. python3 scripts/inference.py --config config.json --model_path checkpoints/best_model.pth --trajectory

# Chạy suy luận robot cho một ảnh thô bất kỳ
PYTHONPATH=. python3 scripts/inference.py --config config.json --model_path checkpoints/best_model.pth --image_path <đường-dẫn-ảnh>
```
