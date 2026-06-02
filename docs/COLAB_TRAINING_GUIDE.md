# Hướng Dẫn Huấn Luyện Mô Hình Trên Google Colab

Tài liệu này hướng dẫn chi tiết cách chạy huấn luyện mô hình **Hierarchical RGB Robot Localization with Dynamic Attention** trên Google Colab sử dụng GPU tốc độ cao và tối ưu hóa việc đọc/ghi dữ liệu thông qua Google Drive.

---

## 📌 Cơ Chế Hoạt Động Trên Colab (Đã Được Tối Ưu)

Đọc trực tiếp hàng nghìn file ảnh nhỏ từ Google Drive rất chậm và dễ gây lỗi nghẽn I/O. Do đó, hệ thống huấn luyện của bạn đã được tích hợp sẵn quy trình tối ưu hóa:
1. **Nén dữ liệu:** Bạn nén toàn bộ thư mục `data` ở máy local thành file `data.zip` và upload lên Google Drive.
2. **Giải nén tốc độ cao:** Khi chạy `train.py` với cờ `--colab`, mã nguồn sẽ tự động giải nén file `data.zip` từ Google Drive sang ổ cứng SSD cục bộ của máy ảo Colab (`/content/dataset`) chỉ trong vài giây.
3. **Tự động sao lưu:** Trong suốt quá trình huấn luyện, các file model checkpoint (`best_model.pth`, `epoch_xxx.pth`) sẽ được lưu đồng thời ở máy ảo và **sao lưu trực tiếp về Google Drive** để tránh mất dữ liệu khi máy ảo Colab bị ngắt kết nối.

---

## 🛠 Bước 1: Chuẩn Bị Dữ Liệu Trên Google Drive

1. Trên máy tính cá nhân (sau khi đã chạy tiền xử lý sinh ra các ảnh attention map thành công), bạn nén thư mục `data/` thành file `data.zip`.
   * *Lưu ý:* Cấu trúc bên trong file zip phải chứa các thư mục con:
     ```text
     data.zip
     ├── raw/
     │   ├── room/
     │   └── corridor/
     └── processed/
         ├── attention_maps/
         └── total_poses.csv
     ```
2. Truy cập Google Drive cá nhân của bạn.
3. Tạo một thư mục có tên: `REGRESSION_MODEL`.
4. Tải file `data.zip` bạn vừa nén lên thư mục này.
   * Đường dẫn đầy đủ trên Drive sẽ là: `/content/drive/MyDrive/REGRESSION_MODEL/data.zip` (khớp hoàn toàn với cấu hình mặc định trong `config.json`).

---

## 🚀 Bước 2: Thiết Lập Và Chạy Trên Google Colab

1. Truy cập vào [Google Colab](https://colab.research.google.com/) và tạo một **Notebook mới** (`New Notebook`).
2. **Kích hoạt GPU:** Vào `Runtime` -> `Change runtime type` -> Chọn **T4 GPU** (hoặc L4, A100 nếu có tài khoản Pro) -> Nhấn **Save**.

### Tạo các Cell Code và chạy lần lượt:

#### **Cell 1: Kết nối với Google Drive (Mount Drive)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### **Cell 2: Clone dự án từ GitHub và di chuyển vào thư mục dự án**
```bash
!git clone https://github.com/Phamminh-217/rgb_dynamic_attention_localization.git
%cd rgb_dynamic_attention_localization
```

#### **Cell 3: Cài đặt các thư viện cần thiết**
```bash
!pip install -r requirements.txt
```

#### **Cell 4: Khởi chạy Huấn Luyện (Training)**
Kích hoạt cờ `--colab` để tự động kích hoạt tối ưu hóa SSD và sao lưu Drive:
```bash
!python3 train.py --config config.json --colab
```

---

## 💾 Bước 3: Thu Thập Kết Quả Huấn Luyện

* Trong quá trình huấn luyện, sau mỗi `checkpoint_interval` (mặc định là 10 epochs) hoặc bất cứ khi nào đạt Loss thấp nhất trên tập Val, hệ thống sẽ tự động copy model về Google Drive của bạn tại thư mục:
  `/content/drive/MyDrive/REGRESSION_MODEL/checkpoints/`
* Bạn sẽ nhận được các file:
  * `best_model.pth`: Trọng số tốt nhất đạt độ chính xác cao nhất.
  * `epoch_xxx.pth`: File checkpoint lưu trạng thái huấn luyện để bạn có thể tiếp tục train tiếp nếu bị ngắt kết nối.

---

## 🔄 Cách Tiếp Tục Huấn Luyện Khi Bị Ngắt Kết Nối (Resume)

Nếu phiên làm việc trên Colab bị timeout hoặc ngắt kết nối giữa chừng, bạn chỉ cần mở lại Colab, chạy lại các **Cell 1, 2, 3** và chạy lệnh huấn luyện dưới đây để tự động khôi phục lại trạng thái và tiếp tục chạy tiếp từ epoch bị ngắt:

```bash
!python3 train.py --config config.json --colab --resume /content/drive/MyDrive/REGRESSION_MODEL/checkpoints/best_model.pth
```
