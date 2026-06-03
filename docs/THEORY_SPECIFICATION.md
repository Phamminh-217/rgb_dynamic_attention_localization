# THEORY_SPECIFICATION.md
# Đặc Tả Lý Thuyết và Phương Pháp Đề Xuất

> **Mục đích:** Tài liệu này là chuẩn đối chiếu bắt buộc để AI Agent kiểm tra tính đúng đắn của mã nguồn so với cơ sở toán học và kiến trúc hệ thống đã thiết kế. Mọi module trong code phải truy xuất về đây để xác minh.

---

## 1. Mô Hình Hóa Bài Toán Tổng Quán

### 1.1 Tổng quan
Hệ thống hướng tới **định vị phân cấp và nhận thức vùng động** từ ảnh RGB đầu vào — **không** hồi quy trực tiếp.

### 1.2 Đầu vào
Ảnh RGB tại thời điểm $t$:

$$I_t \in \mathbb{R}^{H \times W \times 3}$$

- $H$: chiều cao ảnh
- $W$: chiều rộng ảnh

### 1.3 Đầu ra đa nhiệm

$$(\hat{c}_t,\ \hat{\mathbf{p}}_t) = f_{\theta}(I_t)$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $f_{\theta}(\cdot)$ | Mô hình định vị, tham số hóa bởi $\theta$ |
| $\hat{c}_t \in \{1, 2, \ldots, K\}$ | Nhãn khu vực topo dự đoán ($K$ = tổng số khu vực) |
| $\hat{\mathbf{p}}_t = [\hat{x}_t, \hat{y}_t]^T$ | Tọa độ vị trí dự đoán trong hệ quy chiếu bản đồ |

### ✅ Checklist đối chiếu
- [ ] Model trả về **đồng thời** 2 đầu ra: nhãn topo + tọa độ 2D
- [ ] Tọa độ đầu ra dạng $[\hat{x}_t, \hat{y}_t]^T$, không phải 3D

---

## 2. Mô-đun Phân Tách Vùng Động — YOLOv8

### 2.1 Tập phát hiện đối tượng

$$\mathcal{D}_t = \{(b_i, s_i, l_i)\}_{i=1}^{N_t}$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $N_t$ | Số đối tượng phát hiện trong khung hình |
| $b_i = (x_i, y_i, w_i, h_i)$ | Bounding box của đối tượng thứ $i$ |
| $s_i$ | Confidence score |
| $l_i$ | Class label |

### 2.2 Điều kiện lọc đối tượng động

Một đối tượng được coi là **nguồn nhiễu động cần loại bỏ** khi và chỉ khi:

$$s_i \geq \delta, \quad l_i \in \mathcal{C}_d$$

- **Ngưỡng cố định:** $\delta = 0.6$
- $\mathcal{C}_d$: tập lớp đối tượng động (ưu tiên lớp người)

### 2.3 Mở rộng bounding box

$$b_i' = (x_i - k,\ y_i - k,\ w_i + 2k,\ h_i + 2k)$$

- Nới rộng thêm $k$ pixel về **mọi phía** để tránh sai số vùng biên do chuyển động

### 2.4 Mặt nạ động nhị phân

$$M_t(u,v) = \begin{cases} 1, & \text{nếu pixel } (u,v) \text{ thuộc vùng động } b_i' \\ 0, & \text{ngược lại} \end{cases}$$

### ✅ Checklist đối chiếu
- [ ] `confidence_threshold` trong code = `0.6`
- [ ] Chỉ lọc các class thuộc $\mathcal{C}_d$ (không lọc toàn bộ detections)
- [ ] Bounding box được mở rộng đúng công thức: `-k` ở góc trên-trái, `+2k` ở width/height
- [ ] Mask $M_t$ là ma trận nhị phân $\{0, 1\}$ cùng kích thước $H \times W$

---

## 3. Cơ Chế Chú Ý Không Gian Suy Giảm Mềm (Spatial Attention)

### 3.1 Ma trận chú ý

$$A_t(u,v) = 1 - \alpha \cdot M_t(u,v), \quad \alpha \in [0, 1]$$

| Vùng | $M_t$ | Trọng số $A_t$ |
|---|---|---|
| Tĩnh | 0 | 1 (giữ nguyên) |
| Động | 1 | $1 - \alpha$ (suy giảm) |

> $\alpha$ càng lớn → vùng động càng bị triệt tiêu mạnh

### 3.2 Phương án triển khai

**Phương án 1 — Image-level:**
$$I_t' = I_t \odot A_t$$

**Phương án 2 — Feature-level** (dùng bilinear interpolation đưa $A_t$ về cùng kích thước Feature Map):
$$F_t' = F_t \odot A_t'$$

### ✅ Checklist đối chiếu
- [ ] Xác nhận code đang dùng phương án nào (image-level hay feature-level)
- [ ] Phép nhân là element-wise ($\odot$), không phải matrix multiplication
- [ ] Nếu dùng feature-level: $A_t$ phải được resize bằng bilinear interpolation trước khi nhân
- [ ] Giá trị $\alpha$ nằm trong khoảng $[0, 1]$

---

## 4. Mạng Trích Xuất Đặc Trưng — ResNet50 Backbone

### 4.1 Khối Residual cơ bản

$$y = F(x, W) + x$$

### 4.2 Khối Residual có chiếu tuyến tính
(áp dụng khi kích thước đầu vào/đầu ra không đồng nhất)

$$y = F(x, W) + W_s x$$

- $x, y$: tensor đầu vào/đầu ra của khối
- $F(x, W)$: ánh xạ phần dư qua các lớp tích chập
- $W_s$: ma trận chiếu điều chỉnh kích thước/số kênh

### 4.3 Global Average Pooling

Sau các lớp tích chập sâu:

$$\mathbf{z}_t = \text{GAP}(F_t), \quad \mathbf{z}_t \in \mathbb{R}^{2048}$$

- $F_t = \phi_{\text{res}}(I_t')$: Feature Map sau Backbone
- $\mathbf{z}_t$: vector ngữ cảnh phẳng chung, **kích thước 2048**

### ✅ Checklist đối chiếu
- [ ] Backbone là ResNet50 (không phải ResNet34, ResNet101,...)
- [ ] Vector $\mathbf{z}_t$ sau GAP có kích thước đúng bằng **2048**
- [ ] GAP được áp dụng trước khi vào các nhánh đa nhiệm

---

## 5. Mô-đun Định Vị Đa Nhiệm Song Song (Multi-task Heads)

**Đầu vào chung:** $\mathbf{z}_t \in \mathbb{R}^{2048}$

### 5.1 Nhánh 1 — Phân loại Topo rời rạc

**Tính logit:**
$$\mathbf{o}_t = W_c \mathbf{z}_t + b_c$$

**Chuyển sang phân phối xác suất:**
$$\hat{\mathbf{q}}_t = \text{softmax}(\mathbf{o}_t)$$

**Nhãn dự đoán cuối cùng:**
$$\hat{c}_t = \arg\max_k \hat{q}_k$$

### 5.2 Nhánh 2 — Hồi quy tọa độ liên tục

$$\hat{\mathbf{p}}_t = g_r(\mathbf{z}_t) = [\hat{x}_t,\ \hat{y}_t]^T$$

### 5.3 Chuẩn hóa Min-Max (bắt buộc)

**Chuẩn hóa trước khi tính Loss:**
$$x_n = \frac{x - x_{\min}}{x_{\max} - x_{\min}}, \quad y_n = \frac{y - y_{\min}}{y_{\max} - y_{\min}}$$

**Giải chuẩn hóa khi Inference (về đơn vị mét):**
$$\hat{x} = \hat{x}_n \cdot (x_{\max} - x_{\min}) + x_{\min}$$
$$\hat{y} = \hat{y}_n \cdot (y_{\max} - y_{\min}) + y_{\min}$$

### ✅ Checklist đối chiếu
- [ ] Hai nhánh MLP **độc lập** nhau, cùng nhận $\mathbf{z}_t$
- [ ] Nhánh phân loại dùng Softmax (không phải Sigmoid)
- [ ] Nhánh hồi quy đầu ra **2 giá trị** $[\hat{x}_t, \hat{y}_t]$
- [ ] Tọa độ Ground Truth phải được **Min-Max normalize** trước khi tính $\mathcal{L}_{reg}$
- [ ] Inference phải **denormalize** đầu ra về đơn vị mét
- [ ] $x_{\min}, x_{\max}, y_{\min}, y_{\max}$ phải được lưu lại từ tập train để dùng khi inference

---

## 6. Hàm Mất Mát Đa Nhiệm Tích Hợp (Multi-task Loss)

### 6.1 Hàm Loss tổng hợp

$$\mathcal{L} = \lambda_c \mathcal{L}_{cls} + \lambda_r \mathcal{L}_{reg}$$

> **Ràng buộc bắt buộc:** $\lambda_c > \lambda_r$

### 6.2 Mất mát phân loại khu vực

$$\mathcal{L}_{cls} = -\sum_{k=1}^{K} y_k \log(\hat{q}_k)$$

- Sử dụng **Categorical Cross-Entropy Loss**
- $y_k$: nhãn One-hot của phân khu thực tế tại thời điểm $t$

### 6.3 Mất mát hồi quy vị trí

$$\mathcal{L}_{reg} = \|\hat{\mathbf{p}}_t - \mathbf{p}_t\|_2^2$$

- Sử dụng **MSE Loss** (khoảng cách Euclid bình phương)
- Cả $\hat{\mathbf{p}}_t$ lẫn $\mathbf{p}_t$ đều ở dạng **đã chuẩn hóa** khi tính loss

### ✅ Checklist đối chiếu
- [ ] Xác nhận `lambda_c > lambda_r` trong config/hyperparameters
- [ ] $\mathcal{L}_{cls}$ dùng Cross-Entropy (không phải Binary CE)
- [ ] $\mathcal{L}_{reg}$ dùng MSE (không phải MAE hay Huber)
- [ ] Loss tổng = tổng **có trọng số** của hai thành phần, không phải trung bình đơn thuần

---

## 7. Tổng Quan Luồng Dữ Liệu

```
Input: I_t (H×W×3)
    │
    ▼
[YOLOv8 Detection]
    │  → Lọc đối tượng động: s_i ≥ 0.6, l_i ∈ C_d
    │  → Mở rộng bbox: b_i' = (x-k, y-k, w+2k, h+2k)
    │  → Tạo Binary Mask: M_t (H×W)
    │
    ▼
[Spatial Attention]
    │  → A_t = 1 - α·M_t
    │  → Image-level: I_t' = I_t ⊙ A_t
    │     hoặc Feature-level: F_t' = F_t ⊙ A_t'
    │
    ▼
[ResNet50 Backbone]
    │  → F_t = φ_res(I_t')
    │  → z_t = GAP(F_t)  ∈ ℝ²⁰⁴⁸
    │
    ├──────────────────────┬──────────────────────┐
    ▼                      ▼
[Classification Head]   [Regression Head]
    o_t = W_c·z_t + b_c   p̂_t = g_r(z_t)
    q̂_t = softmax(o_t)    = [x̂_t, ŷ_t]ᵀ
    ĉ_t = argmax(q̂_t)
    │                      │
    ▼                      ▼
  L_cls (CE)           L_reg (MSE)
         \                /
          ▼              ▼
    L = λ_c·L_cls + λ_r·L_reg   (λ_c > λ_r)
```

---

## 8. Bảng Tham Số và Hằng Số Quan Trọng

| Tham số | Giá trị / Ràng buộc | Ghi chú |
|---|---|---|
| $\delta$ | `0.6` (cố định) | Ngưỡng confidence YOLOv8 |
| $\alpha$ | $\in [0, 1]$ | Hệ số suy giảm attention |
| $k$ | Tunable | Số pixel mở rộng bbox |
| $K$ | Tùy dataset | Số khu vực topo |
| $\dim(\mathbf{z}_t)$ | `2048` | Output dimension của ResNet50 GAP |
| $\lambda_c$ | $> \lambda_r$ | Trọng số loss phân loại |
| $\lambda_r$ | $< \lambda_c$ | Trọng số loss hồi quy |
| Normalize range | $[0, 1]$ | Min-Max cho tọa độ |

---

*Tài liệu này được tạo tự động từ đặc tả lý thuyết gốc. Mọi thay đổi kiến trúc cần cập nhật đồng bộ vào file này.*
