# Walkthrough: Model Evaluation Results & Data Extraction

Báo cáo kết quả đánh giá mô hình định vị robot trong nhà dựa trên file trọng số `best_model.pth` và lịch sử huấn luyện `training_history.txt`, cùng các bảng thống kê và chỉ số đánh giá chi tiết cho cả hai nhánh nhiệm vụ (Topological Classification và Coordinate Regression).

---

## 1. Biểu đồ Huấn luyện Hàm Loss & Accuracy (Figure 1)
Biểu đồ huấn luyện thể hiện sự hội tụ của mô hình qua 80 epochs cho cả 2 nhánh (Phân loại Topo và Hồi quy tọa độ) trên tập Train và tập Validation:

![Training Loss and Accuracy Curves](figure1/training_loss_curves.png)

> *Lưu ý*: Trục Y của các biểu đồ Loss đã được giới hạn một cách tự động để tránh việc các điểm nhiễu (outliers) làm biến dạng đồ thị, giúp quan sát rõ ràng xu hướng hội tụ của mô hình.

---

## 2. Báo cáo Tóm tắt Hiệu năng Mô hình (Model Performance Summary Report)

Dưới đây là tổng hợp chỉ số Precision, Recall, F1-score và Accuracy của hai nhánh mô hình:

### Nhánh 1: Phân loại Topological Area (Phân loại vùng)
*   **Accuracy (Độ chính xác toàn cục)**: **100.0000%**
*   **Precision (Macro Average)**: **100.0000%**
*   **Recall (Macro Average)**: **100.0000%**
*   **F1-Score (Macro Average)**: **100.0000%**

### Nhánh 2: Hồi quy Vị trí (Coordinate Positioning)
Đối với nhánh hồi quy, hiệu năng được đánh giá thông qua sai số khoảng cách Euclidean và tỉ lệ định vị thành công (Localization Success Rate / Accuracy) dưới các ngưỡng sai số chấp nhận được (Tolerance Thresholds):

| Ngưỡng Sai số (Threshold) | Accuracy (Tỉ lệ thành công) | Precision (Độ chính xác) | Recall (Độ nhạy) | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **0.5 m** | 45.37% | 100.00% | 45.37% | 62.42% |
| **1.0 m** | **83.90%** | **100.00%** | **83.90%** | **91.25%** |
| **1.5 m** | 97.32% | 100.00% | 97.32% | 98.64% |
| **2.0 m** | 99.51% | 100.00% | 99.51% | 99.76% |

> *Lưu ý*: Chỉ số Precision, Recall, F1 của nhánh hồi quy được tính toán bằng cách coi việc robot ước lượng vị trí lệch nhỏ hơn hoặc bằng ngưỡng sai số là lớp Positive (Thành công), và lệch vượt quá ngưỡng là lớp Negative (Thất bại). Do mục tiêu là định vị cho 100% mẫu thử nên nhãn thực tế luôn là Positive, dẫn tới Precision đạt 100% và Recall trùng với Accuracy.

*   File báo cáo chi tiết: [model_performance_summary.txt](figure6/model_performance_summary.txt).

---

## 3. Bảng 3: Thống kê sai số định vị chi tiết (X, Y, Euclidean)
Bảng 3 thống kê chi tiết sai số vị trí theo từng trục X, Y và khoảng cách Euclid (đơn vị: mét):

| Metrics | A | B | C | D | E | Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Average error X (m) | 0.395365 | 0.231299 | 0.242672 | 0.330781 | 0.288704 | 0.295373 |
| 2. Maximum error X (m) | 1.265546 | 0.897990 | 0.631478 | 1.014481 | 1.353651 | 1.353651 |
| 3. Minimum error X (m) | 0.000630 | 0.000042 | 0.027274 | 0.000383 | 0.002831 | 0.000042 |
| 4. Error deviation in X (m) | 0.314785 | 0.175079 | 0.156516 | 0.238145 | 0.277342 | 0.244442 |
| 5. Average error Y (m) | 0.335238 | 0.533836 | 0.437103 | 0.499071 | 0.667170 | 0.487449 |
| 6. Maximum error Y (m) | 1.191291 | 1.675127 | 1.400230 | 1.705570 | 2.371574 | 2.371574 |
| 7. Minimum error Y (m) | 0.002841 | 0.009935 | 0.013775 | 0.008957 | 0.030472 | 0.002841 |
| 8. Error deviation in Y (m) | 0.266936 | 0.378787 | 0.328979 | 0.340890 | 0.557138 | 0.387303 |
| **9. Average error Euclidean (m)** | **0.572579** | **0.606230** | **0.527078** | **0.647703** | **0.783686** | **0.618743** |
| 10. Maximum error Euclidean (m) | 1.665008 | 1.713048 | 1.438586 | 1.950969 | 2.374933 | 2.374933 |
| 11. Minimum error Euclidean (m) | 0.020986 | 0.036131 | 0.078641 | 0.095212 | 0.043088 | 0.020986 |
| 12. Error deviation Euclidean (m) | 0.333460 | 0.380915 | 0.323825 | 0.334496 | 0.549203 | 0.389570 |

*   File dữ liệu nguồn: [table3_metrics.csv](figure6/table3_metrics.csv) và [table3_metrics.txt](figure6/table3_metrics.txt).

---

## 4. Dữ liệu trích xuất cho các biểu đồ (Figure 2, 3, 5)

*   **Figure 2 (Confusion Matrix)**: [confusion_matrix.txt](figure2/confusion_matrix.txt)
*   **Figure 3 (Trajectory by Area)**: [trajectory_by_area.txt](figure3/trajectory_by_area.txt)
*   **Figure 5 (Trajectory Comparison)**: [trajectory_comparison.txt](figure5/trajectory_comparison.txt)

---

## 5. Biểu đồ trực quan hóa (Figures)

### Figure 2: Ma trận nhầm lẫn Topological Area
![Topological Confusion Matrix](figure2/confusion_matrix.png)

### Figure 3: Quỹ đạo dự đoán phân màu theo khu vực
![Trajectory by Area](figure3/trajectory_by_area.png)

### Figure 5: So sánh Quỹ đạo CNN dự đoán với Ground Truth từ Encoder 
![Trajectory Comparison](figure5/trajectory_comparison.png)

### Figure 6: Sai số Euclidean trung bình và độ lệch chuẩn theo từng Area
![Translation Error Comparison](figure6/errors_bar_chart.png)
