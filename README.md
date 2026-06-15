# Medical Datamining & AI Diagnosis Support

Đồ án môn **Khai Phá Dữ Liệu** (Data Mining) — Ứng dụng hỗ trợ chẩn đoán và tư vấn y khoa chuyên sâu, tích hợp toàn diện các kỹ thuật tiền xử lý dữ liệu, phân lớp, khai phá luật kết hợp, gom cụm bệnh nhân, phân tích hồi quy và kiểm định thống kê. Hệ thống kết hợp Trợ lý bác sĩ AI thông minh từ mô hình ngôn ngữ lớn Google Gemini để tối ưu hóa khả năng ra quyết định lâm sàng.

---

## 📋 Mục tiêu & Tính năng chính của hệ thống

1. **Tiền xử lý & Làm sạch dữ liệu tự động:**
   - Điền khuyết dữ liệu lâm sàng tự động bằng trung vị (Median) cho biến số liên tục và yếu vị (Mode) cho biến phân loại.
   - Loại bỏ và co hẹp các giá trị ngoại lệ (Outliers) bằng phương pháp biên giới hạn IQR (Capping).
   - Chuẩn hóa các biến liên tục bằng `StandardScaler` phục vụ huấn luyện và chẩn đoán.
   - Loại bỏ trùng lặp dữ liệu thô (`drop_duplicates()`) một cách nhất quán trên mọi module phân tích để tránh rò rỉ dữ liệu (data leakage).

2. **Mô hình Phân lớp (Classification):**
   - Huấn luyện song song và đánh giá hiệu năng chéo của 5 thuật toán học máy hàng đầu: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Naive Bayes.
   - Lưu trữ tất cả mô hình và tự động lựa chọn mô hình tốt nhất dựa trên chỉ số **Recall** (độ nhạy) nhằm giảm thiểu tối đa tỷ lệ bỏ sót ca bệnh trong chẩn đoán y khoa.

3. **Giao diện Trực quan hóa Tương tác (Plotly Interactive Charts):**
   - Biểu đồ phân phối xác suất chẩn đoán và biểu đồ so sánh hiệu năng mô hình được chuyển đổi hoàn toàn sang **Plotly**.
   - Người dùng có thể di chuột (hover) để xem trực tiếp các số liệu chi tiết (xác suất, điểm số Accuracy/Precision/Recall/F1) thay vì xem biểu đồ ảnh tĩnh thụ động.

4. **Trợ lý tim mạch AI (Gemini AI Advisor):**
   - Tích hợp mô hình **Gemini 2.5 Flash** để phân tích sâu sắc 13 thông số lâm sàng đầu vào cùng kết quả dự đoán của AI, từ đó đề xuất chế độ ăn uống, cường độ thể chất và khuyến nghị kiểm tra chuyên sâu phù hợp cho từng bệnh nhân.
   - Cơ chế bảo mật API Key ngay trên giao diện cùng thuật toán **Cache kết quả chẩn đoán thông minh (MD5 Hashing)** giúp tối ưu số lượng token gửi lên và tăng tốc thời gian phản hồi.

5. **Lịch sử chẩn đoán & Xuất dữ liệu toàn diện:**
   - Lưu trữ tạm thời toàn bộ lịch sử chẩn đoán trong phiên làm việc.
   - Cho phép xem lại lời khuyên đóng góp từ Gemini AI trực tiếp trong các thẻ lịch sử bệnh nhân.
   - Hỗ trợ xuất kết quả (từng ca đơn lẻ hoặc toàn bộ lịch sử) ra file CSV tích hợp kèm theo lời khuyên của Gemini AI dưới cột `gemini_advice`.
   - Widget kéo thả cho phép nạp lại file lịch sử chẩn đoán cũ (.csv) để xem lại và tiếp tục chẩn đoán.

6. **Khai phá luật kết hợp (Association Rules):**
   - Rời rạc hóa dữ liệu chuẩn y khoa, ánh xạ nhãn chữ đầy đủ và áp dụng thuật toán Apriori & FP-Growth từ thư viện `mlxtend` để tìm kiếm các mối quan hệ triệu chứng ẩn sâu.

7. **Gom cụm bệnh nhân (Clustering):**
   - Thuật toán K-Means Clustering kết hợp PCA trực quan hóa 2D phân nhóm bệnh nhân thành 3 cụm chính (trẻ tuổi, đau ngực điển hình, lớn tuổi nguy cơ cao).

8. **Kiểm định Thống kê & Hồi quy:**
   - Kiểm định giả thuyết **ANOVA** và **Chi-Square** phân tích tính độc lập/ảnh hưởng của các biến đối với bệnh tim mạch.
   - Huấn luyện mô hình hồi quy tuyến tính đơn biến, đa biến và hồi quy đa thức bậc 2 dự đoán nhịp tim tối đa gắng sức (`thalach`).

---

## 👥 Thành viên nhóm thực hiện

| Thành viên   | Vai trò |
| ------------ | ------- |
| **Thành Danh** | Phát triển mô hình chẩn đoán phân lớp, tích hợp pipeline, nâng cấp Streamlit |
| **Hồng Vỹ**     | Thu thập dữ liệu, trực quan hóa và phân tích EDA |
| **Quốc An**      | Tiền xử lý dữ liệu, làm sạch và xử lý outliers |
| **Minh Thiện**   | Đánh giá mô hình học máy, trực quan hóa ROC/Confusion Matrix |
| **Quang Ngọc**   | Khai phá luật kết hợp Apriori & FP-Growth |
| **Lê Hậu**       | Phát triển giao diện dashboard Streamlit, slide báo cáo |

---

## 📁 Cấu trúc thư mục dự án

```
medical-datamining-diagnosis-support/
├── data/
│   ├── raw/                    # Dữ liệu gốc (heart.csv - đã gộp)
│   └── processed/              # Dữ liệu sạch cho phân lớp (heart_processed.csv)
│                               # Dữ liệu rời rạc hóa cho luật kết hợp (heart_association.csv)
├── src/
│   ├── app/                    # Mã nguồn giao diện Streamlit (app.py, display_result.py, ui_components.py)
│   ├── association_rules/      # Xử lý luật kết hợp Apriori & FP-Growth
│   ├── evaluation/             # Đánh giá phân lớp (Confusion Matrix, ROC Curve)
│   ├── models/                 # Huấn luyện mô hình, gom cụm, hồi quy, dự đoán (predict.py)
│   ├── preprocessing/          # Module làm sạch, scale dữ liệu (clean_data.py, transform_data.py)
│   └── utils/                  # Kiểm định ANOVA/Chi-Square (statistical_tests.py), cấu hình (config.py)
├── outputs/
│   ├── figures/                # Thư mục lưu biểu đồ đánh giá, biểu đồ phân cụm PCA, biểu đồ hồi quy
│   │   └── eda/                # Chứa 18 biểu đồ EDA sạch của dữ liệu
│   ├── metrics/                # Lưu bảng so sánh phân lớp, hồi quy, JSON thống kê ANOVA/Chi2, tóm tắt cụm
│   └── models/                 # Chứa mô hình học máy (.pkl) và scaler chuẩn hóa (.pkl)
├── main.py                     # File chạy pipeline chính tích hợp tự động toàn diện
├── requirements.txt            # Danh sách thư viện Python cần cài đặt
└── README.md
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Cài đặt thư viện yêu cầu
Khuyến nghị sử dụng Python từ phiên bản **3.9** trở lên (hỗ trợ đầy đủ đến Python 3.14).
```bash
python -m pip install -r requirements.txt
```

### 2. Khởi chạy toàn bộ Pipeline tự động
Chạy lệnh sau để thực thi toàn diện luồng xử lý từ tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá, khai phá luật kết hợp, gom cụm bệnh nhân đến kiểm định thống kê và hồi quy:
```bash
python main.py
```

### 3. Khởi chạy riêng lẻ từng Module qua CLI
Bạn có thể tùy ý gọi thực thi từng phần của dự án:
- `python main.py --preprocess` : Thực hiện tiền xử lý dữ liệu (sinh dữ liệu processed và association).
- `python main.py --train`      : Huấn luyện so sánh các mô hình máy học.
- `python main.py --evaluate`   : Xuất biểu đồ Confusion Matrix và ROC.
- `python main.py --rules`      : Khai phá luật kết hợp Apriori & FP-Growth.
- `python main.py --cluster`    : Thực hiện thuật toán gom cụm K-Means & PCA.
- `python main.py --analytics`  : Chạy kiểm định thống kê ANOVA/Chi-Square & Hồi quy.
- `python main.py --app`        : Khởi chạy bảng điều khiển giao diện web Streamlit.

### 4. Khởi chạy trực tiếp giao diện Web Dashboard
Bạn có thể khởi chạy Streamlit trực tiếp qua lệnh sau:
```bash
python -m streamlit run src/app/app.py
```

---

## 📊 Luồng xử lý & Chẩn đoán dữ liệu

```
                             [ data/raw/heart.csv ]
                                       │
                                       ▼ (src/preprocessing/clean_data.py)
                             [ Làm sạch & loại trùng ] (599 mẫu unique)
                                   /         \
    (scale & keep numericals)     /           \  (discretize & label mapping)
                                 ▼             ▼
                   [ heart_processed.csv ]    [ heart_association.csv ]
                          /          \                     │
     (train models)      /            \ (regression)       ▼ (mlxtend)
                        ▼              ▼             [ Apriori & FP-Growth ]
                 [ src/models ]   [ src/models ]           │
               (classification)   (regression.py)          ▼
                        │              │             [ outputs/rules ]
                        ▼              ▼                   │
               [ src/evaluation ] [ outputs/figures ]      │
                        │              │                   │
                        ▼              ▼                   │
               [ outputs/models ]------┼───────────────────┘
                        │              │
                        ▼ (Scaler & Classifier)
                 [ predict_one ] (src/models/predict.py)
                        │
                        ▼ (Inputs, Predict Result, API Key)
                 [ display_result.py ] (src/app/display_result.py)
                        │
                        ▼ (Request to Google Gemini 2.5 API)
             [ Gemini AI Advisor ] (Tư vấn tim mạch cá nhân hóa)
                        │
                        ▼
                 [ Streamlit UI ] (Giao diện Plotly động & Lịch sử)
```

---

## 📈 Kết quả huấn luyện & Khai phá tri thức

- **Mô hình phân lớp tốt nhất:** **K-Nearest Neighbors (KNN)** được chọn làm mô hình tối ưu cho chẩn đoán với độ nhạy (Recall) cao nhất: **83.33%**, Accuracy **80.00%**, Precision **78.12%**, và F1-Score **80.65%**. Random Forest bám sát vị trí thứ hai với Recall **81.67%**.
- **Khai phá luật kết hợp:** Trích xuất thành công 568 luật kết hợp tim mạch giá trị có chỉ số Lift > 1.2. Ví dụ: luật `{thal_reversible_defect, exang_yes} -> {cp_asymptomatic, heart_disease}` cho thấy những bệnh nhân đau ngực không triệu chứng nhưng bị thiếu máu cơ tim khi gắng sức có khả năng mắc bệnh mạch vành thực tế cao gấp 2.38 lần người bình thường.
- **Gom cụm bệnh nhân (K-Means):** Silhouette Score đạt **0.1339** chia thành 3 cụm rõ rệt từ 599 bệnh nhân:
  - **Cụm 0 (Bệnh nhân trẻ tuổi):** Tuổi trung bình ~48 tuổi, nhịp tim gắng sức tối đa cao (~165 bpm), tổn thương ST cơ tim rất thấp (~0.47) -> Tỷ lệ mắc bệnh thực tế: **47.4%**.
  - **Cụm 1 (Nhóm đau thắt ngực điển hình):** Bệnh nhân trung niên ~57 tuổi, chỉ số đau ngực trung bình thấp, nhịp tim gắng sức đạt ~141 bpm -> Tỷ lệ mắc bệnh thực tế: **31.8%**.
  - **Cụm 2 (Bệnh nhân lớn tuổi - Nguy cơ cao):** Nhóm lớn tuổi nhất ~60 tuổi, huyết áp (~137 mmHg) và cholesterol (~259 mg/dl) cao nhất hệ thống, nhịp tim gắng sức tối đa thấp nhất (~138 bpm), chỉ số tổn thương ST cao nhất (~1.47) -> Tỷ lệ mắc bệnh thực tế: **70.3%**.
- **Kiểm định Thống kê & Hồi quy:**
  - Kiểm định Chi-Square cho thấy các biến phân loại như `cp` (loại đau ngực, $p \approx 4.25 \times 10^{-14}$), `restecg` (điện tâm đồ, $p \approx 6.33 \times 10^{-5}$), `slope` (độ dốc ST, $p \approx 1.03 \times 10^{-14}$) và `thal` (thalassemia, $p \approx 1.32 \times 10^{-35}$) có liên quan có ý nghĩa thống kê rất mạnh với nguy cơ bệnh tim. Các biến liên tục kiểm định qua ANOVA đều cho thấy sự khác biệt trung bình không có ý nghĩa thống kê ($p > 0.05$).
  - Mô hình Hồi quy đa biến chứng minh hiệu năng dự đoán nhịp tim tối đa gắng sức (`thalach`) tối ưu nhất (MSE: 329.64, R²: 27.6%) dựa trên các thông số tuổi, huyết áp, cholesterol và oldpeak.

---

## ⚠️ Tuyên bố miễn trừ trách nhiệm y khoa

Mọi thông tin chẩn đoán, phân loại xác suất và lời khuyên y học từ Trợ lý Gemini AI trong dự án này **chỉ mang tính chất tham khảo học thuật phục vụ báo cáo đồ án môn học**. Hệ thống tuyệt đối không thay thế cho các chỉ định lâm sàng, xét nghiệm chuyên sâu và chẩn đoán y khoa trực tiếp từ các bác sĩ chuyên khoa tim mạch.