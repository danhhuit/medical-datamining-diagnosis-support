# Medical Datamining & AI Diagnosis Support

Đồ án môn **Khai Phá Dữ Liệu** — Ứng dụng hỗ trợ chẩn đoán và tư vấn y khoa chuyên sâu, tích hợp toàn diện các kỹ thuật tiền xử lý dữ liệu, phân lớp, khai phá luật kết hợp, gom cụm, phân tích hồi quy/kiểm định thống kê, kết hợp cùng Trợ lý bác sĩ AI từ mô hình ngôn ngữ lớn Google Gemini.

---

## 📋 Mục tiêu & Tính năng chính của hệ thống

1. **Tiền xử lý & Làm sạch dữ liệu:**
   - Điền khuyết dữ liệu lâm sàng (Median/Mode imputation).
   - Lọc và xử lý giá trị ngoại lệ (Outliers) bằng phương pháp biên giới hạn IQR (Capping).
   - Chuẩn hóa các biến liên tục bằng `StandardScaler` phục vụ huấn luyện và dự đoán mô hình.
   
2. **Mô hình Phân lớp (Classification):**
   - Huấn luyện và đánh giá hiệu năng chéo của 5 thuật toán học máy: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Naive Bayes.
   - Tự động lưu mô hình tốt nhất (dựa trên chỉ số Recall để tránh bỏ sót bệnh nhân) dưới dạng Joblib Artifact.

3. **Trợ lý tim mạch AI (Gemini AI Advisor):**
   - Tích hợp mô hình ngôn ngữ lớn **Gemini 2.5 Flash** thay thế hoàn toàn hệ thống lời khuyên tĩnh.
   - Phân tích sâu sắc 13 thông số đầu vào và kết quả chẩn đoán của mô hình học máy để đưa ra lời khuyên y khoa cá nhân hóa (chế độ dinh dưỡng, cường độ thể chất, các xét nghiệm sâu hơn cần thiết).
   - Cơ chế bảo mật API Key trong giao diện cùng thuật toán **Cache kết quả chẩn đoán thông minh (MD5 Hashing)** giúp tối ưu số lượng token gửi lên và tăng tốc thời gian phản hồi.

4. **Bộ chọn ca bệnh mẫu (Auto-fill Presets):**
   - Hộp chọn nhanh 3 ca lâm sàng thực tế được trích xuất từ tập dữ liệu gốc ngay trên form nhập liệu:
     * **Ca 1 (Lành mạnh)**: Xác suất nguy cơ cực thấp (~2.4%).
     * **Ca 2 (Ranh giới / 50-50)**: Xác suất nằm sát ngưỡng ranh giới quyết định (~49.4%).
     * **Ca 3 (Nguy cơ cao / Mắc bệnh)**: Xác suất bệnh lý mạch vành rất cao (~97.5%).
   - Cơ chế liên kết dữ liệu hai chiều (Two-way data binding) cho phép người dùng điều chỉnh thông số tự do sau khi nạp ca bệnh mẫu.

5. **Lịch sử chẩn đoán nâng cao:**
   - Lưu trữ tạm thời toàn bộ các ca bệnh đã chẩn đoán trong phiên làm việc.
   - Hỗ trợ xuất toàn bộ lịch sử ra file CSV với tên file tự động kèm ngày giờ: `Lich_su_Chan_doan_YYYYMMDD_HHMMSS.csv`.
   - Widget kéo thả cho phép người dùng nạp lại file lịch sử cũ vào ứng dụng để khôi phục và tiếp tục xem lại thông tin.

6. **Khai phá luật kết hợp (Association Rules):**
   - Rời rạc hóa dữ liệu chuẩn y khoa, ánh xạ mã hóa nhãn chữ đầy đủ và chạy thuật toán Apriori & FP-Growth từ thư viện `mlxtend` để tìm kiếm tri thức ẩn từ các nhóm thuộc tính liên kết.

7. **Gom cụm bệnh nhân (Clustering):**
   - Thuật toán K-Means Clustering phân nhóm bệnh nhân tương đồng lâm sàng thành 3 cụm chính nguy cơ (Thấp, Trung bình, Cao) kết hợp PCA trực quan hóa 2D.

8. **Kiểm định Thống kê & Hồi quy:**
   - Kiểm định **ANOVA** và **Chi-Square** độc lập phân tích độ quan trọng của thuộc tính.
   - Hồi quy tuyến tính đơn biến, đa biến và hồi quy đa thức bậc 2 dự đoán nhịp tim tối đa gắng sức (`thalach`).

---

## 👥 Thành viên nhóm thực hiện

| Thành viên   | Vai trò |
| ------------ | ------- |
| Thành Danh   | Mô hình chẩn đoán phân lớp, tích hợp pipeline hệ thống |
| Hồng Vỹ     | Thu thập dữ liệu, trực quan hóa và phân tích EDA |
| Quốc An      | Tiền xử lý dữ liệu, làm sạch và xử lý outliers |
| Minh Thiện   | Đánh giá mô hình học máy, trực quan hóa ROC/Confusion Matrix |
| Quang Ngọc   | Khai phá luật kết hợp Apriori & FP-Growth |
| Lê Hậu       | Phát triển giao diện dashboard Streamlit, slide báo cáo |

---

## 📁 Cấu trúc thư mục dự án

```
medical-datamining-diagnosis-support/
├── data/
│   ├── raw/                    # Dữ liệu gốc (heart.csv)
│   └── processed/              # Dữ liệu sạch cho phân lớp (heart_processed.csv)
│                               # Dữ liệu rời rạc hóa cho luật kết hợp (heart_association.csv)
├── notebooks/
│   ├── eda/                    # Notebook khám phá dữ liệu (eda_dataset.ipynb)
│   ├── preprocessing/          # Thử nghiệm tiền xử lý
│   ├── modeling/               # Thử nghiệm mô hình
│   └── association_rules/      # Thử nghiệm luật kết hợp
├── src/
│   ├── app/                    # Mã nguồn giao diện Streamlit (app.py, display_result.py, ui_components.py)
│   ├── association_rules/      # Xử lý luật kết hợp Apriori & FP-Growth
│   ├── evaluation/             # Đánh giá phân lớp (Confusion Matrix, ROC Curve)
│   ├── models/                 # Huấn luyện mô hình, gom cụm (clustering.py), hồi quy (regression.py), dự đoán (predict.py)
│   ├── preprocessing/          # Module làm sạch, scale dữ liệu (clean_data.py, transform_data.py)
│   └── utils/                  # Kiểm định ANOVA/Chi-Square (statistical_tests.py), cấu hình (config.py)
├── outputs/
│   ├── figures/                # Thư mục lưu biểu đồ đánh giá, biểu đồ phân cụm PCA, biểu đồ hồi quy
│   ├── metrics/                # Lưu bảng so sánh phân lớp, hồi quy, JSON thống kê ANOVA/Chi2, tóm tắt cụm
│   └── models/                 # Chứa mô hình học máy (.pkl) và scaler chuẩn hóa (.pkl)
├── main.py                     # File chạy pipeline chính tích hợp tự động toàn diện
├── requirements.txt            # Danh sách thư viện Python cần cài đặt
└── .gitignore
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Cài đặt thư viện yêu cầu
Khuyến nghị sử dụng Python từ phiên bản **3.9** trở lên. Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
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
streamlit run src/app/app.py
```

---

## 📊 Luồng xử lý & Chẩn đoán dữ liệu

```
                             [ data/raw/heart.csv ]
                                       │
                                       ▼ (src/preprocessing/clean_data.py)
                             [ Làm sạch & lọc Outliers ]
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
             [ Gemini AI Advisor ] (Empathetic Cardiology Advice)
                       │
                       ▼
                 [ Streamlit UI ] (Dashboard hiển thị sạch, không icon)
```

---

## 📈 Kết quả huấn luyện & Khai phá tri thức

- **Mô hình chẩn đoán phân lớp tốt nhất:** **Logistic Regression** đạt chỉ số Accuracy **91.67%** và F1-Score **90.20%**. Mô hình được tối ưu và lựa chọn dựa trên chỉ số **Recall** đạt **82.14%** nhằm giảm thiểu tối đa tỷ lệ bỏ sót bệnh nhân nguy cơ (False Negative) trong thực hành y tế.
- **Khai phá luật kết hợp:** Nhờ rời rạc hóa và mã hóa nhãn chữ chuẩn xác, thuật toán đã loại bỏ hoàn toàn các luật rác (luật luôn đạt 100% True). Trích xuất được các luật kết hợp tim mạch giá trị, ví dụ: `{thal_reversible_defect, exang_yes} -> {cp_asymptomatic, heart_disease}` với độ tin cậy **82.75%** và hệ số Lift **2.38** (biểu thị bệnh nhân đau ngực không triệu chứng nhưng bị thiếu máu cơ tim khi gắng sức có khả năng mắc bệnh mạch vành cao gấp 2.38 lần bình thường).
- **Gom cụm bệnh nhân (K-Means):** Silhouette Score đạt **0.1321** chia thành 3 cụm:
  - Cụm 0 (Nguy cơ thấp): Nhóm trẻ tuổi, nhịp tim gắng sức tối đa đạt mức cao (~162 bpm) -> Tỷ lệ mắc bệnh thực tế: **22.9%**.
  - Cụm 1 (Nguy cơ cao): Nhóm lớn tuổi, tổn thương tim ST cao (~1.89 mm), nhịp tim gắng sức kém -> Tỷ lệ mắc bệnh thực tế: **90.2%**.
  - Cụm 2 (Quá tải mạch): Nhóm lớn tuổi bị huyết áp và cholesterol cao nhưng cơ tim ít bị tổn thương thực thể -> Tỷ lệ mắc bệnh thực tế: **22.3%**.
- **Kiểm định Thống kê & Hồi quy:**
  - Kiểm định ANOVA xác nhận các biến liên tục như `age`, `trestbps`, `thalach`, `oldpeak` có phân bố trung vị hoàn toàn khác biệt có ý nghĩa thống kê giữa hai nhóm bình thường và mắc bệnh ($p < 0.05$).
  - Mô hình Hồi quy đa biến chứng minh hiệu năng dự đoán nhịp tim tối đa gắng sức (`thalach`) ổn định nhất dựa trên các thông số tuổi và các đặc trưng lâm sàng khác.

---

## ⚠️ Tuyên bố miễn trừ trách nhiệm y khoa

Mọi thông tin chẩn đoán, phân loại xác suất và lời khuyên y học từ Trợ lý Gemini AI trong dự án này **chỉ mang tính chất tham khảo học thuật phục vụ báo cáo đồ án môn học**. Hệ thống tuyệt đối không thay thế cho các chỉ định lâm sàng, xét nghiệm chuyên sâu và chẩn đoán y khoa trực tiếp từ các bác sĩ chuyên khoa tim mạch.