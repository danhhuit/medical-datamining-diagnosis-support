# Medical Datamining Diagnosis Support 🏥

Đồ án môn **Khai Phá Dữ Liệu** — Ứng dụng hỗ trợ chẩn đoán và điều trị y khoa.

## 📋 Mục tiêu

- Sử dụng kỹ thuật khai phá dữ liệu để hỗ trợ chẩn đoán bệnh tim
- Áp dụng các thuật toán ML: Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes
- Khai phá luật kết hợp bằng Apriori và FP-Growth
- Xây dựng ứng dụng demo trực quan bằng Streamlit

## 👥 Thành viên

| Thành viên   | Vai trò                                    |
| ------------ | ------------------------------------------ |
| Thành Danh   | Mô hình chẩn đoán, tích hợp pipeline      |
| Hồng Vỹ     | Thu thập dữ liệu, EDA                     |
| Quốc An      | Tiền xử lý dữ liệu                        |
| Minh Thiện   | Đánh giá mô hình                           |
| Quang Ngọc   | Khai phá luật kết hợp                      |
| Lê Hậu       | Giao diện demo, slide                      |

## 📁 Cấu trúc thư mục

```
medical-datamining-diagnosis-support/
├── data/
│   ├── raw/                    # Dữ liệu gốc (heart.csv)
│   └── processed/              # Dữ liệu đã tiền xử lý
├── notebooks/
│   ├── eda/                    # Khám phá dữ liệu
│   ├── preprocessing/          # Thử nghiệm tiền xử lý
│   ├── modeling/               # Thử nghiệm mô hình
│   └── association_rules/      # Thử nghiệm luật kết hợp
├── src/
│   ├── app/                    # Ứng dụng demo Streamlit
│   ├── association_rules/      # Khai phá luật kết hợp
│   ├── evaluation/             # Đánh giá mô hình
│   ├── models/                 # Huấn luyện & dự đoán
│   ├── preprocessing/          # Tiền xử lý dữ liệu
│   └── utils/                  # Hàm dùng chung
├── outputs/
│   ├── figures/                # Biểu đồ
│   ├── metrics/                # Chỉ số đánh giá
│   ├── models/                 # Model đã lưu
│   └── rules/                  # Luật kết hợp
├── reports/                    # Báo cáo
├── slides/                     # Slide thuyết trình
├── main.py                     # File chạy chính
├── requirements.txt            # Thư viện cần cài
└── .gitignore
```

## 🚀 Cài đặt & Chạy

### 1. Clone project

```bash
git clone https://github.com/<your-repo>/medical-datamining-diagnosis-support.git
cd medical-datamining-diagnosis-support
```

### 2. Tạo môi trường ảo

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Chạy toàn bộ pipeline

```bash
python main.py
```

Pipeline sẽ tự động thực hiện theo thứ tự:
1. **Tiền xử lý** dữ liệu gốc → dữ liệu sạch
2. **Huấn luyện** 5 mô hình ML và chọn mô hình tốt nhất
3. **Đánh giá** mô hình (Confusion Matrix, ROC Curve)
4. **Khai phá luật** kết hợp (Apriori + FP-Growth)

### 5. Chạy từng bước riêng lẻ

```bash
python main.py --preprocess     # Chỉ tiền xử lý
python main.py --train          # Chỉ huấn luyện
python main.py --evaluate       # Chỉ đánh giá
python main.py --rules          # Chỉ khai phá luật
```

### 6. Chạy ứng dụng demo

```bash
streamlit run src/app/app.py
```

Hoặc:

```bash
python main.py --app
```

## 📊 Luồng xử lý

```
data/raw → src/preprocessing → data/processed
                                     ↓
                              src/models (train)
                                     ↓
                           src/evaluation (metrics)
                                     ↓
                        outputs/metrics + figures
                                     ↓
                       src/association_rules (Apriori/FP-Growth)
                                     ↓
                              outputs/rules
                                     ↓
                              src/app (demo)
```

## 📈 Kết quả

- **Mô hình tốt nhất:** Logistic Regression (Recall: 82.14%, Accuracy: 91.67%)
- **5 mô hình đã thử:** Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes
- **Luật kết hợp:** Apriori + FP-Growth với Lift ≥ 1.2

## ⚠️ Lưu ý

Hệ thống chỉ mang tính **tham khảo**, không thay thế chẩn đoán y tế chuyên nghiệp.