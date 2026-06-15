"""
Quản lý ngôn ngữ cho ứng dụng: Tiếng Việt và Tiếng Anh.
"""
from __future__ import annotations

LANG = {
    "vi": {
        # Sidebar & Header
        "app_title": "HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN BỆNH TIM",
        "app_subtitle": "Ứng dụng Khai Phá Dữ Liệu trong Y Khoa",
        "sidebar_title": "Medical AI",
        "sidebar_caption": "Đồ án Khai Phá Dữ Liệu",
        "nav_label": "Chức năng",
        "nav_diagnosis": "Chẩn đoán",
        "nav_history": "Lịch sử chẩn đoán",
        "nav_eda": "Khám phá Dữ liệu (EDA)",
        "nav_rules": "Luật kết hợp",
        "nav_compare": "So sánh mô hình",
        "nav_clustering": "Gom cụm",
        "nav_analytics": "Thống kê & Hồi quy",
        "nav_guide": "Hướng dẫn",
        "nav_about": "Thông tin",
        "footer": "Đồ án Khai Phá Dữ Liệu — Ứng dụng hỗ trợ chẩn đoán y khoa",
        "language_label": "Ngôn ngữ",

        # Diagnosis page
        "diag_model_using": "Mô hình đang sử dụng",
        "diag_patient_title": "Thông tin bệnh nhân",
        "diag_patient_hint": "Nhập đầy đủ 13 thuộc tính lâm sàng. Di chuột vào dấu (?) bên cạnh mỗi trường để xem giải thích chi tiết.",
        "diag_result_title": "Kết quả chẩn đoán",
        "diag_no_model": "Chưa có mô hình nào được huấn luyện. Vui lòng chạy `python main.py` trước.",
        "diag_analyzing": "Đang phân tích dữ liệu...",
        "diag_error": "Lỗi khi chẩn đoán",
        "diag_placeholder": "Nhập thông tin bệnh nhân ở cột bên trái, sau đó nhấn nút **Thực hiện chẩn đoán** để xem kết quả.",
        "diag_submit": "Thực hiện chẩn đoán",
        "diag_download_csv": "Tải kết quả (.csv)",

        # Result display
        "result_label_positive": "Có nguy cơ / mắc bệnh tim",
        "result_label_negative": "Không mắc bệnh tim",
        "result_header": "KẾT QUẢ",
        "result_col_item": "Thông tin",
        "result_col_value": "Giá trị",
        "result_model": "Mô hình sử dụng",
        "result_risk": "Mức nguy cơ",
        "result_risk_high": "CAO",
        "result_risk_low": "THẤP",
        "result_prob": "Xác suất mắc bệnh",
        "result_prob_bar": "Xác suất mắc bệnh tim",
        "result_chart_title": "Phân phối xác suất chẩn đoán",
        "result_chart_y": "Xác suất (%)",
        "result_chart_no": "Không bệnh (0)",
        "result_chart_yes": "Có bệnh (1)",
        "result_explain_title": "Giải thích kết quả",
        "result_suggest_title": "Gợi ý tham khảo",
        "result_explain_text": (
            "Mô hình nhận 13 thuộc tính lâm sàng làm đầu vào, sau đó tính toán xác suất "
            "bệnh nhân thuộc hai nhóm: **không mắc bệnh tim (0)** và **có nguy cơ mắc bệnh tim (1)**.\n\n"
            "Ngưỡng quyết định mặc định là 50%%. Nếu xác suất lớp 1 >= 50%%, hệ thống phân loại "
            "bệnh nhân vào nhóm \"Có nguy cơ\".\n\n"
            "> Kết quả chỉ mang tính **tham khảo**, không thay thế chẩn đoán của bác sĩ chuyên khoa."
        ),
        "result_warn_positive": (
            "**Khuyến nghị khi có nguy cơ bệnh tim** (chỉ mang tính tham khảo):\n\n"
            "1. **Thăm khám chuyên khoa:** Đến cơ sở y tế để được bác sĩ tim mạch đánh giá chuyên sâu.\n"
            "2. **Kiểm soát huyết áp:** Tuân thủ phác đồ điều trị và theo dõi huyết áp thường xuyên.\n"
            "3. **Kiểm soát cholesterol:** Giảm chất béo bão hòa, cân nhắc dùng thuốc statin nếu bác sĩ chỉ định.\n"
            "4. **Chế độ dinh dưỡng:** Hạn chế muối (< 5g/ngày), tăng rau xanh, omega-3.\n"
            "5. **Vận động phù hợp:** 150 phút/tuần cường độ vừa phải, tránh gắng sức đột ngột.\n"
            "6. **Ngừng hút thuốc** và hạn chế rượu bia.\n"
            "7. **Theo dõi định kỳ:** Kiểm tra mỗi 3–6 tháng."
        ),
        "result_warn_negative": (
            "**Duy trì lối sống lành mạnh:**\n\n"
            "1. **Tập thể dục đều đặn:** Ít nhất 30 phút/ngày.\n"
            "2. **Chế độ ăn cân đối:** Nhiều rau xanh, trái cây, protein nạc.\n"
            "3. **Ngủ đủ giấc:** 7–8 tiếng mỗi đêm.\n"
            "4. **Khám sức khỏe định kỳ:** Mỗi 6–12 tháng.\n"
            "5. **Quản lý stress:** Thiền, yoga hoặc hoạt động giải trí phù hợp."
        ),

        # Form fields
        "form_demo": "Thông tin nhân khẩu học",
        "form_symptom": "Triệu chứng lâm sàng",
        "form_test": "Xét nghiệm và đo lường",
        "form_extra": "Xét nghiệm bổ sung",
        "f_age": "Tuổi (age)",
        "f_sex": "Giới tính (sex)",
        "f_cp": "Loại đau ngực (cp)",
        "f_trestbps": "Huyết áp lúc nghỉ – mm Hg (trestbps)",
        "f_chol": "Cholesterol huyết thanh – mg/dl (chol)",
        "f_fbs": "Đường huyết lúc đói > 120 mg/dl (fbs)",
        "f_restecg": "Điện tâm đồ lúc nghỉ (restecg)",
        "f_thalach": "Nhịp tim tối đa đạt được (thalach)",
        "f_exang": "Đau thắt ngực khi gắng sức (exang)",
        "f_oldpeak": "Độ chênh ST khi gắng sức (oldpeak)",
        "f_slope": "Độ dốc đoạn ST (slope)",
        "f_ca": "Số mạch máu lớn nhuộm màu (ca)",
        "f_thal": "Thalassemia (thal)",
        "sex_male": "Nam",
        "sex_female": "Nữ",
        "yes": "Có",
        "no": "Không",
        "fbs_yes": "Có (> 120 mg/dl)",
        "fbs_no": "Không (<= 120 mg/dl)",
        "cp_0": "0 – Đau thắt ngực điển hình (typical angina)",
        "cp_1": "1 – Đau thắt ngực không điển hình (atypical angina)",
        "cp_2": "2 – Đau không do tim (non-anginal pain)",
        "cp_3": "3 – Không có triệu chứng (asymptomatic)",
        "restecg_0": "0 – Bình thường",
        "restecg_1": "1 – Bất thường sóng ST-T",
        "restecg_2": "2 – Phì đại thất trái (theo tiêu chuẩn Estes)",
        "slope_0": "0 – Dốc lên (Upsloping)",
        "slope_1": "1 – Bằng phẳng (Flat)",
        "slope_2": "2 – Dốc xuống (Downsloping)",
        "thal_0": "0 – Bình thường",
        "thal_1": "1 – Khuyết tật cố định (fixed defect)",
        "thal_2": "2 – Khuyết tật hồi phục (reversible defect)",

        # Rules page
        "rules_title": "Luật kết hợp đã khai phá",
        "rules_explain_title": "Luật kết hợp là gì?",
        "rules_no_file": "Chưa có file luật kết hợp. Chạy: python main.py --rules",
        "rules_select": "Chọn file luật:",
        "rules_total": "Tổng số luật trong file",
        "rules_min_conf": "Confidence tối thiểu",
        "rules_min_lift": "Lift tối thiểu",
        "rules_max_rows": "Số luật hiển thị",
        "rules_after_filter": "Sau lọc",
        "rules_download": "Tải luật đã lọc (.csv)",
        "rules_explain_text": (
            "**Luật kết hợp (Association Rules)** là kỹ thuật khai phá dữ liệu tìm ra các mối "
            "quan hệ \"nếu ... thì ...\" giữa các thuộc tính.\n\n"
            "| Chỉ số | Ý nghĩa |\n|--------|--------|\n"
            "| **Support** | Tỷ lệ xuất hiện của tập thuộc tính trong toàn bộ dữ liệu. |\n"
            "| **Confidence** | Xác suất xảy ra vế phải khi vế trái đã xảy ra. |\n"
            "| **Lift** | Độ đo sức mạnh liên kết. Lift > 1: liên kết dương. |\n\n"
            "**Thuật toán:** Apriori (duyệt tập ứng viên) và FP-Growth (xây cây FP-Tree, nhanh hơn)."
        ),

        # Model comparison page
        "compare_title": "So sánh mô hình đã huấn luyện",
        "compare_explain_title": "Giải thích các chỉ số đánh giá",
        "compare_no_data": "Chưa có kết quả. Chạy: python main.py --train",
        "compare_chart_title": "So sánh hiệu suất các mô hình",
        "compare_best": "Mô hình tốt nhất",
        "compare_figures": "Biểu đồ đánh giá",
        "compare_download_csv": "Tải bảng so sánh (.csv)",
        "compare_download_chart": "Tải biểu đồ (.png)",
        "compare_explain_text": (
            "| Chỉ số | Ý nghĩa |\n|--------|--------|\n"
            "| **Accuracy** | Tỷ lệ dự đoán đúng trên toàn bộ dữ liệu. |\n"
            "| **Precision** | Trong số ca được dự đoán \"có bệnh\", bao nhiêu ca thực sự có bệnh? |\n"
            "| **Recall** | Trong số ca thực sự có bệnh, mô hình phát hiện được bao nhiêu? |\n"
            "| **F1-Score** | Trung bình điều hòa của Precision và Recall. |\n\n"
            "**Trong y khoa, Recall thường được ưu tiên** vì việc bỏ sót bệnh nhân nguy hiểm hơn báo động giả."
        ),

        # Guide page
        "guide_title": "Hướng dẫn sử dụng hệ thống",
        "guide_content": (
            "---\n**1. Chẩn đoán bệnh tim**\n\n"
            "- Chọn trang **Chẩn đoán** ở thanh điều hướng bên trái.\n"
            "- Nhập đầy đủ 13 thuộc tính lâm sàng của bệnh nhân vào form.\n"
            "- Mỗi trường nhập đều có dấu **(?)** – di chuột vào đó để đọc giải thích chi tiết.\n"
            "- Nhấn nút **Thực hiện chẩn đoán** để hệ thống dự đoán.\n"
            "- Kết quả hiển thị gồm: nhãn dự đoán, xác suất, biểu đồ phân phối và gợi ý tham khảo.\n\n"
            "---\n**2. Xem luật kết hợp**\n\n"
            "- Chọn trang **Luật kết hợp** để xem các luật đã khai phá bằng Apriori và FP-Growth.\n"
            "- Sử dụng bộ lọc Confidence và Lift để thu hẹp kết quả.\n\n"
            "---\n**3. So sánh mô hình**\n\n"
            "- Chọn trang **So sánh mô hình** để xem bảng và biểu đồ so sánh 5 thuật toán.\n\n"
            "---\n**4. Tải dữ liệu**\n\n"
            "- Mỗi trang đều có nút **Tải file** để xuất kết quả dạng CSV hoặc hình ảnh PNG.\n"
        ),

        # About page
        "about_title": "Giới thiệu hệ thống",
        "about_content": (
            "**Đồ án: Ứng dụng hỗ trợ chẩn đoán và điều trị y khoa**\n\n"
            "**Môn học:** Khai Phá Dữ Liệu (Data Mining)\n\n---\n"
            "**Mục tiêu dự án:**\n"
            "- Sử dụng kỹ thuật khai phá dữ liệu để hỗ trợ chẩn đoán bệnh tim.\n"
            "- Áp dụng và so sánh 5 thuật toán Machine Learning.\n"
            "- Khai phá luật kết hợp bằng Apriori và FP-Growth.\n"
            "- Xây dựng ứng dụng demo trực quan bằng Streamlit.\n\n---\n"
            "**Dữ liệu:** Heart Disease UCI Dataset (Kaggle) – 297 mẫu, 13 thuộc tính + 1 nhãn.\n\n---\n"
            "**Thành viên nhóm:**\n\n"
            "| Thành viên | Vai trò |\n|-----------|--------|\n"
            "| Thành Danh | Mô hình chẩn đoán, tích hợp pipeline |\n"
            "| Hồng Vỹ | Thu thập dữ liệu, EDA |\n"
            "| Quốc An | Tiền xử lý dữ liệu |\n"
            "| Minh Thiện | Đánh giá mô hình |\n"
            "| Quang Ngọc | Khai phá luật kết hợp |\n"
            "| Lê Hậu | Giao diện demo, slide |\n\n"
            "> **Lưu ý:** Hệ thống chỉ mang tính **tham khảo**, không thay thế chẩn đoán của bác sĩ."
        ),

        # Attribute table
        "attr_table_header": "Bảng mô tả 13 thuộc tính đầu vào",
        "attr_col_name": "Thuộc tính",
        "attr_col_type": "Kiểu",
        "attr_col_range": "Khoảng",
        "attr_col_desc": "Mô tả",
    },
    "en": {
        "app_title": "HEART DISEASE DIAGNOSIS SUPPORT SYSTEM",
        "app_subtitle": "Data Mining Application in Medical Science",
        "sidebar_title": "Medical AI",
        "sidebar_caption": "Data Mining Project",
        "nav_label": "Navigation",
        "nav_diagnosis": "Diagnosis",
        "nav_history": "Diagnosis History",
        "nav_eda": "Data Exploration (EDA)",
        "nav_rules": "Association Rules",
        "nav_compare": "Model Comparison",
        "nav_clustering": "Patient Clustering",
        "nav_analytics": "Statistics & Regression",
        "nav_guide": "User Guide",
        "nav_about": "About",
        "footer": "Data Mining Project — Medical Diagnosis Support Application",
        "language_label": "Language",

        "diag_model_using": "Model in use",
        "diag_patient_title": "Patient Information",
        "diag_patient_hint": "Enter all 13 clinical attributes. Hover over the (?) icon next to each field for detailed explanation.",
        "diag_result_title": "Diagnosis Result",
        "diag_no_model": "No trained model found. Please run `python main.py` first.",
        "diag_analyzing": "Analyzing data...",
        "diag_error": "Diagnosis error",
        "diag_placeholder": "Enter patient information on the left, then click **Run Diagnosis** to see the result.",
        "diag_submit": "Run Diagnosis",
        "diag_download_csv": "Download result (.csv)",

        "result_label_positive": "At risk / Heart disease detected",
        "result_label_negative": "No heart disease detected",
        "result_header": "RESULT",
        "result_col_item": "Item",
        "result_col_value": "Value",
        "result_model": "Model used",
        "result_risk": "Risk level",
        "result_risk_high": "HIGH",
        "result_risk_low": "LOW",
        "result_prob": "Disease probability",
        "result_prob_bar": "Heart disease probability",
        "result_chart_title": "Diagnosis Probability Distribution",
        "result_chart_y": "Probability (%)",
        "result_chart_no": "No disease (0)",
        "result_chart_yes": "Disease (1)",
        "result_explain_title": "Result Explanation",
        "result_suggest_title": "Recommendations",
        "result_explain_text": (
            "The model takes 13 clinical attributes as input and calculates the probability "
            "of the patient belonging to two groups: **no heart disease (0)** and **at risk of heart disease (1)**.\n\n"
            "The default decision threshold is 50%%. If the probability of class 1 >= 50%%, the system classifies "
            "the patient as \"At risk\".\n\n"
            "> This result is for **reference only** and does not replace professional medical diagnosis."
        ),
        "result_warn_positive": (
            "**Recommendations for high-risk patients** (for reference only):\n\n"
            "1. **See a cardiologist** for a thorough evaluation.\n"
            "2. **Control blood pressure** with prescribed medication.\n"
            "3. **Manage cholesterol** through diet and medication if prescribed.\n"
            "4. **Healthy diet:** Low salt (< 5g/day), more vegetables, omega-3.\n"
            "5. **Moderate exercise:** 150 min/week, avoid sudden exertion.\n"
            "6. **Quit smoking** and limit alcohol.\n"
            "7. **Regular check-ups** every 3–6 months."
        ),
        "result_warn_negative": (
            "**Maintain a healthy lifestyle:**\n\n"
            "1. **Regular exercise:** At least 30 min/day.\n"
            "2. **Balanced diet:** Plenty of vegetables, fruits, lean protein.\n"
            "3. **Adequate sleep:** 7–8 hours per night.\n"
            "4. **Regular check-ups:** Every 6–12 months.\n"
            "5. **Stress management:** Meditation, yoga, or suitable activities."
        ),

        "form_demo": "Demographics",
        "form_symptom": "Clinical Symptoms",
        "form_test": "Tests & Measurements",
        "form_extra": "Additional Tests",
        "f_age": "Age",
        "f_sex": "Sex",
        "f_cp": "Chest pain type (cp)",
        "f_trestbps": "Resting blood pressure – mm Hg (trestbps)",
        "f_chol": "Serum cholesterol – mg/dl (chol)",
        "f_fbs": "Fasting blood sugar > 120 mg/dl (fbs)",
        "f_restecg": "Resting ECG (restecg)",
        "f_thalach": "Maximum heart rate achieved (thalach)",
        "f_exang": "Exercise-induced angina (exang)",
        "f_oldpeak": "ST depression by exercise (oldpeak)",
        "f_slope": "Slope of peak exercise ST (slope)",
        "f_ca": "Number of major vessels by fluoroscopy (ca)",
        "f_thal": "Thalassemia (thal)",
        "sex_male": "Male",
        "sex_female": "Female",
        "yes": "Yes",
        "no": "No",
        "fbs_yes": "Yes (> 120 mg/dl)",
        "fbs_no": "No (<= 120 mg/dl)",
        "cp_0": "0 – Typical angina",
        "cp_1": "1 – Atypical angina",
        "cp_2": "2 – Non-anginal pain",
        "cp_3": "3 – Asymptomatic",
        "restecg_0": "0 – Normal",
        "restecg_1": "1 – ST-T wave abnormality",
        "restecg_2": "2 – Left ventricular hypertrophy (Estes criteria)",
        "slope_0": "0 – Upsloping",
        "slope_1": "1 – Flat",
        "slope_2": "2 – Downsloping",
        "thal_0": "0 – Normal",
        "thal_1": "1 – Fixed defect",
        "thal_2": "2 – Reversible defect",

        "rules_title": "Mined Association Rules",
        "rules_explain_title": "What are Association Rules?",
        "rules_no_file": "No rules file found. Run: python main.py --rules",
        "rules_select": "Select rules file:",
        "rules_total": "Total rules in file",
        "rules_min_conf": "Minimum Confidence",
        "rules_min_lift": "Minimum Lift",
        "rules_max_rows": "Rules to display",
        "rules_after_filter": "After filtering",
        "rules_download": "Download filtered rules (.csv)",
        "rules_explain_text": (
            "**Association Rules** is a data mining technique that discovers \"if ... then ...\" relationships.\n\n"
            "| Metric | Meaning |\n|--------|--------|\n"
            "| **Support** | Frequency of the itemset in the entire dataset. |\n"
            "| **Confidence** | Probability of consequent given antecedent. |\n"
            "| **Lift** | Strength of association. Lift > 1: positive association. |\n\n"
            "**Algorithms:** Apriori (candidate generation) and FP-Growth (FP-Tree, faster)."
        ),

        "compare_title": "Trained Model Comparison",
        "compare_explain_title": "Evaluation Metrics Explained",
        "compare_no_data": "No results found. Run: python main.py --train",
        "compare_chart_title": "Model Performance Comparison",
        "compare_best": "Best model",
        "compare_figures": "Evaluation Charts",
        "compare_download_csv": "Download comparison (.csv)",
        "compare_download_chart": "Download chart (.png)",
        "compare_explain_text": (
            "| Metric | Meaning |\n|--------|--------|\n"
            "| **Accuracy** | Overall correct prediction rate. |\n"
            "| **Precision** | Of predicted positives, how many are truly positive? |\n"
            "| **Recall** | Of actual positives, how many were detected? |\n"
            "| **F1-Score** | Harmonic mean of Precision and Recall. |\n\n"
            "**In medicine, Recall is prioritized** because missing a patient is more dangerous than a false alarm."
        ),

        "guide_title": "User Guide",
        "guide_content": (
            "---\n**1. Heart Disease Diagnosis**\n\n"
            "- Select the **Diagnosis** page from the left sidebar.\n"
            "- Fill in all 13 clinical attributes in the form.\n"
            "- Each field has a **(?)** icon – hover for detailed medical explanation.\n"
            "- Click **Run Diagnosis** for the system to predict.\n"
            "- Results include: prediction label, probability, distribution chart and recommendations.\n\n"
            "---\n**2. Association Rules**\n\n"
            "- Select **Association Rules** to view rules mined by Apriori and FP-Growth.\n"
            "- Use Confidence and Lift filters to refine results.\n\n"
            "---\n**3. Model Comparison**\n\n"
            "- Select **Model Comparison** to see table and chart comparing 5 algorithms.\n\n"
            "---\n**4. Download Data**\n\n"
            "- Every page has a **Download** button to export results as CSV or PNG.\n"
        ),

        "about_title": "About the System",
        "about_content": (
            "**Project: Medical Diagnosis and Treatment Support Application**\n\n"
            "**Course:** Data Mining\n\n---\n"
            "**Objectives:**\n"
            "- Use data mining techniques to support heart disease diagnosis.\n"
            "- Apply and compare 5 Machine Learning algorithms.\n"
            "- Mine association rules using Apriori and FP-Growth.\n"
            "- Build a visual demo application with Streamlit.\n\n---\n"
            "**Data:** Heart Disease UCI Dataset (Kaggle) – 297 samples, 13 attributes + 1 label.\n\n---\n"
            "**Team Members:**\n\n"
            "| Member | Role |\n|--------|------|\n"
            "| Thanh Danh | Diagnosis models, pipeline integration |\n"
            "| Hong Vy | Data collection, EDA |\n"
            "| Quoc An | Data preprocessing |\n"
            "| Minh Thien | Model evaluation |\n"
            "| Quang Ngoc | Association rules mining |\n"
            "| Le Hau | Demo UI, presentation |\n\n"
            "> **Note:** This system is for **reference only** and does not replace professional medical diagnosis."
        ),

        "attr_table_header": "Description of 13 Input Attributes",
        "attr_col_name": "Attribute",
        "attr_col_type": "Type",
        "attr_col_range": "Range",
        "attr_col_desc": "Description",
    },
}


def t(key: str, lang: str = "vi") -> str:
    """Trả về chuỗi dịch theo key và ngôn ngữ."""
    return LANG.get(lang, LANG["vi"]).get(key, key)
