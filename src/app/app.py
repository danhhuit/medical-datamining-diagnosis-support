"""
Ứng dụng demo Streamlit – Hỗ trợ chẩn đoán bệnh tim.
Chạy: streamlit run src/app/app.py
"""
from __future__ import annotations

import io
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.app.translations import t
from src.app.ui_components import input_form
from src.app.display_result import show_result

# Đảm bảo Streamlit nạp lại mã nguồn predict.py mới nhất khi chạy
import importlib
import src.models.predict
importlib.reload(src.models.predict)
from src.models.predict import predict_one

from src.models.save_model import list_saved_models

# ── CONFIG ──
st.set_page_config(page_title="Medical AI", page_icon="H", layout="wide")

st.markdown("""
<style>
/* ── Header ── */
.header{background:linear-gradient(135deg,#1a3a5c 0%,#2d6a9f 100%);padding:18px 24px;
border-radius:8px;color:#fff;text-align:center;font-size:22px;font-weight:600;
letter-spacing:.5px;margin-bottom:16px}
.header small{display:block;font-size:13px;font-weight:400;margin-top:4px;opacity:.85}

/* ── Footer ── */
.footer{text-align:center;font-size:13px;padding:8px;margin-top:16px;opacity:.7}

/* ── Sidebar: Use Streamlit's CSS variables so colors auto-adapt to theme ── */
[data-testid="stSidebar"]{
    background-color: var(--secondary-background-color) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label{
    color: var(--text-color) !important;
}
</style>""", unsafe_allow_html=True)


def get_lang() -> str:
    return st.session_state.get("lang", "vi")


def get_model() -> str | None:
    models = list_saved_models()
    if not models:
        return None
    for m in models:
        if "logistic_regression" in m:
            return m
    return models[0]


# ════════════════════════════════════════════
# TRANG 1: CHẨN ĐOÁN
# ════════════════════════════════════════════
def page_diagnosis():
    lang = get_lang()
    model_file = get_model()

    if model_file is None:
        st.error(t("diag_no_model", lang))
        return

    st.caption(f"{t('diag_model_using', lang)}: {model_file}")

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown(f"#### {t('diag_patient_title', lang)}")
        st.caption(t("diag_patient_hint", lang))
        
        # Tải hồ sơ bệnh nhân mẫu (Sample Patient Profiles)
        st.markdown("##### " + ("Tải hồ sơ bệnh nhân mẫu:" if lang == "vi" else "Load Sample Patient Profile:"))
        sample_options = {
            "Mặc định (Default)": None,
            "Ca 1: Không mắc bệnh tim (Healthy - Xác suất ~2.4%)": {
                "age": 34, "sex": 1, "cp": 0, "trestbps": 118, "chol": 182, "fbs": 0,
                "restecg": 2, "thalach": 174, "exang": 0, "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 0
            },
            "Ca 2: Ranh giới / Nghi ngờ (Borderline - Xác suất ~49.4%)": {
                "age": 38, "sex": 1, "cp": 0, "trestbps": 120, "chol": 231, "fbs": 0,
                "restecg": 0, "thalach": 182, "exang": 1, "oldpeak": 3.8, "slope": 1, "ca": 0, "thal": 2
            },
            "Ca 3: Mắc bệnh tim / Nguy cơ cao (Diseased - Xác suất ~97.5%)": {
                "age": 70, "sex": 1, "cp": 2, "trestbps": 160, "chol": 269, "fbs": 0,
                "restecg": 0, "thalach": 112, "exang": 1, "oldpeak": 2.9, "slope": 1, "ca": 1, "thal": 2
            }
        }
        
        selected_sample_name = st.selectbox(
            "Chọn một ca lâm sàng mẫu để kiểm thử nhanh:" if lang == "vi" else "Select a sample clinical case for testing:",
            list(sample_options.keys()),
            key="selected_sample_case"
        )
        
        # Cập nhật session state của các widget form khi thay đổi selectbox mẫu
        if st.session_state.get("prev_sample_case") != selected_sample_name:
            selected_sample = sample_options[selected_sample_name]
            if selected_sample:
                for k, v in selected_sample.items():
                    st.session_state[f"form_{k}"] = v
            else:
                defaults = {
                    "age": 50, "sex": 1, "cp": 0, "trestbps": 130, "chol": 240,
                    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
                    "oldpeak": 1.0, "slope": 0, "ca": 0, "thal": 0
                }
                for k, v in defaults.items():
                    st.session_state[f"form_{k}"] = v
            st.session_state["prev_sample_case"] = selected_sample_name
            
        submitted, user_data = input_form(lang)
        
        if submitted:
            with st.spinner(t("diag_analyzing", lang)):
                try:
                    result = predict_one(user_data, model_file)
                    st.session_state["current_result"] = {
                        "result": result,
                        "inputs": user_data
                    }
                    
                    # Lưu vào lịch sử (History)
                    from datetime import datetime
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    history_entry = {
                        "timestamp": now_str,
                        "inputs": user_data.copy(),
                        "result": result
                    }
                    st.session_state["history"].insert(0, history_entry)
                except Exception as e:
                    st.error(f"{t('diag_error', lang)}: {e}")

    with col2:
        st.markdown(f"#### {t('diag_result_title', lang)}")
        if st.session_state.get("current_result") is not None:
            show_result(
                st.session_state["current_result"]["result"],
                lang,
                inputs=st.session_state["current_result"]["inputs"]
            )
            
            st.markdown("---")
            if st.button(
                "Làm mới & Chẩn đoán ca tiếp theo" if lang == "vi" else "Refresh & Next Diagnosis",
                use_container_width=True,
                type="secondary"
            ):
                st.session_state["current_result"] = None
                # Reset all form fields in session state
                defaults = {
                    "age": 50, "sex": 1, "cp": 0, "trestbps": 130, "chol": 240,
                    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
                    "oldpeak": 1.0, "slope": 0, "ca": 0, "thal": 0
                }
                for k, v in defaults.items():
                    st.session_state[f"form_{k}"] = v
                # Reset selectbox select state
                if "selected_sample_case" in st.session_state:
                    st.session_state["selected_sample_case"] = "Mặc định (Default)"
                st.session_state["prev_sample_case"] = "Mặc định (Default)"
                st.rerun()
        else:
            st.info(t("diag_placeholder", lang))





# ════════════════════════════════════════════
# TRANG 1.6: LỊCH SỬ CHẨN ĐOÁN (HISTORY)
# ════════════════════════════════════════════
def page_history():
    lang = get_lang()
    st.markdown(f"#### {t('nav_history', lang)}")
    
    # Nhập lịch sử từ file CSV
    uploaded_file = st.file_uploader(
        "Nhập lịch sử từ file CSV đã lưu" if lang == "vi" else "Import history from saved CSV file",
        type=["csv"],
        key="history_uploader"
    )
    if uploaded_file is not None:
        if st.session_state.get("loaded_file_name") != uploaded_file.name:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
                required_cols = [
                    "timestamp", "age", "sex", "cp", "trestbps", "chol", "fbs", 
                    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
                    "prediction", "probability_class_0", "probability_class_1", "model_name"
                ]
                if all(c in df.columns for c in required_cols):
                    imported_history = []
                    for _, row in df.iterrows():
                        inputs = {
                            "age": int(row["age"]),
                            "sex": int(row["sex"]),
                            "cp": int(row["cp"]),
                            "trestbps": int(row["trestbps"]),
                            "chol": int(row["chol"]),
                            "fbs": int(row["fbs"]),
                            "restecg": int(row["restecg"]),
                            "thalach": int(row["thalach"]),
                            "exang": int(row["exang"]),
                            "oldpeak": float(row["oldpeak"]),
                            "slope": int(row["slope"]),
                            "ca": int(row["ca"]),
                            "thal": int(row["thal"])
                        }
                        result = {
                            "prediction": int(row["prediction"]),
                            "positive_class_probability": float(row["probability_class_1"]),
                            "probabilities": [float(row["probability_class_0"]), float(row["probability_class_1"])],
                            "model_name": str(row["model_name"])
                        }
                        imported_history.append({
                            "timestamp": str(row["timestamp"]),
                            "inputs": inputs,
                            "result": result
                        })
                    st.session_state["history"] = imported_history
                    st.session_state["loaded_file_name"] = uploaded_file.name
                    st.rerun()
                else:
                    st.error("File CSV không đúng định dạng lịch sử chẩn đoán." if lang == "vi" else "Invalid CSV history file format.")
            except Exception as e:
                st.error(f"Lỗi khi đọc file CSV: {e}" if lang == "vi" else f"Error reading CSV file: {e}")
    else:
        st.session_state["loaded_file_name"] = None

    if "history" not in st.session_state or not st.session_state["history"]:
        st.info(
            "Chưa có lượt chẩn đoán nào được thực hiện trong phiên làm việc này. Bạn có thể kéo thả file CSV đã lưu trước đó vào ô phía trên để xem lại."
            if lang == "vi" else
            "No diagnosis history has been recorded in this session yet. You can drag and drop a previously saved CSV file above to review."
        )
        return

    history = st.session_state["history"]
    
    # Chuẩn bị dữ liệu để xuất CSV toàn bộ lịch sử
    export_rows = []
    for entry in history:
        inputs = entry["inputs"]
        result = entry["result"]
        row = {
            "timestamp": entry["timestamp"],
            "age": inputs["age"],
            "sex": inputs["sex"],
            "cp": inputs["cp"],
            "trestbps": inputs["trestbps"],
            "chol": inputs["chol"],
            "fbs": inputs["fbs"],
            "restecg": inputs["restecg"],
            "thalach": inputs["thalach"],
            "exang": inputs["exang"],
            "oldpeak": inputs["oldpeak"],
            "slope": inputs["slope"],
            "ca": inputs["ca"],
            "thal": inputs["thal"],
            "prediction": result["prediction"],
            "probability_class_0": result["probabilities"][0] if "probabilities" in result and result["probabilities"] else 1.0 - result["positive_class_probability"],
            "probability_class_1": result["positive_class_probability"],
            "model_name": result["model_name"]
        }
        export_rows.append(row)

    df_export = pd.DataFrame(export_rows)
    csv_data = df_export.to_csv(index=False).encode("utf-8-sig")
    
    from datetime import datetime
    export_filename = datetime.now().strftime("Lich_su_Chan_doan_%Y%m%d_%H%M%S.csv")

    col_summary, col_download, col_clear = st.columns([1.5, 1.5, 1])
    with col_summary:
        st.write(
            f"**Tổng số ca:** `{len(history)}` ca chẩn đoán."
            if lang == "vi" else
            f"**Total cases:** `{len(history)}` diagnoses."
        )
    with col_download:
        st.download_button(
            label="Tải toàn bộ lịch sử (.csv)" if lang == "vi" else "Download All History (.csv)",
            data=csv_data,
            file_name=export_filename,
            mime="text/csv",
            use_container_width=True
        )
    with col_clear:
        if st.button(
            "Xóa lịch sử" if lang == "vi" else "Clear History", 
            use_container_width=True,
            type="primary"
        ):
            st.session_state["history"] = []
            st.session_state["current_result"] = None
            st.rerun()

    st.markdown("---")

    for idx, entry in enumerate(history):
        timestamp = entry["timestamp"]
        inputs = entry["inputs"]
        result = entry["result"]
        
        prediction = result.get("prediction", 0)
        prob = result.get("positive_class_probability", 0.5)
        model_name = result.get("model_name", "Unknown").replace("_", " ").title()
        
        label = t("result_label_positive", lang) if prediction == 1 else t("result_label_negative", lang)
        
        expander_title = (
            f"Ca #{len(history) - idx} - {timestamp} | Kết quả: {label} (Xác suất: {prob * 100:.1f}%)"
            if lang == "vi" else
            f"Case #{len(history) - idx} - {timestamp} | Result: {label} (Prob: {prob * 100:.1f}%)"
        )
        
        with st.expander(expander_title, expanded=False):
            col1, col2 = st.columns([1.2, 1], gap="medium")
            
            with col1:
                st.markdown("**Thông tin lâm sàng đầu vào**" if lang == "vi" else "**Clinical Input Features**")
                
                # Tạo bảng đẹp cho inputs
                input_rows = []
                for k, v in inputs.items():
                    # Format các trường cho dễ đọc
                    desc = t(f"f_{k}", lang)
                    if k == "sex":
                        val_str = t("sex_male", lang) if v == 1 else t("sex_female", lang)
                    elif k == "cp":
                        val_str = t(f"cp_{v}", lang).split(" – ")[-1] if lang == "vi" else t(f"cp_{v}", lang)
                    elif k == "fbs":
                        val_str = t("yes", lang) if v == 1 else t("no", lang)
                    elif k == "restecg":
                        val_str = t(f"restecg_{v}", lang)
                    elif k == "exang":
                        val_str = t("yes", lang) if v == 1 else t("no", lang)
                    elif k == "slope":
                        val_str = t(f"slope_{v}", lang)
                    elif k == "thal":
                        val_str = t(f"thal_{v}", lang)
                    else:
                        val_str = str(v)
                    input_rows.append([desc, val_str])
                    
                df_inputs = pd.DataFrame(input_rows, columns=["Thuộc tính", "Giá trị"] if lang == "vi" else ["Attribute", "Value"])
                st.dataframe(df_inputs, use_container_width=True, hide_index=True, height=250)
                
            with col2:
                st.markdown("**Kết quả chẩn đoán**" if lang == "vi" else "**Diagnosis Outcome**")
                
                risk = t("result_risk_high", lang) if prediction == 1 else t("result_risk_low", lang)
                
                if prediction == 1:
                    st.error(f"**{t('result_header', lang)}: {label}**")
                else:
                    st.success(f"**{t('result_header', lang)}: {label}**")
                    
                summary_data = [
                    {"Mục": t("result_model", lang), "Chi tiết": model_name},
                    {"Mục": t("result_risk", lang), "Chi tiết": risk},
                    {"Mục": t("result_prob", lang), "Chi tiết": f"{prob * 100:.1f}%"}
                ] if lang == "vi" else [
                    {"Item": t("result_model", lang), "Detail": model_name},
                    {"Item": t("result_risk", lang), "Detail": risk},
                    {"Item": t("result_prob", lang), "Detail": f"{prob * 100:.1f}%"}
                ]
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                
                # Nút tải lại kết quả của ca này
                download_df = pd.DataFrame([{
                    t("result_model", lang): model_name,
                    t("result_risk", lang): risk,
                    t("result_prob", lang): f"{prob * 100:.1f}%",
                    t("result_header", lang): label,
                }])
                csv_bytes = download_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=(t("diag_download_csv", lang) + f" (Ca #{len(history) - idx})"),
                    data=csv_bytes,
                    file_name=f"diagnosis_case_{len(history) - idx}.csv",
                    mime="text/csv",
                    key=f"dl_btn_{idx}"
                )


# ════════════════════════════════════════════
# TRANG 1.5: KHÁM PHÁ DỮ LIỆU (EDA)
# ════════════════════════════════════════════
def page_eda():
    lang = get_lang()
    if lang == "vi":
        st.markdown("### Phân tích Khám phá Dữ liệu (EDA)")
        st.markdown(
            "Khám phá dữ liệu (Exploratory Data Analysis - EDA) là bước đi đầu tiên vô cùng quan trọng "
            "trong mọi bài toán Khai phá dữ liệu. Quá trình này giúp chúng ta hiểu rõ phân bố của các thuộc tính, "
            "phát hiện bất thường (outliers), và tìm kiếm các mối quan hệ tương quan ban đầu giữa các triệu chứng "
            "với nguy cơ mắc bệnh tim."
        )
    else:
        st.markdown("### Exploratory Data Analysis (EDA)")
        st.markdown(
            "Exploratory Data Analysis (EDA) is a crucial first step in any Data Mining project. "
            "It helps us understand attribute distributions, identify anomalies (outliers), "
            "and uncover underlying correlations between clinical symptoms and heart disease risk."
        )

    eda_dir = PROJECT_ROOT / "outputs" / "figures" / "eda"
    if not eda_dir.exists():
        st.warning(
            "Không tìm thấy các biểu đồ EDA tại thư mục outputs/figures/eda/. Vui lòng chạy pipeline tiền xử lý trước."
            if lang == "vi"
            else "EDA figures not found at outputs/figures/eda/. Please run the preprocessing pipeline first."
        )
        return

    tab_titles = (
        ["Tổng quan & Phân phối", "Biến số liên tục", "Biến phân loại", "Ma trận tương quan"]
        if lang == "vi"
        else ["Overview & Distribution", "Continuous Features", "Categorical Features", "Correlations"]
    )
    
    t1, t2, t3, t4 = st.tabs(tab_titles)

    with t1:
        st.markdown("#### " + ("Biểu đồ phân phối thuộc tính mục tiêu & tần suất các biến số" if lang == "vi" else "Target Distribution & Feature Histograms"))
        
        # Target distribution
        p_target = eda_dir / "01_target_distribution.png"
        if p_target.exists():
            col_img, col_desc = st.columns([1.5, 1])
            with col_img:
                st.image(str(p_target), use_container_width=True)
            with col_desc:
                st.markdown("**" + ("Biểu đồ phân phối biến mục tiêu (Target Distribution)" if lang == "vi" else "Target Distribution Chart") + "**")
                if lang == "vi":
                    st.write(
                        "Biểu đồ thể hiện tỷ lệ số ca không mắc bệnh tim (nhãn 0 - màu xanh) "
                        "và có nguy cơ/mắc bệnh tim (nhãn 1 - màu đỏ) trong tập dữ liệu UCI.\n\n"
                        "**Ý nghĩa lâm sàng:** Số mẫu ở hai lớp khá cân bằng (~46% lành mạnh và ~54% mắc bệnh). "
                        "Sự cân bằng này rất lý tưởng để huấn luyện các thuật toán phân lớp, giúp mô hình không bị lệch "
                        "về bất kỳ nhóm nào và đưa ra dự đoán chính xác hơn trên cả hai nhóm."
                    )
                else:
                    st.write(
                        "This chart illustrates the ratio of healthy patients (label 0 - green) "
                        "vs. patients diagnosed with heart disease (label 1 - red) in the UCI dataset.\n\n"
                        "**Analytical Value:** The class distribution is well-balanced (~46% healthy vs ~54% disease). "
                        "This balance is critical for training robust ML models, ensuring they do not learn a bias "
                        "towards a majority group and maintain stable accuracy across both classes."
                    )
        
        st.markdown("---")
        
        # Histograms & Boxplots
        p_hist = eda_dir / "02_numeric_histograms.png"
        p_box = eda_dir / "03_boxplot_outliers.png"
        
        c1, c2 = st.columns(2)
        with c1:
            if p_hist.exists():
                st.markdown("**" + ("Phân phối tần suất các biến số (Histograms)" if lang == "vi" else "Continuous Features Histograms") + "**")
                st.image(str(p_hist), use_container_width=True)
                if lang == "vi":
                    st.caption("Biểu đồ tần suất thể hiện phân bố hình chuông gần chuẩn của các chỉ số Tuổi, Huyết áp, Cholesterol và Nhịp tim tối đa.")
                else:
                    st.caption("Histograms show near-normal bell-curve distributions for Age, Blood Pressure, Cholesterol, and Max Heart Rate.")
        with c2:
            if p_box.exists():
                st.markdown("**" + ("Phát hiện ngoại lệ (Outliers Boxplot)" if lang == "vi" else "Outlier Detection Boxplot") + "**")
                st.image(str(p_box), use_container_width=True)
                if lang == "vi":
                    st.caption("Biểu đồ hộp xác định các điểm ngoại lệ (như huyết áp > 180 mmHg hoặc cholesterol > 360 mg/dl) cần được xử lý co hẹp.")
                else:
                    st.caption("Boxplot identifying extreme outliers (such as trestbps > 180 or chol > 360) targeted for clipping.")

    with t2:
        st.markdown("#### " + ("Phân tích biến số liên tục theo Tình trạng bệnh" if lang == "vi" else "Continuous Features vs Heart Disease"))
        
        num_features = {
            "Age (Tuổi)": ("04_1_boxplot_age_by_condition.png", 
                           "Độ tuổi của bệnh nhân. Nhóm mắc bệnh có trung vị tuổi cao hơn (~58 tuổi).", 
                           "Patient age. Patients diagnosed with heart disease display a higher median age (~58)."),
            "Resting Blood Pressure (Huyết áp lúc nghỉ)": ("04_2_boxplot_trestbps_by_condition.png", 
                                                         "Huyết áp lúc nghỉ (mmHg). Nhóm mắc bệnh có phân bố huyết áp dịch chuyển nhẹ về phía cao hơn.", 
                                                         "Resting blood pressure in mm Hg. The heart disease group shows a slightly elevated pressure spread."),
            "Cholesterol (Cholesterol huyết thanh)": ("04_3_boxplot_chol_by_condition.png", 
                                                    "Nồng độ cholesterol huyết thanh (mg/dl). Trung vị giữa hai nhóm khá tương đồng, cho thấy cholesterol đơn lẻ không đủ làm biến phân tách mạnh.", 
                                                    "Serum cholesterol in mg/dl. Medians are close, showing cholesterol alone offers limited discriminative power without other markers."),
            "Max Heart Rate (Nhịp tim tối đa)": ("04_4_boxplot_thalach_by_condition.png", 
                                               "Nhịp tim gắng sức tối đa đạt được. **Khác biệt rất rõ rệt**: Nhóm lành mạnh có nhịp tim cao hơn nhiều (~160 bpm) so với nhóm mắc bệnh (~140 bpm).", 
                                               "Maximum heart rate achieved. **Significant discrepancy**: Healthy group maintains much higher heart rates (~160 bpm) compared to patients (~140 bpm)."),
            "ST Depression (Độ chênh lệch đoạn ST)": ("04_5_boxplot_oldpeak_by_condition.png", 
                                                    "Độ chênh ST (oldpeak) đo bằng điện tâm đồ khi gắng sức. **Chỉ số cực kỳ quan trọng**: Nhóm mắc bệnh có mức độ chênh lệch cao hơn hẳn (trung vị ~1.5mm so với ~0.2mm ở nhóm bình thường).", 
                                                    "ST depression induced by exercise relative to rest (oldpeak). **Key predictor**: Heart disease patients exhibit significantly higher ST depression (median ~1.5mm vs ~0.2mm in healthy).")
        }
        
        selected_num = st.selectbox(
            "Chọn thuộc tính số liên tục:" if lang == "vi" else "Select a continuous feature:",
            list(num_features.keys())
        )
        
        fname, desc_vi, desc_en = num_features[selected_num]
        img_path = eda_dir / fname
        if img_path.exists():
            col_img, col_desc = st.columns([1.5, 1])
            with col_img:
                st.image(str(img_path), use_container_width=True)
            with col_desc:
                st.markdown(f"**{selected_num}**")
                st.write(desc_vi if lang == "vi" else desc_en)

    with t3:
        st.markdown("#### " + ("Tần suất mắc bệnh theo các nhóm thuộc tính phân loại" if lang == "vi" else "Categorical Features Distributions"))
        
        cat_features = {
            "Sex (Giới tính)": ("05_1_countplot_sex_by_condition.png",
                                "1 = Nam, 0 = Nữ. Tỷ lệ mắc bệnh ở nam giới chiếm ưu thế tuyệt đối trong tập dữ liệu này.",
                                "1 = Male, 0 = Female. Heart disease incidence is notably higher among males in this cohort."),
            "Chest Pain Type (cp - Loại đau ngực)": ("05_2_countplot_cp_by_condition.png",
                                                    "Loại đau ngực (0-3). Đau thắt ngực không triệu chứng (asymptomatic - nhãn 3) lại chiếm tỷ lệ mắc bệnh cao nhất, cảnh báo về triệu chứng âm thầm.",
                                                    "Chest pain type (0-3). Asymptomatic pain (3) corresponds to the highest disease count, highlighting silent risks."),
            "Fasting Blood Sugar (fbs - Đường huyết lúc đói)": ("05_3_countplot_fbs_by_condition.png",
                                                               "Đường huyết lúc đói > 120 mg/dl (1 = Có, 0 = Không). Tỷ lệ phân bố mắc bệnh tương đối đồng đều ở cả hai nhóm.",
                                                               "Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No). The distribution reveals similar ratios of disease in both groups."),
            "Resting ECG (restecg - Điện tâm đồ lúc nghỉ)": ("05_4_countplot_restecg_by_condition.png",
                                                            "Kết quả điện tâm đồ (0-2). Nhóm phì đại thất trái (nhãn 2) có số ca mắc bệnh nhiều hơn.",
                                                            "Resting electrocardiographic results (0-2). Left ventricular hypertrophy (2) is heavily represented in heart disease patients."),
            "Exercise Induced Angina (exang - Đau thắt ngực khi gắng sức)": ("05_5_countplot_exang_by_condition.png",
                                                                            "Đau thắt ngực khi vận động gắng sức (1 = Có, 0 = Không). **Liên hệ mạnh**: Những người bị exang=1 hầu hết đều mắc bệnh.",
                                                                            "Exercise-induced angina (1 = Yes, 0 = No). **Strong link**: Patients experiencing pain during exertion are highly likely to have heart disease."),
            "ST Slope (slope - Độ dốc đoạn ST)": ("05_6_countplot_slope_by_condition.png",
                                                  "Độ dốc của đoạn ST gắng sức cực đại (0-2). Nhãn 1 (Flat) và nhãn 2 (Downsloping) chiếm đa số ở nhóm bệnh lý.",
                                                  "Slope of peak exercise ST segment (0-2). Flat (1) and downsloping (2) slopes are highly associated with abnormal cardiac perfusion."),
            "Major Vessels (ca - Số mạch máu lớn nhuộm màu)": ("05_7_countplot_ca_by_condition.png",
                                                               "Số lượng mạch máu chính (0-3) phát hiện qua chụp huỳnh quang. Càng có nhiều mạch máu bị tắc/nhuộm màu (ca > 0), khả năng bệnh lý càng nghiêm trọng.",
                                                               "Number of major vessels (0-3) colored by fluoroscopy. Higher vessel blockage counts (ca > 0) strongly indicate heart disease."),
            "Thalassemia (thal)": ("05_8_countplot_thal_by_condition.png",
                                   "Chỉ số Thalassemia (0 = Bình thường, 1 = Khuyết tật cố định, 2 = Khuyết tật hồi phục). Khuyết tật hồi phục (nhãn 2) có sự tương quan cực kỳ lớn với bệnh lý tim mạch.",
                                   "Thalassemia indicator (0-2). Reversible defect (2) exhibits an extremely high concentration of diagnosed cases.")
        }
        
        selected_cat = st.selectbox(
            "Chọn thuộc tính định tính:" if lang == "vi" else "Select a categorical feature:",
            list(cat_features.keys())
        )
        
        fname, desc_vi, desc_en = cat_features[selected_cat]
        img_path = eda_dir / fname
        if img_path.exists():
            col_img, col_desc = st.columns([1.5, 1])
            with col_img:
                st.image(str(img_path), use_container_width=True)
            with col_desc:
                st.markdown(f"**{selected_cat}**")
                st.write(desc_vi if lang == "vi" else desc_en)

    with t4:
        st.markdown("#### " + ("Ma trận tương quan và Hệ số tương quan với biến mục tiêu" if lang == "vi" else "Correlation Heatmap & Target Association"))
        
        p_heat = eda_dir / "06_correlation_heatmap.png"
        p_corr_target = eda_dir / "07_correlation_with_target.png"
        
        c1, c2 = st.columns([1.2, 1])
        with c1:
            if p_heat.exists():
                st.markdown("**" + ("Ma trận tương quan Pearson" if lang == "vi" else "Pearson Correlation Matrix") + "**")
                st.image(str(p_heat), use_container_width=True)
                st.write(
                    "Bản đồ nhiệt ma trận tương quan thể hiện hệ số Pearson r giữa tất cả các cặp thuộc tính. "
                    "Hệ số dao động từ -1.0 (tương quan nghịch tuyệt đối) đến +1.0 (tương quan thuận tuyệt đối).\n\n"
                    "**Ý nghĩa phân tích:** Các hệ số tương quan giữa các biến đầu vào đều nằm trong khoảng từ -0.58 đến 0.48. "
                    "Không có cặp biến nào có tương quan quá mạnh (> 0.8), chứng tỏ dữ liệu không gặp phải lỗi đa cộng tuyến nghiêm trọng. "
                    "Điều này đảm bảo tính ổn định cho các thuật toán học máy tuyến tính (như Logistic Regression)."
                )
            else:
                st.warning("Not found")
        with c2:
            if p_corr_target.exists():
                st.markdown("**" + ("Mức độ tương quan với biến mục tiêu" if lang == "vi" else "Correlation with Target Variable") + "**")
                st.image(str(p_corr_target), use_container_width=True)
                st.write(
                    "Biểu đồ cột sắp xếp các thuộc tính theo độ mạnh của mối tương quan tuyến tính với chẩn đoán bệnh tim (`condition`).\n\n"
                    "- **Tương quan dương mạnh nhất:** `oldpeak` (độ chênh ST), `ca` (số mạch máu tắc), `cp` (loại đau ngực), và `exang` (đau thắt ngực gắng sức). Khi các chỉ số này tăng, nguy cơ mắc bệnh tim tăng cao.\n"
                    "- **Tương quan âm mạnh nhất:** `thalach` (nhịp tim tối đa). Nhịp tim tối đa khi gắng sức càng thấp thì khả năng mắc bệnh tim càng cao."
                )
            else:
                st.warning("Not found")

# ════════════════════════════════════════════
# TRANG 2: LUẬT KẾT HỢP
# ════════════════════════════════════════════
def page_rules():
    lang = get_lang()
    st.markdown(f"#### {t('rules_title', lang)}")

    tab1, tab2 = st.tabs(
        ["Bảng luật kết hợp", "Biểu đồ trực quan hóa luật"]
        if lang == "vi" else
        ["Association Rules Table", "Association Rules Visualization"]
    )
    
    with tab1:
        with st.expander(t("rules_explain_title", lang), expanded=False):
            st.markdown(t("rules_explain_text", lang))

        rules_dir = PROJECT_ROOT / "outputs" / "rules"
        if not rules_dir.exists() or not list(rules_dir.glob("*.csv")):
            st.warning(t("rules_no_file", lang))
            return

        rule_files = [f for f in sorted(rules_dir.glob("*.csv")) if f.stat().st_size > 0]
        selected = st.selectbox(t("rules_select", lang), rule_files, format_func=lambda x: x.name)

        if selected:
            df = pd.read_csv(selected)
            st.caption(f"{t('rules_total', lang)}: {len(df)}")

            c1, c2, c3 = st.columns(3)
            with c1:
                mc = st.slider(t("rules_min_conf", lang), 0.0, 1.0, 0.7, 0.05)
            with c2:
                ml = st.slider(t("rules_min_lift", lang), 0.0, 5.0, 1.2, 0.1)
            with c3:
                mr = st.number_input(t("rules_max_rows", lang), 10, 500, 50, 10)

            filtered = df[(df["confidence"] >= mc) & (df["lift"] >= ml)].head(mr)
            st.write(f"**{t('rules_after_filter', lang)}:** {len(filtered)}")
            st.dataframe(filtered, use_container_width=True, height=400)

            csv = filtered.to_csv(index=False).encode("utf-8-sig")
            st.download_button(t("rules_download", lang), csv, "filtered_rules.csv", "text/csv")

    with tab2:
        st.markdown("#### " + ("Biểu đồ khai phá luật kết hợp" if lang == "vi" else "Association Rules Visualizations"))
        rules_fig_dir = PROJECT_ROOT / "outputs" / "figures" / "association_rules"
        if not rules_fig_dir.exists():
            st.warning(
                "Không tìm thấy thư mục biểu đồ luật kết hợp tại outputs/figures/association_rules/."
                if lang == "vi"
                else "Association rules figures directory not found at outputs/figures/association_rules/."
            )
        else:
            p_sup_conf = rules_fig_dir / "support_vs_confidence.png"
            p_lift_dist = rules_fig_dir / "lift_distribution.png"
            p_top_lift = rules_fig_dir / "top15_rules_by_lift.png"
            p_lift_heat = rules_fig_dir / "lift_heatmap.png"
            
            c1, c2 = st.columns(2)
            with c1:
                if p_sup_conf.exists():
                    st.markdown("**" + ("Biểu đồ Support vs. Confidence" if lang == "vi" else "Support vs. Confidence Scatter Plot") + "**")
                    st.image(str(p_sup_conf), use_container_width=True)
                    st.write(
                        "Biểu đồ phân tán (scatter plot) biểu diễn các luật theo độ hỗ trợ (Support) và độ tin cậy (Confidence). Màu sắc và độ lớn các chấm biểu diễn giá trị Lift.\n\n"
                        "**Ý nghĩa:** Giúp các bác sĩ và nhà nghiên cứu quan sát sự đánh đổi giữa độ phủ của luật (Support) và độ chính xác của luật (Confidence) để tìm ra vùng có nhiều luật hữu ích nhất."
                        if lang == "vi" else
                        "Scatter plot mapping each rule's Support on the x-axis and Confidence on the y-axis. The color and size of the points represent their Lift values.\n\n"
                        "**Analytical Value:** Allows researchers to inspect the trade-off between rule coverage (Support) and certainty (Confidence), identifying dense areas of high-interest rules."
                    )
                
                st.markdown("---")
                
                if p_top_lift.exists():
                    st.markdown("**" + ("Top 15 luật có Lift cao nhất" if lang == "vi" else "Top 15 Rules by Lift") + "**")
                    st.image(str(p_top_lift), use_container_width=True)
                    st.write(
                        "Danh sách 15 luật kết hợp mạnh nhất được sắp xếp giảm dần theo chỉ số Lift.\n\n"
                        "**Ý nghĩa:** Hiển thị trực tiếp các liên kết bệnh lý mạnh mẽ nhất. Ví dụ, sự kết hợp giữa đau ngực không triệu chứng (cp=3) và nhịp tim gắng sức thấp (thalach=low) có chỉ số Lift rất cao, thể hiện liên kết thực tế y khoa rất chặt chẽ."
                        if lang == "vi" else
                        "A horizontal bar chart listing the top 15 association rules ranked in descending order of Lift.\n\n"
                        "**Analytical Value:** Direct exposure of the strongest associations. For instance, combinations like asymptomatic chest pain (cp=3) with low max heart rate (thalach=low) exhibit high Lift, validating clinical intuition."
                    )

            with c2:
                if p_lift_dist.exists():
                    st.markdown("**" + ("Phân phối chỉ số Lift" if lang == "vi" else "Lift Score Distribution") + "**")
                    st.image(str(p_lift_dist), use_container_width=True)
                    st.write(
                        "Biểu đồ phân phối tần suất của chỉ số Lift trên toàn bộ tập luật khai phá.\n\n"
                        "**Ý nghĩa:** Lift lớn hơn 1 biểu thị mối liên hệ tích cực (các thuộc tính xuất hiện cùng nhau nhiều hơn ngẫu nhiên). Phần lớn các luật đều tập trung ở khoảng Lift > 1.2, khẳng định tính liên kết có nghĩa lâm sàng cao của bộ dữ liệu."
                        if lang == "vi" else
                        "Frequency distribution histogram showing the spread of Lift values across the generated rules.\n\n"
                        "**Analytical Value:** A Lift > 1 indicates positive correlation (antecedents and consequents appear together more often than expected by chance). Most rules group above 1.2, proving significant clinical relationships."
                    )
                
                st.markdown("---")
                
                if p_lift_heat.exists():
                    st.markdown("**" + ("Bản đồ nhiệt tương tác (Lift Heatmap)" if lang == "vi" else "Lift Heatmap Matrix") + "**")
                    st.image(str(p_lift_heat), use_container_width=True)
                    st.write(
                        "Bản đồ nhiệt trực quan hóa mối liên hệ giữa các thuộc tính vế trái (antecedents) và vế phải (consequents) dựa trên cường độ chỉ số Lift.\n\n"
                        "**Ý nghĩa:** Cung cấp cái nhìn tổng quan dạng lưới về các cặp triệu chứng/chẩn đoán. Các ô màu đậm chỉ ra sự phụ thuộc chặt chẽ giữa các yếu tố lâm sàng với kết quả chẩn đoán bệnh tim."
                        if lang == "vi" else
                        "Heatmap showing the cross-association matrix between antecedents on the y-axis and consequents on the x-axis, color-coded by Lift.\n\n"
                        "**Analytical Value:** Warmer colors reveal strong dependencies between clinical predictors and heart disease labels."
                    )


# ════════════════════════════════════════════
# TRANG 3: SO SÁNH MÔ HÌNH
# ════════════════════════════════════════════
def page_compare():
    lang = get_lang()
    st.markdown(f"#### {t('compare_title', lang)}")

    tab1, tab2 = st.tabs(
        ["So sánh hiệu suất", "Đánh giá chi tiết (Confusion Matrix & ROC)"]
        if lang == "vi" else
        ["Performance Comparison", "Detailed Evaluation (Confusion Matrix & ROC)"]
    )
    
    with tab1:
        with st.expander(t("compare_explain_title", lang), expanded=False):
            st.markdown(t("compare_explain_text", lang))

        metrics_file = PROJECT_ROOT / "outputs" / "metrics" / "model_comparison.csv"
        if not metrics_file.exists():
            st.warning(t("compare_no_data", lang))
            return

        df = pd.read_csv(metrics_file)

        # Bảng hiển thị
        df_show = df.copy()
        for c in ["accuracy", "precision", "recall", "f1_score"]:
            if c in df_show.columns:
                df_show[c] = df_show[c].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Nút tải CSV
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(t("compare_download_csv", lang), csv, "model_comparison.csv", "text/csv")

        # Biểu đồ bằng matplotlib thuần
        cols = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in df.columns]
        if cols:
            import matplotlib.pyplot as plt
            import numpy as np
            import io

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(df))
            width = 0.18
            colors = ["#1a3a5c", "#2d6a9f", "#e67e22", "#c0392b"]
            
            for i, col in enumerate(cols):
                rects = ax.bar(x + i * width, df[col], width, label=col.replace("_", " ").title(), color=colors[i % 4])
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

            ax.set_ylabel('Score')
            ax.set_title(t("compare_chart_title", lang), fontweight="bold")
            ax.set_xticks(x + width * (len(cols) - 1) / 2)
            ax.set_xticklabels(df["model_name"])
            ax.set_ylim([0, 1.15])
            ax.legend(loc='lower right', ncol=len(cols))
            ax.grid(True, linestyle="--", alpha=0.3)
            
            st.pyplot(fig)

            # Nút tải biểu đồ
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
            st.download_button(t("compare_download_chart", lang), buf.getvalue(), "model_comparison.png", "image/png")

        # Best model
        best_file = PROJECT_ROOT / "outputs" / "metrics" / "best_model_info.json"
        if best_file.exists():
            info = json.loads(best_file.read_text(encoding="utf-8"))
            st.success(f"{t('compare_best', lang)}: **{info.get('model_name')}** "
                       f"(Accuracy: {info.get('accuracy', 0):.2%}, Recall: {info.get('recall', 0):.2%})")

    with tab2:
        st.markdown("#### " + ("Biểu đồ Ma trận nhầm lẫn & Đường cong ROC" if lang == "vi" else "Confusion Matrix & ROC Curves"))
        st.markdown(
            "Chọn thuật toán để xem chi tiết biểu đồ Ma trận nhầm lẫn (Confusion Matrix) và Đường cong ROC (Receiver Operating Characteristic) đã được lưu trong quá trình huấn luyện."
            if lang == "vi" else
            "Select a machine learning algorithm to view its saved Confusion Matrix and ROC Curve plots generated during evaluation."
        )
        
        model_options = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "Random Forest": "random_forest",
            "K-Nearest Neighbors (KNN)": "knn",
            "Naive Bayes": "naive_bayes"
        }
        
        selected_model_name = st.selectbox(
            "Chọn thuật toán:" if lang == "vi" else "Select algorithm:",
            list(model_options.keys())
        )
        
        model_key = model_options[selected_model_name]
        modeling_dir = PROJECT_ROOT / "outputs" / "figures" / "modeling"
        
        p_cm = modeling_dir / f"confusion_matrix_{model_key}.png"
        p_roc = modeling_dir / f"roc_curve_{model_key}.png"
        
        if not p_cm.exists() or not p_roc.exists():
            st.warning(
                f"Không tìm thấy biểu đồ cho mô hình {selected_model_name} trong thư mục outputs/figures/modeling/."
                if lang == "vi" else
                f"Evaluation plots for {selected_model_name} not found in outputs/figures/modeling/."
            )
        else:
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**" + ("Ma trận nhầm lẫn (Confusion Matrix)" if lang == "vi" else "Confusion Matrix") + "**")
                st.image(str(p_cm), use_container_width=True)
                st.write(
                    "**Giải thích ý nghĩa:**\n"
                    "- **True Negative (TN - trên cùng bên trái):** Số ca bình thường được mô hình dự đoán chính xác là bình thường.\n"
                    "- **False Positive (FP - trên cùng bên phải):** Số ca bình thường bị dự đoán nhầm thành mắc bệnh (báo động giả).\n"
                    "- **False Negative (FN - dưới cùng bên trái):** Số ca mắc bệnh bị mô hình bỏ sót, dự đoán nhầm thành bình thường.\n"
                    "- **True Positive (TP - dưới cùng bên phải):** Số ca mắc bệnh được mô hình dự đoán chính xác là mắc bệnh.\n\n"
                    "**Tầm quan trọng y khoa:** Chỉ số **False Negative (FN)** cần phải được giữ ở mức thấp nhất có thể. "
                    "Bỏ sót một bệnh nhân mắc bệnh tim nguy hiểm hơn rất nhiều so với việc chẩn đoán nhầm một người bình thường là có nguy cơ (báo động giả)."
                    if lang == "vi" else
                    "**How to interpret:**\n"
                    "- **True Negative (TN - top-left):** Healthy patients correctly classified as healthy.\n"
                    "- **False Positive (FP - top-right):** Healthy patients incorrectly classified as diseased (false alarm).\n"
                    "- **False Negative (FN - bottom-left):** Diseased patients missed by the model and classified as healthy.\n"
                    "- **True Positive (TP - bottom-right):** Diseased patients correctly classified as diseased.\n\n"
                    "**Clinical Warning:** Minimizing **False Negatives (FN)** is of utmost importance in medical diagnostics. "
                    "Missing a sick patient has far worse consequences than triggering a false alarm on a healthy patient."
                )
                
            with c2:
                st.markdown("**" + ("Đường cong ROC & Chỉ số AUC" if lang == "vi" else "ROC Curve & AUC Score") + "**")
                st.image(str(p_roc), use_container_width=True)
                st.write(
                    "**Giải thích ý nghĩa:**\n"
                    "- **Đường cong ROC:** Trực quan hóa tỷ lệ dự đoán đúng bệnh (True Positive Rate) so với tỷ lệ báo động giả (False Positive Rate) trên mọi ngưỡng quyết định xác suất khác nhau (từ 0 đến 1).\n"
                    "- **AUC (Area Under the Curve):** Diện tích dưới đường cong ROC. Chỉ số này phản ánh khả năng phân tách (phân biệt) bệnh nhân mắc bệnh và lành mạnh của mô hình.\n\n"
                    "**Đánh giá hiệu năng:** AUC gần bằng 1.0 biểu thị khả năng chẩn đoán hoàn hảo. "
                    "AUC = 0.5 tương đương với việc đoán ngẫu nhiên. Các mô hình tốt thường có chỉ số AUC trên 0.85, đảm bảo độ tin cậy khi triển khai thực tế."
                    if lang == "vi" else
                    "**How to interpret:**\n"
                    "- **ROC Curve:** Plots the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity) for all decision thresholds between 0 and 1.\n"
                    "- **AUC (Area Under the Curve):** Represents the probability that the model will rank a randomly chosen diseased patient higher than a healthy one.\n\n"
                    "**Performance Metric:** An AUC score close to 1.0 signifies perfect discrimination capability. "
                    "An AUC of 0.5 represents random guessing. Highly reliable diagnostic models usually maintain an AUC score above 0.85."
                )
# ════════════════════════════════════════════
# TRANG 3.5: GOM CỤM BỆNH NHÂN (CLUSTERING)
# ════════════════════════════════════════════
def page_clustering():
    lang = get_lang()
    st.markdown(f"#### {t('nav_clustering', lang)}")
    
    if lang == "vi":
        st.markdown(
            "Phân cụm bệnh nhân (**Clustering**) là kỹ thuật học không giám sát nhằm tự động "
            "gom nhóm bệnh nhân có các chỉ số sức khỏe và lâm sàng tương đồng. "
            "Dự án sử dụng thuật toán **K-Means Clustering** kết hợp với **PCA (Phân tích thành phần chính)** "
            "để trực quan hóa dữ liệu trên mặt phẳng 2D."
        )
    else:
        st.markdown(
            "Patient **Clustering** is an unsupervised learning technique to automatically "
            "group patients with similar health profiles. The project applies **K-Means Clustering** "
            "combined with **PCA (Principal Component Analysis)** for 2D visualization."
        )
        
    metrics_file = PROJECT_ROOT / "outputs" / "metrics" / "clustering_metrics.json"
    summary_file = PROJECT_ROOT / "outputs" / "metrics" / "clustering_summary.csv"
    pca_img = PROJECT_ROOT / "outputs" / "figures" / "clustering_pca.png"
    
    if not metrics_file.exists() or not summary_file.exists() or not pca_img.exists():
        st.warning("Chưa chạy phân cụm. Vui lòng chạy lệnh: `python main.py --cluster`" if lang == "vi" else "Clustering results not found. Run: `python main.py --cluster`")
        return
        
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    sil = metrics.get("silhouette_score", 0)
    n_clus = metrics.get("n_clusters", 3)
    
    st.success(f"Silhouette Score: **{sil:.4f}** (n_clusters = {n_clus})")
    
    col1, col2 = st.columns([1.2, 1], gap="medium")
    
    with col1:
        st.markdown("**Biểu đồ phân bố cụm (PCA 2D)**" if lang == "vi" else "**Cluster Distribution (PCA 2D)**")
        st.image(str(pca_img), use_container_width=True)
        
    with col2:
        st.markdown("**Bảng đặc trưng trung bình của các cụm**" if lang == "vi" else "**Mean Characteristics of Clusters**")
        df_summary = pd.read_csv(summary_file)
        
        # Làm tròn số cho đẹp
        df_summary_show = df_summary.copy()
        for c in df_summary_show.columns:
            if c != "Cluster" and c != "Patient_Count":
                df_summary_show[c] = df_summary_show[c].apply(lambda x: f"{x:.2f}")
        st.dataframe(df_summary_show, use_container_width=True, hide_index=True)
        
    st.markdown("---")
    st.markdown("**Ý nghĩa lâm sàng của từng cụm bệnh nhân:**" if lang == "vi" else "**Clinical Insights of Clusters:**")
    
    if lang == "vi":
        st.markdown(
            "- **Cụm 0 (Nguy cơ Thấp - Bệnh nhân trẻ):** Gồm các bệnh nhân có độ tuổi trung bình trẻ nhất (~47 tuổi), "
            "nhịp tim gắng sức tối đa đạt mức cao nhất (~162 bpm), chỉ số tổn thương tim ST rất thấp (~0.51). "
            "Tỷ lệ mắc bệnh tim thực tế trong cụm này rất thấp, chỉ **22.9%**.\n"
            "- **Cụm 1 (Nguy cơ Cao - Bệnh lý nặng):** Gồm bệnh nhân lớn tuổi (~58 tuổi), chỉ số huyết áp và cholesterol cao, "
            "nhịp tim tối đa thấp (~130 bpm), mức độ tổn thương ST cơ tim rất cao (~1.89) và số mạch máu bị tắc nghẽn lớn. "
            "Tỷ lệ mắc bệnh thực tế lên tới **90.2%**.\n"
            "- **Cụm 2 (Quá tải mạch - Ít tổn thương cơ tim):** Gồm bệnh nhân lớn tuổi (~58 tuổi) có chỉ số huyết áp (~141 mmHg) "
            "và cholesterol (~262 mg/dl) cao nhất hệ thống, nhưng nhịp tim gắng sức vẫn khá tốt (~155 bpm) và chỉ số tổn thương ST thấp. "
            "Tỷ lệ mắc bệnh thực tế thấp, chỉ **22.3%**."
        )
    else:
        st.markdown(
            "- **Cluster 0 (Low Risk - Younger Patients):** Features the youngest average age (~47), highest max heart rate (~162 bpm), "
            "and very low ST depression (~0.51). Actual heart disease rate is only **22.9%**.\n"
            "- **Cluster 1 (High Risk - Severe Patients):** Older patients (~58) with high blood pressure and cholesterol, "
            "low max heart rate (~130 bpm), high ST depression (~1.89), and blocked major vessels. Actual disease rate is **90.2%**.\n"
            "- **Cluster 2 (Cardiovascular Strain - Low Heart Lesion):** Older patients (~58) with the highest blood pressure (~141 mmHg) "
            "and cholesterol (~262 mg/dl), but maintaining high max heart rate (~155 bpm) and low ST depression. Actual disease rate is **22.3%**."
        )


# ════════════════════════════════════════════
# TRANG 3.6: THỐNG KÊ & HỒI QUY (ANALYTICS)
# ════════════════════════════════════════════
def page_analytics():
    lang = get_lang()
    st.markdown(f"#### {t('nav_analytics', lang)}")
    
    tab1, tab2 = st.tabs(
        ["Kiểm định Thống kê (Hypothesis Tests)", "Phân tích Hồi quy (Regression Models)"] 
        if lang == "vi" else 
        ["Hypothesis Tests", "Regression Models"]
    )
    
    with tab1:
        st.markdown(
            "#### Kiểm định giả thuyết ANOVA & Chi-Square" if lang == "vi" else "#### ANOVA & Chi-Square Hypothesis Testing"
        )
        if lang == "vi":
            st.markdown(
                "Kiểm định thống kê giúp xác định xem các thuộc tính đầu vào có ảnh hưởng "
                "có ý nghĩa về mặt toán học đối với chẩn đoán bệnh tim (diagnosis) hay không:\n"
                "- **ANOVA** được áp dụng cho các thuộc tính liên tục để so sánh giá trị trung bình giữa nhóm lành mạnh và nhóm mắc bệnh.\n"
                "- **Chi-Square** độc lập được áp dụng cho các thuộc tính phân loại để đánh giá mức độ liên quan."
            )
        else:
            st.markdown(
                "Hypothesis tests verify if input features have a statistically significant "
                "relationship with the heart disease diagnosis target:\n"
                "- **ANOVA** compares the means of continuous variables between healthy and disease groups.\n"
                "- **Chi-Square** checks the association between categorical variables and the target."
            )
            
        stats_file = PROJECT_ROOT / "outputs" / "metrics" / "statistical_tests.json"
        if not stats_file.exists():
            st.warning("Chưa chạy kiểm định thống kê. Vui lòng chạy lệnh: `python main.py --analytics`" if lang == "vi" else "Statistical tests results not found. Run: `python main.py --analytics`")
        else:
            with open(stats_file, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
                
            col1, col2 = st.columns(2, gap="medium")
            
            with col1:
                st.markdown("**Kết quả ANOVA (Thuộc tính số)**" if lang == "vi" else "**ANOVA Results (Continuous)**")
                anova_rows = []
                for k, v in stats_data.get("anova", {}).items():
                    anova_rows.append([
                        k, f"{v['f_statistic']:.4f}", f"{v['p_value']:.4g}", 
                        "Có (Significant)" if v['is_significant'] else "Không (Not Significant)"
                    ])
                st.dataframe(pd.DataFrame(anova_rows, columns=[
                    "Thuộc tính", "F-Statistic", "p-value", "Ảnh hưởng ý nghĩa"
                ] if lang == "vi" else [
                    "Feature", "F-Statistic", "p-value", "Significant Effect"
                ]), use_container_width=True, hide_index=True)
                
            with col2:
                st.markdown("**Kết quả Chi-Square (Thuộc tính phân loại)**" if lang == "vi" else "**Chi-Square Results (Categorical)**")
                chi_rows = []
                for k, v in stats_data.get("chi_square", {}).items():
                    chi_rows.append([
                        k, f"{v['chi2_statistic']:.4f}", f"{v['p_value']:.4g}", 
                        "Có (Significant)" if v['is_significant'] else "Không (Not Significant)"
                    ])
                st.dataframe(pd.DataFrame(chi_rows, columns=[
                    "Thuộc tính", "Chi2-Statistic", "p-value", "Liên quan ý nghĩa"
                ] if lang == "vi" else [
                    "Feature", "Chi2-Statistic", "p-value", "Significant Relation"
                ]), use_container_width=True, hide_index=True)
                
    with tab2:
        st.markdown(
            "#### Dự đoán Nhịp tim tối đa bằng Hồi quy" if lang == "vi" else "#### Predicting Max Heart Rate via Regression"
        )
        if lang == "vi":
            st.markdown(
                "Bài toán đặt ra: Dự đoán chỉ số nhịp tim tối đa gắng sức (**thalach**) dựa trên tuổi tác (**age**) "
                "và các đặc trưng liên tục khác. Ứng dụng so sánh 3 phương pháp hồi quy phổ biến:"
            )
        else:
            st.markdown(
                "Task: Predict the maximum heart rate (**thalach**) based on age (**age**) and other continuous features. "
                "The app compares 3 regression techniques:"
            )
            
        reg_file = PROJECT_ROOT / "outputs" / "metrics" / "regression_comparison.csv"
        reg_img = PROJECT_ROOT / "outputs" / "figures" / "regression_fit.png"
        
        if not reg_file.exists() or not reg_img.exists():
            st.warning("Chưa chạy phân tích hồi quy. Vui lòng chạy lệnh: `python main.py --analytics`" if lang == "vi" else "Regression results not found. Run: `python main.py --analytics`")
        else:
            col1, col2 = st.columns([1, 1.2], gap="medium")
            
            with col1:
                st.markdown("**Bảng so sánh mô hình Hồi quy**" if lang == "vi" else "**Regression Model Comparison**")
                df_reg = pd.read_csv(reg_file)
                st.dataframe(df_reg, use_container_width=True, hide_index=True)
                
                if lang == "vi":
                    st.info(
                        "**Ý nghĩa chỉ số:**\n"
                        "- **MSE (Sai số bình phương trung bình):** Càng thấp thì mô hình dự đoán càng sát thực tế.\n"
                        "- **R2 score (Hệ số xác định):** Biểu thị tỷ lệ phần trăm sự biến thiên của nhịp tim có thể giải thích bởi các biến đầu vào. R2 càng gần 1 mô hình càng mạnh."
                    )
                else:
                    st.info(
                        "**Metric Explanation:**\n"
                        "- **MSE (Mean Squared Error):** Lower means prediction is closer to actual values.\n"
                        "- **R2 score (Coefficient of Determination):** Percentage of heart rate variance explained by features. Closer to 1 is stronger."
                    )
            with col2:
                st.markdown("**Đồ thị đường khớp Hồi quy theo Tuổi**" if lang == "vi" else "**Regression Fit Curves vs Age**")
                st.image(str(reg_img), use_container_width=True)


# ════════════════════════════════════════════
# TRANG 4: HƯỚNG DẪN
# ════════════════════════════════════════════
def page_guide():
    lang = get_lang()
    st.markdown(f"#### {t('guide_title', lang)}")
    st.markdown(t("guide_content", lang))

    st.markdown(f"---\n**{t('attr_table_header', lang)}**")
    rows = [
        ["age", "int", "29–77", "Tuổi bệnh nhân" if lang == "vi" else "Patient age"],
        ["sex", "binary", "0, 1", "1=Nam, 0=Nữ" if lang == "vi" else "1=Male, 0=Female"],
        ["cp", "cat", "0–3", "Loại đau ngực" if lang == "vi" else "Chest pain type"],
        ["trestbps", "int", "94–200", "Huyết áp lúc nghỉ (mm Hg)" if lang == "vi" else "Resting blood pressure"],
        ["chol", "int", "126–564", "Cholesterol (mg/dl)" if lang == "vi" else "Serum cholesterol"],
        ["fbs", "binary", "0, 1", "Đường huyết > 120" if lang == "vi" else "Fasting blood sugar > 120"],
        ["restecg", "cat", "0–2", "ECG lúc nghỉ" if lang == "vi" else "Resting ECG"],
        ["thalach", "int", "71–202", "Nhịp tim tối đa" if lang == "vi" else "Max heart rate"],
        ["exang", "binary", "0, 1", "Đau ngực khi gắng sức" if lang == "vi" else "Exercise angina"],
        ["oldpeak", "float", "0–6.2", "Độ chênh ST (mm)" if lang == "vi" else "ST depression"],
        ["slope", "cat", "0–2", "Độ dốc đoạn ST" if lang == "vi" else "ST slope"],
        ["ca", "int", "0–3", "Số mạch máu nhuộm màu" if lang == "vi" else "Major vessels colored"],
        ["thal", "cat", "0–2", "Thalassemia" if lang == "vi" else "Thalassemia"],
    ]
    st.dataframe(pd.DataFrame(rows, columns=[
        t("attr_col_name", lang), t("attr_col_type", lang),
        t("attr_col_range", lang), t("attr_col_desc", lang)
    ]), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TRANG 5: THÔNG TIN
# ════════════════════════════════════════════
def page_about():
    lang = get_lang()
    st.markdown(f"#### {t('about_title', lang)}")
    st.markdown(t("about_content", lang))


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
def main():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "current_result" not in st.session_state:
        st.session_state["current_result"] = None
        # Ưu tiên lấy từ biến môi trường hoặc Streamlit Secrets, nếu không thì để trống để người dùng nhập ở sidebar
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            try:
                api_key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                pass
        st.session_state["gemini_api_key"] = api_key

    lang = get_lang()

    st.markdown(
        f'<div class="header">{t("app_title", lang)}'
        f'<small>{t("app_subtitle", lang)}</small></div>',
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.markdown(f"### {t('sidebar_title', lang)}")
    st.sidebar.caption(t("sidebar_caption", lang))
    st.sidebar.markdown("---")

    # Language toggle
    lang_options = {"Tiếng Việt": "vi", "English": "en"}
    chosen = st.sidebar.selectbox(
        t("language_label", lang),
        list(lang_options.keys()),
        index=0 if lang == "vi" else 1,
    )
    new_lang = lang_options[chosen]
    if new_lang != lang:
        st.session_state["lang"] = new_lang
        st.rerun()

    st.sidebar.markdown("---")

    nav_keys = ["nav_diagnosis", "nav_history", "nav_eda", "nav_rules", "nav_compare", "nav_clustering", "nav_analytics", "nav_guide", "nav_about"]
    nav_labels = [t(k, lang) for k in nav_keys]
    menu = st.sidebar.radio(t("nav_label", lang), nav_labels)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Gemini AI Config**" if lang == "vi" else "**Gemini AI Config**")
    
    default_key = os.environ.get("GEMINI_API_KEY", "")
    
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.get("gemini_api_key", default_key),
        help="Nhập Gemini API Key để kích hoạt lời khuyên tự động từ bác sĩ AI." 
             if lang == "vi" else 
             "Enter Gemini API Key to enable automated AI doctor advice."
    )
    if gemini_key:
        st.session_state["gemini_api_key"] = gemini_key

    st.sidebar.markdown("---")
    st.sidebar.caption("Version 1.0")

    # Routing
    idx = nav_labels.index(menu)
    [page_diagnosis, page_history, page_eda, page_rules, page_compare, page_clustering, page_analytics, page_guide, page_about][idx]()

    st.markdown("---")
    st.markdown(f'<div class="footer">{t("footer", lang)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()