"""
Form nhập 13 thuộc tính lâm sàng – hỗ trợ đa ngôn ngữ.
"""
from __future__ import annotations

import streamlit as st
from typing import Tuple, Dict, Any

from src.app.translations import t

# Help text (giữ nguyên tiếng Việt vì nội dung chuyên ngành y khoa)
HELP = {
    "age": "Tuổi của bệnh nhân (năm). Khoảng dữ liệu: 29–77.",
    "sex": "Giới tính sinh học. 1 = Nam, 0 = Nữ.",
    "cp": ("Chest Pain type – Phân loại cơn đau ngực.\n"
           "0: đau thắt ngực điển hình. 1: không điển hình. 2: không do tim. 3: không triệu chứng."),
    "trestbps": "Huyết áp tâm thu lúc nghỉ (mm Hg). Bình thường < 120, Tăng HA >= 140.",
    "chol": "Cholesterol toàn phần (mg/dl). Bình thường < 200, Cao >= 240.",
    "fbs": "Đường huyết lúc đói > 120 mg/dl gợi ý tiền đái tháo đường.",
    "restecg": "ECG lúc nghỉ. 0: bình thường. 1: bất thường ST-T. 2: phì đại thất trái.",
    "thalach": "Nhịp tim tối đa đạt được khi gắng sức. Lý thuyết: 220 − tuổi.",
    "exang": "Đau thắt ngực khi gắng sức gợi ý mạch vành bị hẹp.",
    "oldpeak": "Mức chênh xuống đoạn ST khi gắng sức (mm). Cao = thiếu máu cơ tim.",
    "slope": "Hình dạng đoạn ST. 0: dốc lên (bình thường). 1: phẳng. 2: dốc xuống (nghi ngờ).",
    "ca": "Số mạch vành lớn phát hiện qua chụp huỳnh quang (0–4).",
    "thal": "0: bình thường. 1: khuyết tật cố định (nhồi máu cũ). 2: khuyết tật hồi phục (mạch vành hoạt động). 3: không xác định (unknown).",
}


def input_form(lang: str = "vi") -> Tuple[bool, Dict[str, Any]]:
    # Khởi tạo giá trị mặc định nếu chưa có trong session state
    defaults = {
        "age": 50, "sex": 1, "cp": 0, "trestbps": 130, "chol": 240,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 1.0, "slope": 0, "ca": 0, "thal": 0
    }
    for k, v in defaults.items():
        state_key = f"form_{k}"
        if state_key not in st.session_state:
            st.session_state[state_key] = v

    with st.form("patient_form"):

        st.markdown(f"**{t('form_demo', lang)}**")
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input(t("f_age", lang), 1, 120, key="form_age", help=HELP["age"])
        with col_b:
            sex_list = [1, 0]
            sex = st.selectbox(t("f_sex", lang), sex_list, key="form_sex",
                               format_func=lambda x: t("sex_male", lang) if x == 1 else t("sex_female", lang),
                               help=HELP["sex"])

        st.markdown(f"**{t('form_symptom', lang)}**")
        col_c, col_d = st.columns(2)
        with col_c:
            cp_list = [0, 1, 2, 3]
            cp = st.selectbox(t("f_cp", lang), cp_list, key="form_cp",
                              format_func=lambda x: t(f"cp_{x}", lang), help=HELP["cp"])
            
            exang_list = [0, 1]
            exang = st.selectbox(t("f_exang", lang), exang_list, key="form_exang",
                                 format_func=lambda x: t("yes", lang) if x == 1 else t("no", lang),
                                 help=HELP["exang"])
        with col_d:
            trestbps = st.number_input(t("f_trestbps", lang), 80, 250, key="form_trestbps", help=HELP["trestbps"])
            chol = st.number_input(t("f_chol", lang), 100, 600, key="form_chol", help=HELP["chol"])

        st.markdown(f"**{t('form_test', lang)}**")
        col_e, col_f = st.columns(2)
        with col_e:
            fbs_list = [0, 1]
            fbs = st.selectbox(t("f_fbs", lang), fbs_list, key="form_fbs",
                               format_func=lambda x: t("fbs_yes", lang) if x == 1 else t("fbs_no", lang),
                               help=HELP["fbs"])
            
            restecg_list = [0, 1, 2]
            restecg = st.selectbox(t("f_restecg", lang), restecg_list, key="form_restecg",
                                   format_func=lambda x: t(f"restecg_{x}", lang), help=HELP["restecg"])
            
            thalach = st.number_input(t("f_thalach", lang), 60, 250, key="form_thalach", help=HELP["thalach"])
        with col_f:
            oldpeak = st.number_input(t("f_oldpeak", lang), 0.0, 7.0, step=0.1, key="form_oldpeak", help=HELP["oldpeak"])
            
            slope_list = [0, 1, 2]
            slope = st.selectbox(t("f_slope", lang), slope_list, key="form_slope",
                                 format_func=lambda x: t(f"slope_{x}", lang), help=HELP["slope"])
            
            ca_list = [0, 1, 2, 3, 4]
            ca = st.selectbox(t("f_ca", lang), ca_list, key="form_ca", help=HELP["ca"])

        st.markdown(f"**{t('form_extra', lang)}**")
        thal_list = [0, 1, 2, 3]
        thal = st.selectbox(t("f_thal", lang), thal_list, key="form_thal",
                            format_func=lambda x: t(f"thal_{x}", lang), help=HELP["thal"])

        st.markdown("---")
        submitted = st.form_submit_button(t("diag_submit", lang), use_container_width=True)

    data = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal}
    return submitted, data


def validate_inputs(data: Dict[str, Any], lang: str = "vi") -> str | None:
    """
    Kiểm tra xem các giá trị nhập vào có hợp lệ và nằm trong phạm vi của widget không.
    Trả về chuỗi thông báo lỗi nếu không hợp lệ, ngược lại trả về None.
    """
    # 1. Tuổi
    if not (1 <= data.get("age", 0) <= 120):
        return (
            "Tuổi phải nằm trong khoảng từ 1 đến 120."
            if lang == "vi"
            else "Age must be between 1 and 120."
        )
    # 2. Huyết áp lúc nghỉ
    if not (80 <= data.get("trestbps", 0) <= 250):
        return (
            "Huyết áp lúc nghỉ phải nằm trong khoảng từ 80 đến 250 mm Hg."
            if lang == "vi"
            else "Resting blood pressure must be between 80 and 250 mm Hg."
        )
    # 3. Cholesterol
    if not (100 <= data.get("chol", 0) <= 600):
        return (
            "Cholesterol phải nằm trong khoảng từ 100 đến 600 mg/dl."
            if lang == "vi"
            else "Serum cholesterol must be between 100 and 600 mg/dl."
        )
    # 4. Nhịp tim tối đa
    if not (60 <= data.get("thalach", 0) <= 250):
        return (
            "Nhịp tim tối đa phải nằm trong khoảng từ 60 đến 250 bpm."
            if lang == "vi"
            else "Maximum heart rate must be between 60 and 250 bpm."
        )
    # 5. Độ chênh ST
    if not (0.0 <= data.get("oldpeak", 0.0) <= 7.0):
        return (
            "Độ chênh ST (oldpeak) phải nằm trong khoảng từ 0.0 đến 7.0 mm."
            if lang == "vi"
            else "ST depression (oldpeak) must be between 0.0 and 7.0 mm."
        )
    return None