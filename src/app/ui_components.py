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
    "ca": "Số mạch vành lớn phát hiện qua chụp huỳnh quang (0–3).",
    "thal": "0: bình thường. 1: khuyết tật cố định (nhồi máu cũ). 2: khuyết tật hồi phục (mạch vành hoạt động).",
}


def input_form(lang: str = "vi") -> Tuple[bool, Dict[str, Any]]:
    with st.form("patient_form"):

        st.markdown(f"**{t('form_demo', lang)}**")
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input(t("f_age", lang), 1, 120, 50, 1, help=HELP["age"])
        with col_b:
            sex = st.selectbox(t("f_sex", lang), [1, 0],
                               format_func=lambda x: t("sex_male", lang) if x == 1 else t("sex_female", lang),
                               help=HELP["sex"])

        st.markdown(f"**{t('form_symptom', lang)}**")
        col_c, col_d = st.columns(2)
        with col_c:
            cp = st.selectbox(t("f_cp", lang), [0, 1, 2, 3],
                              format_func=lambda x: t(f"cp_{x}", lang), help=HELP["cp"])
            exang = st.selectbox(t("f_exang", lang), [0, 1],
                                 format_func=lambda x: t("yes", lang) if x == 1 else t("no", lang),
                                 help=HELP["exang"])
        with col_d:
            trestbps = st.number_input(t("f_trestbps", lang), 80, 250, 130, 1, help=HELP["trestbps"])
            chol = st.number_input(t("f_chol", lang), 100, 600, 240, 1, help=HELP["chol"])

        st.markdown(f"**{t('form_test', lang)}**")
        col_e, col_f = st.columns(2)
        with col_e:
            fbs = st.selectbox(t("f_fbs", lang), [0, 1],
                               format_func=lambda x: t("fbs_yes", lang) if x == 1 else t("fbs_no", lang),
                               help=HELP["fbs"])
            restecg = st.selectbox(t("f_restecg", lang), [0, 1, 2],
                                   format_func=lambda x: t(f"restecg_{x}", lang), help=HELP["restecg"])
            thalach = st.number_input(t("f_thalach", lang), 60, 250, 150, 1, help=HELP["thalach"])
        with col_f:
            oldpeak = st.number_input(t("f_oldpeak", lang), 0.0, 7.0, 1.0, 0.1, help=HELP["oldpeak"])
            slope = st.selectbox(t("f_slope", lang), [0, 1, 2],
                                 format_func=lambda x: t(f"slope_{x}", lang), help=HELP["slope"])
            ca = st.selectbox(t("f_ca", lang), [0, 1, 2, 3], help=HELP["ca"])

        st.markdown(f"**{t('form_extra', lang)}**")
        thal = st.selectbox(t("f_thal", lang), [0, 1, 2],
                            format_func=lambda x: t(f"thal_{x}", lang), help=HELP["thal"])

        st.markdown("---")
        submitted = st.form_submit_button(t("diag_submit", lang), use_container_width=True)

    data = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal}
    return submitted, data