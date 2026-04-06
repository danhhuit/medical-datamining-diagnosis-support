import streamlit as st
import pandas as pd

from src.app.ui_components import input_form
from src.app.display_result import show_result
from src.models.predict import predict
from src.preprocessing.transform_data import transform_input

# ===== CONFIG =====
st.set_page_config(
    page_title="Hospital AI System",
    page_icon="🏥",
    layout="wide"
)

# ===== CSS STYLE =====
st.markdown("""
<style>
/* Header */
.header {
    background-color: #0B5ED7;
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

/* Card */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}

/* Section title */
.section-title {
    font-size: 20px;
    font-weight: bold;
    color: #0B5ED7;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


def main():
    # ===== HEADER =====
    st.markdown('<div class="header">🏥 HỆ THỐNG CHẨN ĐOÁN Y KHOA AI</div>', unsafe_allow_html=True)

    st.write("")

    # ===== SIDEBAR =====
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966485.png", width=80)
    st.sidebar.title("🏥 Hospital System")

    menu = st.sidebar.radio(
        "📌 Chức năng",
        ["Chẩn đoán", "Hướng dẫn", "Thông tin"]
    )

    # ===== PAGE: CHẨN ĐOÁN =====
    if menu == "Chẩn đoán":

        col1, col2 = st.columns([1, 1])

        # ===== INPUT =====
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🧾 Thông tin bệnh nhân</div>', unsafe_allow_html=True)

            submitted, user_data = input_form()

            st.markdown('</div>', unsafe_allow_html=True)

        # ===== RESULT =====
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Kết quả chẩn đoán</div>', unsafe_allow_html=True)

            if submitted:
                df = pd.DataFrame([user_data])

                with st.spinner("🔍 Đang phân tích..."):
                    processed_data = transform_input(df)
                    prediction, prob = predict(processed_data)

                show_result(prediction, prob)
            else:
                st.info("👉 Nhập thông tin để xem kết quả")

            st.markdown('</div>', unsafe_allow_html=True)

    # ===== PAGE: HƯỚNG DẪN =====
    elif menu == "Hướng dẫn":
        st.markdown("### 📘 Hướng dẫn sử dụng")
        st.write("""
        1. Nhập thông tin bệnh nhân  
        2. Nhấn nút "Chẩn đoán"  
        3. Xem kết quả và gợi ý điều trị  
        """)

    # ===== PAGE: THÔNG TIN =====
    elif menu == "Thông tin":
        st.markdown("### ℹ️ Giới thiệu hệ thống")
        st.write("""
        - Ứng dụng sử dụng Machine Learning  
        - Dữ liệu từ Kaggle  
        - Hỗ trợ chẩn đoán bệnh cơ bản  
        """)

    # ===== FOOTER =====
    st.write("---")
    st.markdown("© 2026 Hospital AI System")


if __name__ == "__main__":
    main()