import streamlit as st

def input_form():
    age = st.number_input("👤 Tuổi", 0, 120, 25)
    gender = st.selectbox("⚧ Giới tính", ["Nam", "Nữ"])

    st.markdown("### 🧬 Triệu chứng")

    fever = st.checkbox("Sốt")
    cough = st.checkbox("Ho")
    fatigue = st.checkbox("Mệt mỏi")

    submitted = st.button("🩺 Chẩn đoán")

    data = {
        "age": age,
        "gender": 1 if gender == "Nam" else 0,
        "fever": int(fever),
        "cough": int(cough),
        "fatigue": int(fatigue),
    }

    return submitted, data