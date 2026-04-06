import streamlit as st

def show_result(prediction, probability):

    if prediction == 1:
        st.error("⚠️ Nguy cơ mắc bệnh")
    else:
        st.success("✅ Bình thường")

    st.markdown("### 📈 Độ tin cậy")
    st.progress(int(probability * 100))
    st.write(f"{round(probability*100,2)}%")

    st.markdown("### 💊 Gợi ý")

    if prediction == 1:
        st.warning("""
        - Nghỉ ngơi
        - Uống thuốc theo chỉ định
        - Khám bác sĩ
        """)
    else:
        st.info("""
        - Giữ sức khỏe tốt
        - Tập thể dục
        """)