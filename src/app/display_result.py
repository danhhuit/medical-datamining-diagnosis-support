"""
Trình bày kết quả chẩn đoán dạng bảng, biểu đồ, gợi ý điều trị và nút tải file.
"""
from __future__ import annotations

import io
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from src.app.translations import t


def show_result(result: Dict[str, Any], lang: str = "vi", inputs: Dict[str, Any] = None) -> None:
    prediction = result.get("prediction", 0)
    model_name = result.get("model_name", "Unknown")
    prob = result.get("positive_class_probability", 0.5)
    probabilities = result.get("probabilities", None)

    # ── Kết quả chính ──
    label = t("result_label_positive", lang) if prediction == 1 else t("result_label_negative", lang)
    if prediction == 1:
        st.error(f"**{t('result_header', lang)}: {label}**")
    else:
        st.success(f"**{t('result_header', lang)}: {label}**")

    # ── Bảng tổng hợp (thay st.metric) ──
    risk = t("result_risk_high", lang) if prediction == 1 else t("result_risk_low", lang)
    summary_df = pd.DataFrame([
        {t("result_col_item", lang): t("result_model", lang), t("result_col_value", lang): model_name.replace("_", " ").title()},
        {t("result_col_item", lang): t("result_risk", lang), t("result_col_value", lang): risk},
        {t("result_col_item", lang): t("result_prob", lang), t("result_col_value", lang): f"{prob * 100:.1f}%"},
    ])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Thanh tiến trình ──
    st.caption(t("result_prob_bar", lang))
    st.progress(min(int(prob * 100), 100))

    # ── Biểu đồ phân phối xác suất bằng matplotlib ──
    if probabilities and len(probabilities) == 2:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x_labels = [t("result_chart_no", lang), t("result_chart_yes", lang)]
        y_values = [probabilities[0] * 100, probabilities[1] * 100]
        colors = ["#27ae60", "#c0392b"]
        
        rects = ax.bar(x_labels, y_values, color=colors, width=0.4)
        
        # Thêm nhãn xác suất dạng phần trăm lên đầu cột
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
        ax.set_ylabel(t("result_chart_y", lang))
        ax.set_title(t("result_chart_title", lang), fontsize=10, fontweight="bold")
        ax.set_ylim([0, 115])
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

    # ── Giải thích kết quả ──
    with st.expander(t("result_explain_title", lang), expanded=False):
        txt = t("result_explain_text", lang)
        if probabilities and len(probabilities) == 2:
            st.markdown(
                f"**{t('result_model', lang)}:** {model_name.replace('_', ' ').title()}\n\n"
                f"- **{t('result_chart_no', lang)}:** {probabilities[0]*100:.1f}%\n"
                f"- **{t('result_chart_yes', lang)}:** {probabilities[1]*100:.1f}%\n\n"
                + txt
            )
        else:
            st.markdown(txt)

    # ── Gợi ý tham khảo bằng Gemini AI ──
    with st.expander(t("result_suggest_title", lang), expanded=True):
        api_key = st.session_state.get("gemini_api_key", "").strip()
        if not api_key:
            st.warning(
                "Vui lòng nhập **Gemini API Key** ở thanh Sidebar bên trái để hệ thống sinh lời khuyên chẩn đoán cá nhân hóa bằng AI."
                if lang == "vi" else
                "Please enter your **Gemini API Key** in the left sidebar to generate personalized AI medical advice."
            )
        else:
            try:
                import google.generativeai as genai
                import hashlib
                import json
                
                # Tạo hash của inputs + prediction để nhận diện tính duy nhất và cache kết quả
                inputs_str = json.dumps(inputs, sort_keys=True) if inputs else ""
                cache_key = hashlib.md5(f"{inputs_str}_{prediction}_{lang}".encode("utf-8")).hexdigest()
                
                if "gemini_advice_cache" not in st.session_state:
                    st.session_state["gemini_advice_cache"] = {}
                
                if cache_key in st.session_state["gemini_advice_cache"]:
                    advice_text = st.session_state["gemini_advice_cache"][cache_key]
                    st.markdown(advice_text)
                else:
                    with st.spinner(
                        "Đang phân tích hồ sơ và kết nối với Gemini AI để xây dựng lời khuyên..."
                        if lang == "vi" else
                        "Analyzing profile and connecting to Gemini AI to generate advice..."
                    ):
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        
                        # Xây dựng prompt
                        if lang == "vi":
                            gender_str = "Nam" if inputs.get("sex") == 1 else "Nữ"
                            cp_labels = {
                                0: "Đau thắt ngực điển hình (typical angina)",
                                1: "Đau thắt ngực không điển hình (atypical angina)",
                                2: "Đau không do tim (non-anginal pain)",
                                3: "Không có triệu chứng (asymptomatic)"
                            }
                            cp_str = cp_labels.get(inputs.get("cp", 0), str(inputs.get("cp")))
                            fbs_str = "Có (> 120 mg/dl)" if inputs.get("fbs") == 1 else "Không (<= 120 mg/dl)"
                            restecg_labels = {
                                0: "Bình thường",
                                1: "Bất thường sóng ST-T",
                                2: "Phì đại thất trái"
                            }
                            restecg_str = restecg_labels.get(inputs.get("restecg", 0), str(inputs.get("restecg")))
                            exang_str = "Có" if inputs.get("exang") == 1 else "Không"
                            slope_labels = {
                                0: "Dốc lên (Upsloping)",
                                1: "Bằng phẳng (Flat)",
                                2: "Dốc xuống (Downsloping)"
                            }
                            slope_str = slope_labels.get(inputs.get("slope", 0), str(inputs.get("slope")))
                            thal_labels = {
                                0: "Bình thường",
                                1: "Khuyết tật cố định",
                                2: "Khuyết tật hồi phục"
                            }
                            thal_str = thal_labels.get(inputs.get("thal", 0), str(inputs.get("thal")))
                            
                            outcome_str = "Có nguy cơ cao / Mắc bệnh tim" if prediction == 1 else "Nguy cơ thấp / Không mắc bệnh tim"
                            
                            prompt = f"""
Bạn là một bác sĩ chuyên khoa tim mạch giàu kinh nghiệm. Hãy phân tích các thông tin y khoa của bệnh nhân dưới đây và đưa ra những lời khuyên hữu ích, chế độ ăn uống, lối sống lành mạnh, và các khuyến nghị y khoa tiếp theo.

THÔNG TIN LÂM SÀNG CỦA BỆNH NHÂN:
- Tuổi: {inputs.get('age')}
- Giới tính: {gender_str}
- Loại đau ngực (cp): {cp_str}
- Huyết áp lúc nghỉ (trestbps): {inputs.get('trestbps')} mmHg
- Cholesterol huyết thanh (chol): {inputs.get('chol')} mg/dl
- Đường huyết lúc đói > 120 mg/dl (fbs): {fbs_str}
- Điện tâm đồ lúc nghỉ (restecg): {restecg_str}
- Nhịp tim tối đa đạt được (thalach): {inputs.get('thalach')} bpm
- Đau thắt ngực khi gắng sức (exang): {exang_str}
- Độ chênh lệch đoạn ST (oldpeak): {inputs.get('oldpeak')} mm
- Độ dốc đoạn ST (slope): {slope_str}
- Số mạch máu lớn nhuộm màu qua chụp huỳnh quang (ca): {inputs.get('ca')}
- Chỉ số Thalassemia (thal): {thal_str}

KẾT QUẢ CHẨN ĐOÁN CỦA HỆ THỐNG AI:
- Đánh giá sơ bộ: {outcome_str}
- Xác suất nguy cơ: {prob * 100:.1f}%
- Mô hình phân lớp: {model_name}

YÊU CẦU:
1. Đưa ra đánh giá sơ bộ về tình trạng sức khỏe tim mạch của bệnh nhân (kết hợp các thông số đặc trưng nhất như oldpeak, thalach, ca và thal).
2. Đề xuất chế độ ăn uống cụ thể (các thực phẩm nên ăn và nên tránh phù hợp với các chỉ số như cholesterol và huyết áp).
3. Đề xuất hoạt động thể chất và lối sống (cường độ vận động dựa trên nhịp tim tối đa thalach và exang).
4. Khuyến nghị các bước thăm khám hoặc kiểm tra chuyên khoa sâu hơn nếu cần thiết.

HÃY TRẢ LỜI BẰNG TIẾNG VIỆT thật chuyên nghiệp, ân cần, định dạng rõ ràng sử dụng Markdown (dùng các gạch đầu dòng, tô đậm từ khóa).
Lưu ý quan trọng: Cuối phản hồi phải ghi chú rõ rằng "Kết quả chẩn đoán và lời khuyên này chỉ mang tính chất tham khảo dựa trên mô hình AI, không thay thế cho việc thăm khám và chẩn đoán trực tiếp của bác sĩ chuyên khoa."
"""
                        else:
                            gender_str = "Male" if inputs.get("sex") == 1 else "Female"
                            cp_labels = {
                                0: "Typical angina",
                                1: "Atypical angina",
                                2: "Non-anginal pain",
                                3: "Asymptomatic"
                            }
                            cp_str = cp_labels.get(inputs.get("cp", 0), str(inputs.get("cp")))
                            fbs_str = "Yes (> 120 mg/dl)" if inputs.get("fbs") == 1 else "No (<= 120 mg/dl)"
                            restecg_labels = {
                                0: "Normal",
                                1: "ST-T wave abnormality",
                                2: "Left ventricular hypertrophy"
                            }
                            restecg_str = restecg_labels.get(inputs.get("restecg", 0), str(inputs.get("restecg")))
                            exang_str = "Yes" if inputs.get("exang") == 1 else "No"
                            slope_labels = {
                                0: "Upsloping",
                                1: "Flat",
                                2: "Downsloping"
                            }
                            slope_str = slope_labels.get(inputs.get("slope", 0), str(inputs.get("slope")))
                            thal_labels = {
                                0: "Normal",
                                1: "Fixed defect",
                                2: "Reversible defect"
                            }
                            thal_str = thal_labels.get(inputs.get("thal", 0), str(inputs.get("thal")))
                            
                            outcome_str = "At High Risk / Heart Disease Detected" if prediction == 1 else "Low Risk / No Heart Disease Detected"
                            
                            prompt = f"""
You are an experienced cardiologist. Please analyze the patient's clinical inputs below and provide personalized medical advice, dietary recommendations, lifestyle adjustments, and further clinical steps.

PATIENT CLINICAL DATA:
- Age: {inputs.get('age')}
- Sex: {gender_str}
- Chest Pain Type (cp): {cp_str}
- Resting Blood Pressure (trestbps): {inputs.get('trestbps')} mmHg
- Serum Cholesterol (chol): {inputs.get('chol')} mg/dl
- Fasting Blood Sugar > 120 mg/dl (fbs): {fbs_str}
- Resting ECG (restecg): {restecg_str}
- Max Heart Rate Achieved (thalach): {inputs.get('thalach')} bpm
- Exercise Induced Angina (exang): {exang_str}
- ST Depression (oldpeak): {inputs.get('oldpeak')} mm
- ST Slope (slope): {slope_str}
- Number of Major Vessels (ca): {inputs.get('ca')}
- Thalassemia (thal): {thal_str}

AI DIAGNOSIS RESULT:
- Preliminary Assessment: {outcome_str}
- Risk Probability: {prob * 100:.1f}%
- ML Model Classifer: {model_name}

INSTRUCTIONS:
1. Provide a brief assessment of the patient's cardiovascular state based on key metrics (especially oldpeak, thalach, ca, and thal).
2. Recommend specific dietary adjustments (foods to favor and foods to avoid based on cholesterol and blood pressure).
3. Recommend physical activity and lifestyle modifications (intensity of exercise based on thalach and exang).
4. Recommend next clinical check-ups or advanced tests if needed.

WRITE YOUR RESPONSE IN ENGLISH. Be professional, empathetic, and clear. Format your answer using Markdown (bullet points, bold keywords).
Crucial Note: End the response with: "This AI-generated advice and diagnosis are for reference only and do not replace a direct clinical diagnosis and check-up by a cardiologist."
"""
                        
                        response = model.generate_content(prompt)
                        advice_text = response.text
                        st.session_state["gemini_advice_cache"][cache_key] = advice_text
                        st.markdown(advice_text)
            except Exception as e:
                st.error(
                    f"Lỗi khi kết nối với Gemini API: {e}\nVui lòng kiểm tra lại API Key hoặc kết nối mạng."
                    if lang == "vi" else
                    f"Error connecting to Gemini API: {e}\nPlease check your API Key or network connection."
                )

    # ── Nút tải kết quả ──
    download_df = pd.DataFrame([{
        t("result_model", lang): model_name,
        t("result_risk", lang): risk,
        t("result_prob", lang): f"{prob * 100:.1f}%",
        t("result_header", lang): label,
    }])
    csv_bytes = download_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=t("diag_download_csv", lang),
        data=csv_bytes,
        file_name="diagnosis_result.csv",
        mime="text/csv",
    )