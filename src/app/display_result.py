"""
Trình bày kết quả chẩn đoán dạng bảng, biểu đồ, gợi ý điều trị và nút tải file.
"""
from __future__ import annotations

import io
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from src.app.translations import t


def show_result(result: Dict[str, Any], lang: str = "vi") -> None:
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

    # ── Biểu đồ phân phối xác suất ──
    if probabilities and len(probabilities) == 2:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Bar(
            x=[t("result_chart_no", lang), t("result_chart_yes", lang)],
            y=[probabilities[0] * 100, probabilities[1] * 100],
            marker_color=["#27ae60", "#c0392b"],
            text=[f"{probabilities[0]*100:.1f}%", f"{probabilities[1]*100:.1f}%"],
            textposition='auto', width=0.5,
        )])
        fig.update_layout(
            title=dict(text=t("result_chart_title", lang), font=dict(size=14)),
            yaxis_title=t("result_chart_y", lang),
            yaxis=dict(range=[0, 100]),
            height=280,
            margin=dict(t=40, b=20, l=50, r=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

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

    # ── Gợi ý tham khảo ──
    with st.expander(t("result_suggest_title", lang), expanded=True):
        if prediction == 1:
            st.warning(t("result_warn_positive", lang))
        else:
            st.info(t("result_warn_negative", lang))

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