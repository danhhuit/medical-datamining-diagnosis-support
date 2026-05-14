"""
Ứng dụng demo Streamlit – Hỗ trợ chẩn đoán bệnh tim.
Chạy: streamlit run src/app/app.py
"""
from __future__ import annotations

import io
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
        submitted, user_data = input_form(lang)

    with col2:
        st.markdown(f"#### {t('diag_result_title', lang)}")
        if submitted:
            with st.spinner(t("diag_analyzing", lang)):
                try:
                    result = predict_one(user_data, model_file)
                    show_result(result, lang)
                except Exception as e:
                    st.error(f"{t('diag_error', lang)}: {e}")
        else:
            st.info(t("diag_placeholder", lang))


# ════════════════════════════════════════════
# TRANG 2: LUẬT KẾT HỢP
# ════════════════════════════════════════════
def page_rules():
    lang = get_lang()
    st.markdown(f"#### {t('rules_title', lang)}")

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


# ════════════════════════════════════════════
# TRANG 3: SO SÁNH MÔ HÌNH
# ════════════════════════════════════════════
def page_compare():
    lang = get_lang()
    st.markdown(f"#### {t('compare_title', lang)}")

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

    # Biểu đồ
    import plotly.graph_objects as go
    cols = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in df.columns]
    if cols:
        colors = ["#2c3e50", "#27ae60", "#e67e22", "#c0392b"]
        fig = go.Figure()
        for i, c in enumerate(cols):
            fig.add_trace(go.Bar(
                name=c.replace("_", " ").title(), x=df["model_name"], y=df[c],
                text=[f"{v:.3f}" for v in df[c]], textposition="auto",
                marker_color=colors[i % 4]
            ))
        fig.update_layout(
            title=dict(text=t("compare_chart_title", lang), font=dict(size=14)),
            barmode="group", yaxis=dict(range=[0, 1.05]), height=380,
            margin=dict(t=40, b=40, l=50, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Nút tải biểu đồ
        img_bytes = fig.to_image(format="png", width=1200, height=500, scale=2)
        st.download_button(t("compare_download_chart", lang), img_bytes, "model_comparison.png", "image/png")

    # Best model
    best_file = PROJECT_ROOT / "outputs" / "metrics" / "best_model_info.json"
    if best_file.exists():
        info = json.loads(best_file.read_text(encoding="utf-8"))
        st.success(f"{t('compare_best', lang)}: **{info.get('model_name')}** "
                   f"(Accuracy: {info.get('accuracy', 0):.2%}, Recall: {info.get('recall', 0):.2%})")

    # Biểu đồ đánh giá
    figs_dir = PROJECT_ROOT / "outputs" / "figures"
    if figs_dir.exists():
        imgs = sorted(figs_dir.glob("*.png"))
        if imgs:
            st.markdown(f"---\n**{t('compare_figures', lang)}**")
            img_cols = st.columns(min(len(imgs), 2))
            for i, p in enumerate(imgs[:4]):
                with img_cols[i % 2]:
                    st.image(str(p), caption=p.stem, use_container_width=True)


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

    nav_keys = ["nav_diagnosis", "nav_rules", "nav_compare", "nav_guide", "nav_about"]
    nav_labels = [t(k, lang) for k in nav_keys]
    menu = st.sidebar.radio(t("nav_label", lang), nav_labels)

    st.sidebar.markdown("---")
    st.sidebar.caption("Version 1.0")

    # Routing
    idx = nav_labels.index(menu)
    [page_diagnosis, page_rules, page_compare, page_guide, page_about][idx]()

    st.markdown("---")
    st.markdown(f'<div class="footer">{t("footer", lang)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()