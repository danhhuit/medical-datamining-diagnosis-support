"""
NOTEBOOK: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH (Modeling)
=====================================================
Bộ dữ liệu chẩn đoán bệnh tim

Notebook này minh họa toàn bộ quy trình huấn luyện và đánh giá mô hình:
1. Đọc dữ liệu đã tiền xử lý
2. Chia tập train/test
3. Huấn luyện nhiều mô hình (Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes)
4. So sánh metrics (Accuracy, Precision, Recall, F1-Score)
5. Chọn mô hình tốt nhất
6. Vẽ biểu đồ Confusion Matrix và ROC Curve
7. Vẽ biểu đồ so sánh mô hình
8. Lưu model và metrics

Chạy: python notebooks/modeling/modeling_notebook.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix encoding cho Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay
)

from src.utils.config import (
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    PRIORITY_METRIC,
    MODEL_DIR,
    METRICS_DIR,
    ensure_directories,
)
from src.models.train_model import load_processed_data, validate_training_data, split_data, save_best_model, save_metrics
from src.models.model_selector import compare_models, select_best_model

# Thư mục lưu biểu đồ
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "modeling"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════
# 1. Đọc dữ liệu
# ════════════════════════════════════════
print("=" * 60)
print(" HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH")
print("=" * 60)

ensure_directories()

print(f"\n[1] Đọc dữ liệu từ: {PROCESSED_DATA_FILE}")
X, y = load_processed_data(PROCESSED_DATA_FILE, TARGET_COLUMN)
print(f"    Features: {X.shape[1]} cột")
print(f"    Samples: {X.shape[0]} dòng")
print(f"    Target distribution:\n{y.value_counts().to_string()}")


# ════════════════════════════════════════
# 2. Kiểm tra dữ liệu
# ════════════════════════════════════════
print(f"\n[2] Kiểm tra dữ liệu...")
validate_training_data(X, y)
print("    ✓ Dữ liệu hợp lệ, sẵn sàng train.")


# ════════════════════════════════════════
# 3. Chia train/test
# ════════════════════════════════════════
print(f"\n[3] Chia train/test (test_size={TEST_SIZE}, stratified)...")
X_train, X_test, y_train, y_test = split_data(X, y)
print(f"    Train: {X_train.shape[0]} samples")
print(f"    Test:  {X_test.shape[0]} samples")


# ════════════════════════════════════════
# 4. So sánh nhiều mô hình
# ════════════════════════════════════════
print(f"\n[4] So sánh nhiều mô hình...")
results_df, trained_models = compare_models(
    X_train=X_train, y_train=y_train,
    X_valid=X_test, y_valid=y_test,
    random_state=RANDOM_STATE
)
print("\n    Bảng so sánh:")
print(results_df.to_string(index=False))


# ════════════════════════════════════════
# 5. Chọn mô hình tốt nhất
# ════════════════════════════════════════
print(f"\n[5] Chọn mô hình tốt nhất (metric ưu tiên: {PRIORITY_METRIC})...")
best_model_name, best_model, best_info = select_best_model(
    results_df=results_df,
    trained_models=trained_models,
    priority_metric=PRIORITY_METRIC
)
print(f"    --> Mô hình tốt nhất: {best_model_name}")
print(f"    --> Metrics: {json.dumps(best_info, indent=4, default=str)}")


# ════════════════════════════════════════
# 6. Vẽ biểu đồ Confusion Matrix
# ════════════════════════════════════════
print(f"\n[6] Vẽ biểu đồ đánh giá...")

for model_name, model in trained_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Không bệnh", "Có bệnh"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    ✓ {path.name}")

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="#2c3e50", lw=2, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – {model_name}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / f"roc_curve_{model_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"    ✓ {path.name}")


# ════════════════════════════════════════
# 7. Biểu đồ so sánh mô hình
# ════════════════════════════════════════
print(f"\n[7] Vẽ biểu đồ so sánh mô hình...")
metric_cols = ["accuracy", "precision", "recall", "f1_score"]
colors = ["#2c3e50", "#27ae60", "#e67e22", "#c0392b"]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.2

for i, col in enumerate(metric_cols):
    bars = ax.bar(x + i * width, results_df[col], width, label=col.replace("_", " ").title(), color=colors[i])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel("Mô hình")
ax.set_ylabel("Giá trị")
ax.set_title("So sánh hiệu suất các mô hình phân loại")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df["model_name"], rotation=20, ha="right")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path = FIGURES_DIR / "model_comparison_barplot.png"
plt.savefig(path, dpi=150)
plt.close()
print(f"    ✓ {path.name}")


# ════════════════════════════════════════
# 8. Lưu model và metrics
# ════════════════════════════════════════
print(f"\n[8] Lưu model và metrics...")
saved_model_path = save_best_model(
    model=best_model, feature_names=X.columns,
    best_model_name=best_model_name
)
save_metrics(results_df, best_info)

print(f"\n{'=' * 60}")
print(" HUẤN LUYỆN VÀ ĐÁNH GIÁ HOÀN TẤT!")
print(f"{'=' * 60}")
print(f"  Model: {saved_model_path}")
print(f"  Metrics: {METRICS_DIR / 'model_comparison.csv'}")
print(f"  Figures: {FIGURES_DIR}")
