"""
Vẽ biểu đồ đánh giá mô hình:
- ROC Curve
- Biểu đồ so sánh trước/sau tiền xử lý
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from src.utils.config import FIGURES_DIR, ensure_directories


def plot_roc_curve(y_true, y_prob, model_name="Model"):
    """
    Vẽ đường cong ROC (Receiver Operating Characteristic)
    để đánh giá khả năng phân loại của mô hình ở các ngưỡng khác nhau.
    """
    ensure_directories()

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")

    # Lưu biểu đồ
    output_path = FIGURES_DIR / f'roc_curve_{model_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Da luu ROC Curve vao: {output_path}")
    plt.close()


def plot_preprocessing_impact(metrics_before, metrics_after,
                              metric_names=None):
    """
    Biểu đồ cột so sánh hiệu suất mô hình TRƯỚC và SAU khi
    thực hiện Tiền xử lý (Z-Score & Làm sạch dữ liệu).
    """
    if metric_names is None:
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    ensure_directories()

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, metrics_before, width,
                    label='Before preprocessing (Raw Data)', color='lightcoral')
    rects2 = ax.bar(x + width / 2, metrics_after, width,
                    label='After preprocessing (Z-Score & Cleaned)',
                    color='mediumseagreen')

    ax.set_ylabel('Score (0.0 - 1.0)')
    ax.set_title('Model Performance: Impact of Data Preprocessing')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right')

    # Hiển thị giá trị trên từng cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    output_path = FIGURES_DIR / 'preprocessing_impact_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Da luu bieu do so sanh vao: {output_path}")
    plt.close()


def plot_model_comparison(results_df):
    """
    Vẽ biểu đồ so sánh tổng hợp nhiều mô hình.
    """
    ensure_directories()

    metrics_cols = ["accuracy", "precision", "recall", "f1_score"]
    available_cols = [c for c in metrics_cols if c in results_df.columns]

    if not available_cols:
        print("Khong co metrics nao de ve.")
        return

    x = np.arange(len(results_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    for i, col in enumerate(available_cols):
        ax.bar(x + i * width, results_df[col], width,
               label=col.replace("_", " ").title(), color=colors[i % len(colors)])

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * (len(available_cols) - 1) / 2)
    ax.set_xticklabels(results_df["model_name"], rotation=15)
    ax.set_ylim([0, 1.1])
    ax.legend()

    fig.tight_layout()
    output_path = FIGURES_DIR / 'model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Da luu bieu do so sanh mo hinh vao: {output_path}")
    plt.close()


if __name__ == "__main__":
    # 1. Test hàm vẽ ROC
    y_test_demo = [0, 1, 0, 1, 1, 0, 0, 1]
    y_prob_demo = [0.1, 0.8, 0.4, 0.9, 0.6, 0.2, 0.3, 0.7]
    plot_roc_curve(y_test_demo, y_prob_demo, "Demo_Model")

    # 2. Test hàm biểu đồ so sánh
    scores_before = [0.75, 0.72, 0.78, 0.74]
    scores_after = [0.85, 0.84, 0.88, 0.85]
    plot_preprocessing_impact(scores_before, scores_after)