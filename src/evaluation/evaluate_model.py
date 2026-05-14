"""
Đánh giá toàn bộ mô hình trên tập kiểm tra,
tổng hợp chỉ số và xuất bảng kết quả.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.evaluation.metrics import calculate_classification_metrics, print_metrics_report
from src.utils.config import FIGURES_DIR, ensure_directories


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Hàm đánh giá mô hình dựa trên các chỉ số đo lường phổ biến.

    Bao gồm:
    - Tính Accuracy, Precision, Recall, F1-Score
    - In báo cáo chi tiết (classification report)
    - Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix)
    """
    ensure_directories()

    # 1. Tính toán các chỉ số cơ bản
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = calculate_classification_metrics(y_true, y_pred)
    print_metrics_report(metrics)

    print(f"--- KẾT QUẢ ĐÁNH GIÁ: {model_name} ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nChi tiết báo cáo phân lớp:")
    print(classification_report(y_true, y_pred))

    # 2. Vẽ Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Khong benh', 'Co benh'],
                yticklabels=['Khong benh', 'Co benh'])
    plt.xlabel('Du doan (Predicted)')
    plt.ylabel('Thuc te (Actual)')
    plt.title(f'Confusion Matrix - {model_name}')

    # Lưu biểu đồ vào thư mục outputs/figures
    output_path = FIGURES_DIR / f'confusion_matrix_{model_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Da luu ma tran nham lan vao: {output_path}")
    plt.close()

    return metrics


if __name__ == "__main__":
    # Ví dụ demo để kiểm tra file (Mock data)
    y_test_example = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred_example = [0, 1, 1, 1, 0, 0, 1, 0]

    evaluate_model(y_test_example, y_pred_example, "Demo_HeartDisease")
