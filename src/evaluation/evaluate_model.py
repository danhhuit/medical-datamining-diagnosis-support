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

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Hàm đánh giá mô hình dựa trên các chỉ số đo lường phổ biến.
    """
    # 1. Tính toán các chỉ số cơ bản
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

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
    # Giúp quan sát dữ liệu không nhất quán có gây ra sai sót dự đoán không
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Không bệnh', 'Có bệnh'],
                yticklabels=['Không bệnh', 'Có bệnh'])
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title(f'Ma trận nhầm lẫn - {model_name}')

    # Lưu biểu đồ vào thư mục outputs của đồ án
    plt.savefig(f'outputs/confusion_matrix_{model_name}.png')
    print(f"Đã lưu ma trận nhầm lẫn vào: outputs/confusion_matrix_{model_name}.png")
    plt.show()


if __name__ == "__main__":
    # Ví dụ demo để kiểm tra file (Mock data)
    # Khi chạy thực tế, bạn sẽ import hàm evaluate_model vào file chính (main.py)
    y_test_example = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred_example = [0, 1, 1, 1, 0, 0, 1, 0]

    evaluate_model(y_test_example, y_pred_example, "Demo_HeartDisease")

