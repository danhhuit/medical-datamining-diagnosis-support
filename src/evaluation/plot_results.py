import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_roc_curve(y_true, y_prob, model_name="Model"):
    """
    Vẽ đường cong ROC (Receiver Operating Characteristic)
    để đánh giá khả năng phân loại của mô hình ở các ngưỡng khác nhau.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính giả (False Positive Rate)')
    plt.ylabel('Tỷ lệ Dương tính thật (True Positive Rate)')
    plt.title(f'Đường cong ROC - {model_name}')
    plt.legend(loc="lower right")

    # Lưu biểu đồ
    output_path = f'outputs/roc_curve_{model_name}.png'
    plt.savefig(output_path)
    print(f"[OK] Đã lưu ROC Curve vào: {output_path}")
    plt.close()


def plot_preprocessing_impact(metrics_before, metrics_after,
                              metric_names=['Accuracy', 'Precision', 'Recall', 'F1-Score']):
    """
    Biểu đồ cột so sánh hiệu suất mô hình TRƯỚC và SAU khi
    thực hiện Tiền xử lý (Z-Score & Làm sạch dữ liệu).
    """
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, metrics_before, width, label='Trước khi xử lý (Raw Data)', color='lightcoral')
    rects2 = ax.bar(x + width / 2, metrics_after, width, label='Sau khi xử lý (Z-Score & Cleaned)',
                    color='mediumseagreen')

    ax.set_ylabel('Điểm số (0.0 - 1.0)')
    ax.set_title('So sánh hiệu suất mô hình: Đánh giá tác động của Tiền xử lý dữ liệu')
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
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    output_path = 'outputs/preprocessing_impact_comparison.png'
    plt.savefig(output_path)
    print(f"[OK] Đã lưu Biểu đồ so sánh vào: {output_path}")
    plt.close()


if __name__ == "__main__":
    # 1. Test hàm vẽ ROC
    # Lưu ý: y_prob là xác suất dự đoán (predict_proba), không phải nhãn 0/1 (predict)
    y_test_demo = [0, 1, 0, 1, 1, 0, 0, 1]
    y_prob_demo = [0.1, 0.8, 0.4, 0.9, 0.6, 0.2, 0.3, 0.7]
    plot_roc_curve(y_test_demo, y_prob_demo, "Demo_Model")

    # 2. Test hàm biểu đồ so sánh (rất hữu ích cho báo cáo đồ án của bạn)
    # Giả lập kết quả mô hình trước khi bạn làm chuẩn hóa Z-score
    scores_before = [0.75, 0.72, 0.78, 0.74]
    # Giả lập kết quả mô hình sau khi bạn chuẩn hóa và xóa dữ liệu nhiễu
    scores_after = [0.85, 0.84, 0.88, 0.85]

    plot_preprocessing_impact(scores_before, scores_after)