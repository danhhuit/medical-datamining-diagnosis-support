import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_classification_metrics(y_true, y_pred):
    """
    Tính toán bộ chỉ số đánh giá cơ bản cho bài toán phân loại bệnh tim.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def calculate_z_score_stats(data_column):
    """
    Hàm bổ trợ :
    Kiểm tra xem dữ liệu sau khi chuẩn hóa Z-score có đạt chuẩn (mean=0, std=1) chưa.
    """
    mean = np.mean(data_column)
    std = np.std(data_column)
    return {"mean": round(mean, 4), "std": round(std, 4)}

def print_metrics_report(metrics_dict):
    """
    In báo cáo các chỉ số ra màn hình console một cách đẹp mắt.
    """
    print("\n" + "="*30)
    print(" CHI SỐ ĐÁNH GIÁ MÔ HÌNH ")
    print("="*30)
    for key, value in metrics_dict.items():
        print(f"{key.capitalize():<12}: {value:.4f}")
    print("="*30)