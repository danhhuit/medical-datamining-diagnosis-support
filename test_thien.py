from src.evaluation.metrics import calculate_classification_metrics
from src.evaluation.evaluate_model import evaluate_model
from src.evaluation.plot_results import plot_roc_curve

# 1. Giả lập dữ liệu thực tế (y_true) và dữ liệu mô hình dự đoán (y_pred/y_prob)
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0, 1, 0]  # Nhãn dự đoán (0 hoặc 1)
y_prob = [0.1, 0.9, 0.6, 0.8, 0.4, 0.2, 0.7, 0.3] # Xác suất dự đoán (cho ROC curve)

print("--- BẮT ĐẦU TEST HỆ THỐNG ĐÁNH GIÁ ---")

# 2. Test tính toán chỉ số
metrics = calculate_classification_metrics(y_true, y_pred)
print(f"Kết quả Precision: {metrics['precision']}")

# 3. Test vẽ Ma trận nhầm lẫn và in báo cáo
evaluate_model(y_true, y_pred, model_name="Test_ZScore_Impact")

# 4. Test vẽ đường cong ROC
plot_roc_curve(y_true, y_prob, model_name="Test_ZScore_Impact")

print("--- KIỂM TRA THƯ MỤC OUTPUTS ĐỂ XEM KẾT QUẢ BIỂU ĐỒ ---")