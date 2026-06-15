from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from src.utils.config import (
    RAW_DATA_DIR,
    MODEL_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
)

def run_regression_pipeline() -> None:
    print("=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH HỒI QUY (REGRESSION) ===")
    
    # 1. Đọc dữ liệu gốc để giữ nguyên đơn vị y khoa
    raw_data_file = RAW_DATA_DIR / "heart.csv"
    if not raw_data_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc tại: {raw_data_file}")
        
    df = pd.read_csv(raw_data_file)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Đặt bài toán: Dự đoán nhịp tim tối đa (thalach) từ các thuộc tính khác
    # Cột dự đoán (target): thalach
    target_col = "thalach"
    
    # Các đặc trưng liên tục
    continuous_features = ["age", "trestbps", "chol", "oldpeak"]
    
    X_multivariate = df[continuous_features]
    y = df[target_col]
    
    X_simple = df[["age"]]
    
    # Chia dữ liệu train/test (80/20)
    X_train_s, X_test_s, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train_m, X_test_m, _, _ = train_test_split(X_multivariate, y, test_size=0.2, random_state=RANDOM_STATE)
    
    results = []
    
    # 2. Hồi quy tuyến tính đơn biến (Simple Linear Regression)
    print("Huấn luyện Hồi quy tuyến tính đơn biến (Tuổi -> Nhịp tim tối đa)...")
    lr_simple = LinearRegression()
    lr_simple.fit(X_train_s, y_train)
    y_pred_s = lr_simple.predict(X_test_s)
    
    mse_s = mean_squared_error(y_test, y_pred_s)
    r2_s = r2_score(y_test, y_pred_s)
    results.append({
        "model_name": "Simple Linear Regression",
        "features": "age",
        "MSE": float(mse_s),
        "R2_score": float(r2_s)
    })
    
    # 3. Hồi quy đa biến (Multivariate Linear Regression)
    print("Huấn luyện Hồi quy đa biến (Tuổi, Huyết áp, Cholesterol, ST depression -> Nhịp tim tối đa)...")
    lr_multi = LinearRegression()
    lr_multi.fit(X_train_m, y_train)
    y_pred_m = lr_multi.predict(X_test_m)
    
    mse_m = mean_squared_error(y_test, y_pred_m)
    r2_m = r2_score(y_test, y_pred_m)
    results.append({
        "model_name": "Multivariate Linear Regression",
        "features": ", ".join(continuous_features),
        "MSE": float(mse_m),
        "R2_score": float(r2_m)
    })
    
    # 4. Hồi quy đa thức (Polynomial Regression - Degree 2, đơn biến)
    print("Huấn luyện Hồi quy đa thức bậc 2 (Tuổi^2 -> Nhịp tim tối đa)...")
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train_s)
    X_test_poly = poly_features.transform(X_test_s)
    
    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train)
    y_pred_p = lr_poly.predict(X_test_poly)
    
    mse_p = mean_squared_error(y_test, y_pred_p)
    r2_p = r2_score(y_test, y_pred_p)
    results.append({
        "model_name": "Polynomial Regression (Deg 2)",
        "features": "age, age^2",
        "MSE": float(mse_p),
        "R2_score": float(r2_p)
    })
    
    # 5. Xuất và lưu metrics so sánh
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results)
    comparison_path = METRICS_DIR / "regression_comparison.csv"
    df_results.to_csv(comparison_path, index=False)
    print(f"--> Đã lưu bảng so sánh hồi quy vào: {comparison_path}")
    
    # 6. Vẽ và lưu biểu đồ đường khớp hồi quy (đối với biến age)
    plt.figure(figsize=(10, 6))
    
    # Vẽ các điểm dữ liệu thực tế tập test
    plt.scatter(X_test_s["age"], y_test, color="black", alpha=0.6, label="Thực tế (Test Set)", edgecolors="w")
    
    # Sắp xếp age để vẽ đường khớp trơn tru
    sort_idx = np.argsort(X_test_s["age"].values)
    sorted_age = X_test_s["age"].values[sort_idx]
    
    # Đường hồi quy đơn biến
    plt.plot(sorted_age, y_pred_s[sort_idx], color="red", linewidth=2, linestyle="-", label="Tuyến tính đơn biến")
    
    # Đường hồi quy đa thức bậc 2
    plt.plot(sorted_age, y_pred_p[sort_idx], color="blue", linewidth=2, linestyle="--", label="Đa thức bậc 2")
    
    plt.title("So sánh các mô hình Hồi quy dự đoán Nhịp tim tối đa (thalach) theo Tuổi (age)", fontsize=11, fontweight="bold")
    plt.xlabel("Tuổi (age)")
    plt.ylabel("Nhịp tim tối đa (thalach)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plot_path = FIGURES_DIR / "regression_fit.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> Đã lưu biểu đồ khớp hồi quy vào: {plot_path}")
    
    # 7. Lưu trữ model pkl
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_simple, MODEL_DIR / "regression_simple.pkl")
    joblib.dump(lr_multi, MODEL_DIR / "regression_multivariate.pkl")
    joblib.dump({"model": lr_poly, "poly_transformer": poly_features}, MODEL_DIR / "regression_poly.pkl")
    print("--> Đã lưu các artifact mô hình hồi quy vào thư mục outputs/models/")
    print("=== HỒI QUY XONG ===")

if __name__ == "__main__":
    import sys
    # Fix encoding cho Windows console
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
        
    run_regression_pipeline()
