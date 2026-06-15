from __future__ import annotations

import json
from typing import Tuple, Dict

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

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
from src.models.model_selector import compare_models, select_best_model


def load_processed_data(file_path, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Đọc dữ liệu đã xử lý và tách X, y.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {file_path}")

    df = pd.read_csv(file_path)

    if target_column not in df.columns:
        raise ValueError(f"Cột target '{target_column}' không tồn tại trong dữ liệu.")

    if df.empty:
        raise ValueError("File dữ liệu rỗng.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def validate_training_data(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Kiểm tra dữ liệu trước khi train.
    """
    if X.isnull().sum().sum() > 0:
        raise ValueError("Dữ liệu đầu vào X vẫn còn missing values.")

    if y.isnull().sum() > 0:
        raise ValueError("Cột target vẫn còn missing values.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("Số dòng của X và y không khớp.")

    # Kiểm tra toàn bộ feature phải là số
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            f"Các cột sau chưa phải số, cần preprocessing lại: {non_numeric_cols}"
        )


def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Chia train/test cho bài toán phân lớp.
    """
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def save_best_model(model, feature_names, best_model_name: str) -> str:
    """
    Lưu model tốt nhất.
    """
    model_path = MODEL_DIR / f"{best_model_name}_best_model.pkl"

    artifact = {
        "model": model,
        "feature_names": list(feature_names),
        "target_column": TARGET_COLUMN,
        "model_name": best_model_name,
    }

    joblib.dump(artifact, model_path)
    return str(model_path)


def save_metrics(results_df: pd.DataFrame, best_info: Dict) -> None:
    """
    Lưu bảng so sánh model và thông tin best model.
    """
    metrics_csv = METRICS_DIR / "model_comparison.csv"
    best_json = METRICS_DIR / "best_model_info.json"

    results_df.to_csv(metrics_csv, index=False)

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=4)


def run_training_pipeline() -> None:
    """
    Pipeline train hoàn chỉnh.
    """
    print("=== BẮT ĐẦU TRAIN MÔ HÌNH CHẨN ĐOÁN BỆNH ===")
    ensure_directories()

    print(f"[1] Đọc dữ liệu từ: {PROCESSED_DATA_FILE}")
    X, y = load_processed_data(PROCESSED_DATA_FILE, TARGET_COLUMN)

    print("[2] Kiểm tra dữ liệu...")
    validate_training_data(X, y)

    print("[3] Chia train/test...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("[4] So sánh nhiều mô hình...")
    results_df, trained_models = compare_models(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_test,
        y_valid=y_test,
        random_state=RANDOM_STATE
    )

    print("[5] Chọn mô hình tốt nhất...")
    best_model_name, best_model, best_info = select_best_model(
        results_df=results_df,
        trained_models=trained_models,
        priority_metric=PRIORITY_METRIC
    )

    print(f"--> Mô hình tốt nhất: {best_model_name}")
    print(f"--> Thông tin best model: {best_info}")

    print("[6] Lưu tất cả model đã train...")
    for model_name, model in trained_models.items():
        save_best_model(
            model=model,
            feature_names=X.columns,
            best_model_name=model_name
        )
    print(f"    Đã lưu {len(trained_models)} model vào {MODEL_DIR}")

    print("[7] Lưu metrics...")
    save_metrics(results_df, best_info)

    print("=== TRAIN XONG ===")
    print(f"Bảng so sánh model lưu tại: {METRICS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    run_training_pipeline()