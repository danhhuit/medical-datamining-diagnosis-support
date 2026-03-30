from __future__ import annotations

import json
import pandas as pd

from src.preprocessing.clean_data import load_raw_data, basic_data_check, clean_data
from src.preprocessing.transform_data import (
    rename_target_column,
    scale_continuous_features,
    save_preprocessor_artifact,
)
from src.preprocessing.feature_engineering import validate_final_dataset
from src.preprocessing.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    ensure_directories,
)

def run_preprocessing_pipeline() -> None:
    """
    Pipeline tiền xử lý hoàn chỉnh cho heart.csv.
    """
    print("=== BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU ===")
    ensure_directories()

    print(f"[1] Đọc dữ liệu gốc từ: {RAW_DATA_FILE}")
    df = load_raw_data(RAW_DATA_FILE)

    print("[2] Kiểm tra dữ liệu ban đầu...")
    summary = basic_data_check(df)
    print(json.dumps(summary, ensure_ascii=False, indent=4, default=str))

    print("[3] Làm sạch dữ liệu...")
    df_clean = clean_data(df)

    print("[4] Đổi tên cột target...")
    df_renamed = rename_target_column(df_clean)

    print("[5] Chuẩn hóa các cột liên tục...")
    df_processed, scaler = scale_continuous_features(df_renamed)

    print("[6] Kiểm tra bộ dữ liệu sau xử lý...")
    validate_final_dataset(df_processed, target_column="diagnosis")

    print(f"[7] Lưu dữ liệu đã xử lý vào: {PROCESSED_DATA_FILE}")
    df_processed.to_csv(PROCESSED_DATA_FILE, index=False)

    print("[8] Lưu preprocessor artifact...")
    preprocessor_path = save_preprocessor_artifact(scaler)

    print("=== TIỀN XỬ LÝ XONG ===")
    print(f"Dữ liệu processed: {PROCESSED_DATA_FILE}")
    print(f"Preprocessor artifact: {preprocessor_path}")

if __name__ == "__main__":
    run_preprocessing_pipeline()