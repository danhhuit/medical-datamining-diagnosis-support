"""
NOTEBOOK: TIỀN XỬ LÝ DỮ LIỆU (Preprocessing)
================================================
Bộ dữ liệu chẩn đoán bệnh tim (heart.csv)

Notebook này minh họa toàn bộ quy trình tiền xử lý dữ liệu:
1. Đọc dữ liệu gốc
2. Kiểm tra chất lượng dữ liệu
3. Làm sạch dữ liệu (xử lý missing values, duplicates)
4. Đổi tên cột target
5. Chuẩn hóa các cột liên tục (StandardScaler)
6. Kiểm tra bộ dữ liệu sau xử lý
7. Lưu kết quả

Chạy: python notebooks/preprocessing/preprocessing_notebook.py
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

# ── Import các module tiền xử lý ──
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


# ════════════════════════════════════════
# 1. Đọc dữ liệu gốc
# ════════════════════════════════════════
print("=" * 60)
print(" TIỀN XỬ LÝ DỮ LIỆU – HEART DISEASE DATASET")
print("=" * 60)

ensure_directories()

print(f"\n[1] Đọc dữ liệu gốc từ: {RAW_DATA_FILE}")
df = load_raw_data(RAW_DATA_FILE)
print(f"    Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột")
print(f"    Các cột: {df.columns.tolist()}")
print(f"\n    5 dòng đầu:")
print(df.head())


# ════════════════════════════════════════
# 2. Kiểm tra chất lượng dữ liệu
# ════════════════════════════════════════
print(f"\n[2] Kiểm tra dữ liệu ban đầu...")
summary = basic_data_check(df)
print(json.dumps(summary, ensure_ascii=False, indent=4, default=str))


# ════════════════════════════════════════
# 3. Làm sạch dữ liệu
# ════════════════════════════════════════
print(f"\n[3] Làm sạch dữ liệu...")
df_clean = clean_data(df)
print(f"    Kích thước sau khi làm sạch: {df_clean.shape}")
print(f"    Missing values: {df_clean.isnull().sum().sum()}")
print(f"    Duplicate rows: {df_clean.duplicated().sum()}")


# ════════════════════════════════════════
# 4. Đổi tên cột target
# ════════════════════════════════════════
print(f"\n[4] Đổi tên cột target (condition -> diagnosis)...")
df_renamed = rename_target_column(df_clean)
print(f"    Các cột sau đổi tên: {df_renamed.columns.tolist()}")


# ════════════════════════════════════════
# 5. Chuẩn hóa các cột liên tục
# ════════════════════════════════════════
print(f"\n[5] Chuẩn hóa các cột liên tục (StandardScaler)...")
df_processed, scaler = scale_continuous_features(df_renamed)
print(f"    Thống kê mô tả sau chuẩn hóa:")
print(df_processed.describe().T.to_string())


# ════════════════════════════════════════
# 6. Kiểm tra bộ dữ liệu sau xử lý
# ════════════════════════════════════════
print(f"\n[6] Kiểm tra bộ dữ liệu sau xử lý...")
validate_final_dataset(df_processed, target_column="diagnosis")


# ════════════════════════════════════════
# 7. Lưu kết quả
# ════════════════════════════════════════
print(f"\n[7] Lưu dữ liệu đã xử lý vào: {PROCESSED_DATA_FILE}")
df_processed.to_csv(PROCESSED_DATA_FILE, index=False)

print("[8] Lưu preprocessor artifact...")
preprocessor_path = save_preprocessor_artifact(scaler)

print(f"\n{'=' * 60}")
print(" TIỀN XỬ LÝ HOÀN TẤT!")
print(f"{'=' * 60}")
print(f"  Dữ liệu processed: {PROCESSED_DATA_FILE}")
print(f"  Preprocessor artifact: {preprocessor_path}")
