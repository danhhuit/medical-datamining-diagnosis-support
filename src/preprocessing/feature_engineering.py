from __future__ import annotations

import pandas as pd

def validate_final_dataset(df: pd.DataFrame, target_column: str = "diagnosis") -> None:
    """
    Kiểm tra bộ dữ liệu sau tiền xử lý trước khi bàn giao cho module mô hình.
    Yêu cầu:
    - không còn missing values
    - target tồn tại
    - toàn bộ feature là số
    """
    if target_column not in df.columns:
        raise ValueError(f"Không tìm thấy cột target: {target_column}")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dữ liệu sau tiền xử lý vẫn còn missing values.")

    feature_df = df.drop(columns=[target_column])
    non_numeric_cols = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"Các cột sau chưa phải số: {non_numeric_cols}")

def split_features_target(df: pd.DataFrame, target_column: str = "diagnosis"):
    """
    Tách dữ liệu thành X, y để kiểm tra nhanh trước khi giao cho phần mô hình.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y