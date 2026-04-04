from __future__ import annotations

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.config import (
    CONTINUOUS_COLUMNS,
    TARGET_COLUMN_RAW,
    TARGET_COLUMN_PROCESSED,
    PREPROCESSOR_DIR,
)

def rename_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đổi tên cột target từ condition -> diagnosis
    để thống nhất với phần mô hình.
    """
    df = df.copy()
    if TARGET_COLUMN_RAW not in df.columns:
        raise ValueError(f"Không tìm thấy cột target gốc: {TARGET_COLUMN_RAW}")

    df = df.rename(columns={TARGET_COLUMN_RAW: TARGET_COLUMN_PROCESSED})
    return df

def scale_continuous_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Chuẩn hóa các cột liên tục bằng StandardScaler.
    Chỉ scale các cột:
    age, trestbps, chol, thalach, oldpeak

    Các cột mã hóa sẵn như sex, cp, fbs, restecg, exang, slope, ca, thal
    được giữ nguyên vì chúng đã là numeric code.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[CONTINUOUS_COLUMNS] = scaler.fit_transform(df[CONTINUOUS_COLUMNS])
    return df, scaler

def save_preprocessor_artifact(scaler: StandardScaler) -> str:
    """
    Lưu scaler để dùng lại khi demo/predict dữ liệu mới.
    """
    artifact = {
        "scaler": scaler,
        "continuous_columns": CONTINUOUS_COLUMNS,
        "renamed_target_column": TARGET_COLUMN_PROCESSED,
    }
    output_path = PREPROCESSOR_DIR / "heart_preprocessor.pkl"
    joblib.dump(artifact, output_path)
    return str(output_path)