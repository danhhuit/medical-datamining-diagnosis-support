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

def bin_and_encode_for_association(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rời rạc hóa các thuộc tính liên tục và ánh xạ các biến phân loại mã số
    về dạng chuỗi (labels) để có thể one-hot encode đầy đủ cho luật kết hợp.
    """
    df = df.copy()
    
    # 1. Rời rạc hóa các thuộc tính số liên tục và chuyển sang string
    df["age"] = pd.cut(df["age"], bins=[0, 45, 60, 120], labels=["age_young", "age_middle", "age_old"]).astype(str)
    df["trestbps"] = pd.cut(df["trestbps"], bins=[0, 120, 140, 300], labels=["bps_normal", "bps_prehypertension", "bps_high"]).astype(str)
    df["chol"] = pd.cut(df["chol"], bins=[0, 200, 240, 1000], labels=["chol_normal", "chol_borderline", "chol_high"]).astype(str)
    df["thalach"] = pd.cut(df["thalach"], bins=[0, 140, 160, 300], labels=["thalach_low", "thalach_medium", "thalach_high"]).astype(str)
    df["oldpeak"] = pd.cut(df["oldpeak"], bins=[-1.0, 1.0, 15.0], labels=["oldpeak_normal", "oldpeak_risk"]).astype(str)
    
    # 2. Ánh xạ các biến phân loại số về dạng chữ để pd.get_dummies phân biệt
    df["sex"] = df["sex"].map({0: "sex_female", 1: "sex_male"}).astype(str)
    df["cp"] = df["cp"].map({0: "cp_typical", 1: "cp_atypical", 2: "cp_non_anginal", 3: "cp_asymptomatic"}).astype(str)
    df["fbs"] = df["fbs"].map({0: "fbs_low", 1: "fbs_high"}).astype(str)
    df["restecg"] = df["restecg"].map({0: "restecg_normal", 1: "restecg_st_t_wave", 2: "restecg_hypertrophy"}).astype(str)
    df["exang"] = df["exang"].map({0: "exang_no", 1: "exang_yes"}).astype(str)
    df["slope"] = df["slope"].map({0: "slope_upsloping", 1: "slope_flat", 2: "slope_downsloping"}).astype(str)
    df["ca"] = df["ca"].map({0: "ca_0", 1: "ca_1", 2: "ca_2", 3: "ca_3", 4: "ca_4"}).astype(str)
    df["thal"] = df["thal"].map({0: "thal_normal", 1: "thal_fixed_defect", 2: "thal_reversible_defect", 3: "thal_unknown"}).astype(str)
    
    # Cột target
    df[TARGET_COLUMN_PROCESSED] = df[TARGET_COLUMN_PROCESSED].map({0: "no_heart_disease", 1: "heart_disease"}).astype(str)
    
    # Điền giá trị khuyết nếu có lỗi ánh xạ phát sinh ngoài ý muốn
    df = df.fillna("unknown")
    
    return df