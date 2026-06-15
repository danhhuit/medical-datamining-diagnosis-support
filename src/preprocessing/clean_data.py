from __future__ import annotations
import pandas as pd

def load_raw_data(file_path) -> pd.DataFrame:
    """
    Đọc dữ liệu gốc từ file CSV.
    """
    df = pd.read_csv(file_path)
    return df

def basic_data_check(df: pd.DataFrame) -> dict:
    """
    Kiểm tra tổng quan dữ liệu:
    - số dòng, số cột
    - missing values
    - số dòng trùng
    """
    summary = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return summary

from src.preprocessing.config import CONTINUOUS_COLUMNS, CATEGORICAL_CODED_COLUMNS

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu:
    - Loại bỏ dòng trùng lặp
    - Xử lý giá trị khuyết (missing values) bằng median/mode
    - Giới hạn ngoại lệ (outliers) bằng phương pháp IQR
    """
    df = df.copy()
    
    # 1. Loại bỏ dữ liệu trùng lặp
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Xử lý giá trị khuyết (nếu có)
    # Với biến liên tục: thay bằng median
    for col in CONTINUOUS_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # Với biến phân loại: thay bằng mode
    for col in CATEGORICAL_CODED_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else 0
            df[col] = df[col].fillna(fill_val)
            
    # 3. Xử lý dữ liệu ngoại lệ (Outliers) cho các thuộc tính số liên tục
    for col in CONTINUOUS_COLUMNS:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Giới hạn giá trị nằm ngoài biên [lower_bound, upper_bound]
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
    return df