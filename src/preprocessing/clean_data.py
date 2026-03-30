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

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu cơ bản.
    Với file heart.csv hiện tại:
    - không có missing values
    - không có duplicate rows
    - toàn bộ cột đã ở dạng số
    => bước clean chủ yếu là kiểm tra, bỏ trùng nếu phát sinh.
    """
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    return df