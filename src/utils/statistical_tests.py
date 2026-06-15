from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from scipy import stats

from src.utils.config import (
    RAW_DATA_DIR,
    METRICS_DIR,
)

def run_statistical_tests() -> None:
    print("=== BẮT ĐẦU KIỂM ĐỊNH THỐNG KÊ (ANOVA & CHI-SQUARE) ===")
    
    # 1. Đọc bộ dữ liệu gốc
    from src.preprocessing.config import RAW_DATA_FILE
    from src.preprocessing.clean_data import load_raw_data
    df = load_raw_data(RAW_DATA_FILE)
    df = df.drop_duplicates().reset_index(drop=True)
    
    target_col = "condition"
    if target_col not in df.columns:
        # Nếu đã đổi tên
        if "diagnosis" in df.columns:
            target_col = "diagnosis"
        else:
            raise KeyError(f"Không tìm thấy cột target (condition/diagnosis) trong dữ liệu.")
            
    # Tách nhóm bệnh (1) và không bệnh (0)
    group_no_disease = df[df[target_col] == 0]
    group_disease = df[df[target_col] == 1]
    
    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    results = {
        "anova": {},
        "chi_square": {}
    }
    
    # 2. Chạy ANOVA cho các biến liên tục
    print("Đang chạy kiểm định ANOVA...")
    for col in continuous_cols:
        if col in df.columns:
            f_val, p_val = stats.f_oneway(group_no_disease[col], group_disease[col])
            # Giải nghĩa kết quả
            is_significant = p_val < 0.05
            results["anova"][col] = {
                "f_statistic": float(f_val),
                "p_value": float(p_val),
                "is_significant": bool(is_significant),
                "interpretation": "Khác biệt có ý nghĩa thống kê" if is_significant else "Khác biệt không có ý nghĩa thống kê"
            }
            print(f"  -> ANOVA '{col}': F-stat={f_val:.4f}, p-val={p_val:.4g} ({results['anova'][col]['interpretation']})")

    # 3. Chạy Chi-Square cho các biến phân loại
    print("Đang chạy kiểm định Chi-Square...")
    for col in categorical_cols:
        if col in df.columns:
            # Tạo bảng chéo (contingency table)
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
            is_significant = p_val < 0.05
            results["chi_square"][col] = {
                "chi2_statistic": float(chi2),
                "p_value": float(p_val),
                "dof": int(dof),
                "is_significant": bool(is_significant),
                "interpretation": "Có liên quan có ý nghĩa thống kê với bệnh tim" if is_significant else "Độc lập thống kê với bệnh tim"
            }
            print(f"  -> Chi2 '{col}': Chi2-stat={chi2:.4f}, p-val={p_val:.4g} ({results['chi_square'][col]['interpretation']})")
            
    # 4. Xuất kết quả ra file JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "statistical_tests.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"--> Đã lưu kết quả kiểm định thống kê vào: {out_path}")
    print("=== KIỂM ĐỊNH THỐNG KÊ XONG ===")

if __name__ == "__main__":
    import sys
    # Fix encoding cho Windows console
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
        
    run_statistical_tests()
