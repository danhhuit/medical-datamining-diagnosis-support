"""
Xuất luật kết hợp ra CSV/Excel và lọc các luật mạnh
để đưa vào báo cáo hoặc phần gợi ý.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.utils.config import BASE_DIR


RULES_DIR = BASE_DIR / "outputs" / "rules"


def export_rules_to_csv(rules_df, output_filename: str,
                        top_k: int | None = 100, min_lift: float = 1.2) -> None:
    """
    Xuất tập luật ra file CSV với các bộ lọc tối ưu cho y khoa.

    Parameters:
        rules_df: DataFrame chứa luật kết hợp từ Apriori/FP-Growth.
        output_filename: Tên file output (VD: 'apriori_rules.csv').
        top_k: Chỉ lấy top K luật mạnh nhất.
        min_lift: Ngưỡng Lift tối thiểu.
    """
    if rules_df is None or rules_df.empty:
        print("Không có luật nào để xuất.")
        return

    # TỐI ƯU 1: Chỉ lấy các luật có sức mạnh kết hợp thực sự (Lift >= min_lift)
    filtered_rules = rules_df[rules_df['lift'] >= min_lift].copy()

    if filtered_rules.empty:
        print(f"Không có luật nào đạt tiêu chuẩn Lift >= {min_lift}.")
        return

    # TỐI ƯU 2: Sắp xếp ưu tiên theo độ Nâng (Lift) và Độ tin cậy (Confidence)
    sorted_rules = filtered_rules.sort_values(
        by=['lift', 'confidence'], ascending=[False, False]
    )

    # TỐI ƯU 3: Giới hạn số lượng (Top K) để app Demo không bị treo
    if top_k is not None:
        sorted_rules = sorted_rules.head(top_k)

    # Đường dẫn lưu file chuẩn xác
    RULES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RULES_DIR / output_filename

    # Làm sạch giao diện text cho file CSV
    sorted_rules['antecedents'] = sorted_rules['antecedents'].apply(
        lambda x: ', '.join(list(x))
    )
    sorted_rules['consequents'] = sorted_rules['consequents'].apply(
        lambda x: ', '.join(list(x))
    )

    sorted_rules.to_csv(output_path, index=False)
    print(f"-> Đã tối ưu và xuất {len(sorted_rules)} luật MẠNH NHẤT ra file: {output_path}")


def export_rules_to_excel(rules_df, output_filename: str = "rules_report.xlsx",
                          top_k: int | None = 50, min_lift: float = 1.2) -> None:
    """
    Xuất luật kết hợp ra Excel (dùng cho báo cáo).
    """
    if rules_df is None or rules_df.empty:
        print("Không có luật nào để xuất.")
        return

    filtered_rules = rules_df[rules_df['lift'] >= min_lift].copy()
    if filtered_rules.empty:
        print(f"Không có luật nào đạt tiêu chuẩn Lift >= {min_lift}.")
        return

    sorted_rules = filtered_rules.sort_values(
        by=['lift', 'confidence'], ascending=[False, False]
    )

    if top_k is not None:
        sorted_rules = sorted_rules.head(top_k)

    RULES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RULES_DIR / output_filename

    sorted_rules['antecedents'] = sorted_rules['antecedents'].apply(
        lambda x: ', '.join(list(x))
    )
    sorted_rules['consequents'] = sorted_rules['consequents'].apply(
        lambda x: ', '.join(list(x))
    )

    sorted_rules.to_excel(output_path, index=False, engine='openpyxl')
    print(f"-> Đã xuất {len(sorted_rules)} luật ra Excel: {output_path}")