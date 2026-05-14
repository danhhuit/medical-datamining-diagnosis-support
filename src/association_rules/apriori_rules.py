"""
Chạy thuật toán Apriori để tìm tập phổ biến và sinh luật kết hợp.
Dữ liệu đầu vào là file CSV đã xử lý.
"""
from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from src.association_rules.export_rules import export_rules_to_csv
from src.preprocessing.config import PROCESSED_DATA_FILE


def binarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển dữ liệu đã xử lý sang dạng boolean cho Apriori.
    Dùng get_dummies để mã hóa one-hot, rồi chuyển sang bool.
    """
    df_encoded = pd.get_dummies(df)
    df_boolean = df_encoded.astype(bool)
    return df_boolean


def run_apriori(data_path, min_support: float = 0.15, min_confidence: float = 0.7):
    """
    Chạy thuật toán Apriori:
    1. Đọc dữ liệu CSV
    2. Chuyển sang dạng boolean
    3. Tìm tập phổ biến (frequent itemsets)
    4. Sinh luật kết hợp (association rules)

    Returns: DataFrame chứa các luật, hoặc None nếu không tìm được.
    """
    print(f"Đang chạy Apriori với min_support={min_support} và min_confidence={min_confidence}...")
    df = pd.read_csv(data_path)

    df_boolean = binarize_data(df)

    frequent_itemsets = apriori(df_boolean, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        print("Không tìm thấy tập phổ biến. Thử giảm min_support.")
        return None

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    print(f"Tìm được {len(frequent_itemsets)} tập phổ biến.")
    print(f"Sinh được {len(rules)} luật kết hợp.")
    return rules


if __name__ == "__main__":
    data_path = str(PROCESSED_DATA_FILE)

    # Đã nâng thông số đầu vào để quét chặt hơn
    rules = run_apriori(data_path, min_support=0.15, min_confidence=0.7)

    if rules is not None:
        print(f"-> Tìm được {len(rules)} luật thô.")
        # Xuất file với Top 100 và Lift tối thiểu 1.2
        export_rules_to_csv(rules, 'apriori_rules.csv', top_k=100, min_lift=1.2)