import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from export_rules import export_rules_to_csv


def run_apriori(data_path, min_support=0.15, min_confidence=0.7):
    print("Đang chạy Apriori với min_support={} và min_confidence={}...".format(min_support, min_confidence))
    df = pd.read_csv(data_path)

    df_encoded = pd.get_dummies(df)
    df_boolean = df_encoded.astype(bool)

    frequent_itemsets = apriori(df_boolean, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        print("Không tìm thấy tập phổ biến. Thử giảm min_support.")
        return None

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules


if __name__ == "__main__":
    data_path = 'data/processed/heart_processed.csv'

    # Đã nâng thông số đầu vào để quét chặt hơn
    rules = run_apriori(data_path, min_support=0.15, min_confidence=0.7)

    if rules is not None:
        print(f"-> Tìm được {len(rules)} luật thô.")
        # Xuất file với Top 100 và Lift tối thiểu 1.2
        export_rules_to_csv(rules, 'apriori_rules.csv', top_k=100, min_lift=1.2)