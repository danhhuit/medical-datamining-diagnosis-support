import os
import pandas as pd


def export_rules_to_csv(rules_df, output_filename, top_k=100, min_lift=1.2):
    """
    Xuất tập luật ra file CSV với các bộ lọc tối ưu cho y khoa.
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
    sorted_rules = filtered_rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])

    # TỐI ƯU 3: Giới hạn số lượng (Top K) để app Demo không bị treo
    if top_k is not None:
        sorted_rules = sorted_rules.head(top_k)

    # Đường dẫn lưu file chuẩn xác
    output_dir = 'outputs/rules/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Làm sạch giao diện text cho file CSV
    sorted_rules['antecedents'] = sorted_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    sorted_rules['consequents'] = sorted_rules['consequents'].apply(lambda x: ', '.join(list(x)))

    sorted_rules.to_csv(output_path, index=False)
    print(f"-> Đã tối ưu và xuất {len(sorted_rules)} luật MẠNH NHẤT ra file: {output_path}")