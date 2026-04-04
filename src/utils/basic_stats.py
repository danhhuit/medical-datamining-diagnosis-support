from pathlib import Path
from src.utils.load_data import load_heart_data


def run_basic_statistics(file_path):
    """In ra các thống kê cơ bản của bộ dữ liệu bệnh tim."""
    print("Đang nạp dữ liệu...")
    df = load_heart_data(file_path)

    print("\n" + "=" * 60)
    print("THỐNG KÊ CƠ BẢN DỮ LIỆU BỆNH TIM")
    print("=" * 60)

    print(f"\n1. Kích thước dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")

    print("\n2. Cấu trúc và kiểu dữ liệu:")
    df.info()

    print("\n3. Kiểm tra dữ liệu thiếu:")
    print(df.isnull().sum())

    print("\n4. Kiểm tra dữ liệu trùng:")
    print(f"Số dòng trùng: {df.duplicated().sum()}")

    print("\n5. Thống kê mô tả các cột số:")
    print(df.describe())

    print("\n6. Phân bố chẩn đoán bệnh (condition):")
    print(df["condition"].value_counts())

    print("\n7. Tỷ lệ phần trăm các lớp:")
    print((df["condition"].value_counts(normalize=True) * 100).round(2))

    print("=" * 60)


if __name__ == "__main__":
    file_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "heart.csv"
    run_basic_statistics(file_path)