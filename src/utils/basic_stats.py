# Import hàm đọc dữ liệu bạn đã viết ở file load_data.py
from load_data import load_heart_data


def run_basic_statistics(file_path):
    """Script in ra các thống kê cơ bản của bộ dữ liệu"""
    print("Đang nạp dữ liệu...")
    df = load_heart_data(file_path)

    if df is not None:
        print("\n" + "=" * 50)
        print("THỐNG KÊ CƠ BẢN DỮ LIỆU BỆNH TIM")
        print("=" * 50)

        print(f"\n1. Kích thước dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")

        print("\n2. Cấu trúc và kiểu dữ liệu:")
        df.info()

        print("\n3. Kiểm tra dữ liệu thiếu (Missing values):")
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            print("✅ Tuyệt vời! Bộ dữ liệu rất sạch, không có dòng nào bị thiếu.")
        else:
            print(df.isnull().sum()[df.isnull().sum() > 0])

        print("\n4. Phân bố chẩn đoán bệnh (Condition):")
        print(df['condition'].value_counts())
        print("=" * 50)


if __name__ == "__main__":
    # Đường dẫn từ thư mục src/utils/ ra data/raw/
    run_basic_statistics('../../data/raw/heart.csv')