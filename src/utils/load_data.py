import pandas as pd
import os

def load_heart_data(file_path):
    """
    Hàm đọc dữ liệu bệnh tim (Heart Disease) từ file CSV.
    """
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file tại {file_path}. Vui lòng kiểm tra lại đường dẫn!")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"✅ Đọc dữ liệu thành công!")
        print(f"📊 Kích thước: {df.shape[0]} dòng (bệnh nhân) x {df.shape[1]} cột (thuộc tính).")
        return df
    except Exception as e:
        print(f"❌ Có lỗi xảy ra khi đọc file: {e}")
        return None


# Test thử khi chạy trực tiếp file này trên PyCharm
if __name__ == "__main__":
    # Đường dẫn tương đối từ thư mục src/utils/ ra data/raw/
    file_path = "../../data/raw/heart.csv"
    df = load_heart_data(file_path)