from pathlib import Path
import pandas as pd


def load_heart_data(file_path):
    """
    Đọc dữ liệu bệnh tim từ file CSV.
    Trả về DataFrame nếu thành công, raise lỗi nếu thất bại.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {file_path}")

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file CSV: {e}") from e


if __name__ == "__main__":
    file_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "heart.csv"
    df = load_heart_data(file_path)
    print("Đọc dữ liệu thành công!")
    print(f"Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột")