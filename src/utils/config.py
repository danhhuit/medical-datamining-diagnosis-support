from pathlib import Path

# Thư mục gốc project
BASE_DIR = Path(__file__).resolve().parents[2]

# Data
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Outputs
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

# File dữ liệu đầu vào cho mô hình
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "heart_processed.csv"
ASSOCIATION_DATA_FILE = PROCESSED_DATA_DIR / "heart_association.csv"

# Cột target
TARGET_COLUMN = "diagnosis"

# Tham số train
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Ưu tiên chọn mô hình theo metric nào
PRIORITY_METRIC = "recall"   # có thể đổi thành "f1" hoặc "accuracy"

def ensure_directories() -> None:
    """Tạo thư mục output nếu chưa tồn tại."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)