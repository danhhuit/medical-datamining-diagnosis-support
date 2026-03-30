from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PREPROCESSOR_DIR = BASE_DIR / "outputs" / "preprocessors"

RAW_DATA_FILE = RAW_DATA_DIR / "heart.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "heart_processed.csv"

TARGET_COLUMN_RAW = "condition"
TARGET_COLUMN_PROCESSED = "diagnosis"

CONTINUOUS_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_CODED_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

def ensure_directories() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)