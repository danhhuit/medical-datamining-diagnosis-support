"""
Các hàm hỗ trợ dùng chung cho toàn bộ project.
- Kiểm tra file tồn tại
- Tạo thư mục nếu chưa có
- Định dạng kết quả
- Ghi log đơn giản
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from datetime import datetime

from src.utils.config import BASE_DIR


# ---------- Logging ----------
def setup_logger(name: str = "medical_dm", level=logging.INFO) -> logging.Logger:
    """Tạo logger đơn giản cho project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------- File / Directory ----------
def ensure_dir(dir_path) -> Path:
    """Tạo thư mục nếu chưa tồn tại, trả về Path."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_exists(file_path) -> bool:
    """Kiểm tra file có tồn tại hay không."""
    return Path(file_path).is_file()


def get_project_path(*parts) -> Path:
    """Trả về đường dẫn tuyệt đối từ gốc project."""
    return BASE_DIR.joinpath(*parts)


# ---------- Formatting ----------
def format_metrics(metrics: dict, decimal: int = 4) -> dict:
    """Làm tròn các giá trị trong dict metrics."""
    return {k: round(v, decimal) if isinstance(v, float) else v for k, v in metrics.items()}


def dict_to_pretty_json(data: dict) -> str:
    """Chuyển dict thành chuỗi JSON đẹp."""
    return json.dumps(data, ensure_ascii=False, indent=4, default=str)


# ---------- Timestamp ----------
def get_timestamp() -> str:
    """Trả về chuỗi timestamp hiện tại."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------- Save / Load JSON ----------
def save_json(data: dict, file_path) -> None:
    """Lưu dict ra file JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)


def load_json(file_path) -> dict:
    """Đọc file JSON trả về dict."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
