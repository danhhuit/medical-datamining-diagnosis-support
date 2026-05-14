"""
Lưu hoặc nạp model bằng joblib nhằm phục vụ bước dự đoán và demo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib

from src.utils.config import MODEL_DIR, ensure_directories


def save_model(model, feature_names: List[str], target_column: str,
               model_name: str, extra_meta: Dict[str, Any] | None = None) -> str:
    """
    Lưu model artifact gồm:
    - model: object sklearn đã fit
    - feature_names: danh sách tên cột đầu vào
    - target_column: tên cột target
    - model_name: tên thuật toán
    - extra_meta: thông tin bổ sung nếu cần

    Returns: đường dẫn file đã lưu.
    """
    ensure_directories()

    artifact: Dict[str, Any] = {
        "model": model,
        "feature_names": list(feature_names),
        "target_column": target_column,
        "model_name": model_name,
    }

    if extra_meta:
        artifact.update(extra_meta)

    file_path = MODEL_DIR / f"{model_name}_best_model.pkl"
    joblib.dump(artifact, file_path)
    print(f"[OK] Model '{model_name}' đã lưu tại: {file_path}")
    return str(file_path)


def load_model(model_file_name: str) -> Dict[str, Any]:
    """
    Nạp model artifact từ thư mục outputs/models.

    Returns: dict chứa model, feature_names, target_column, model_name.
    """
    model_path = MODEL_DIR / model_file_name

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    artifact = joblib.load(model_path)

    required_keys = ["model", "feature_names", "target_column", "model_name"]
    missing_keys = [k for k in required_keys if k not in artifact]
    if missing_keys:
        raise ValueError(f"Artifact model thiếu các key: {missing_keys}")

    return artifact


def list_saved_models() -> List[str]:
    """Liệt kê tất cả file model đã lưu."""
    if not MODEL_DIR.exists():
        return []
    return [f.name for f in MODEL_DIR.glob("*.pkl")]


if __name__ == "__main__":
    print("Các model đã lưu:")
    for m in list_saved_models():
        print(f"  - {m}")
