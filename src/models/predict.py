from __future__ import annotations

from typing import Dict, Any

import joblib
import pandas as pd

from src.utils.config import MODEL_DIR


def load_model_artifact(model_file_name: str):
    """
    Load artifact model đã lưu.
    """
    model_path = MODEL_DIR / model_file_name
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    artifact = joblib.load(model_path)
    return artifact


def build_input_dataframe(input_data: Dict[str, Any], feature_names):
    """
    Chuyển dữ liệu đầu vào từ dict sang DataFrame đúng thứ tự cột.
    """
    missing_features = [col for col in feature_names if col not in input_data]
    if missing_features:
        raise ValueError(f"Thiếu các feature đầu vào: {missing_features}")

    row = {feature: input_data[feature] for feature in feature_names}
    return pd.DataFrame([row])


def predict_one(input_data: Dict[str, Any], model_file_name: str):
    """
    Dự đoán 1 mẫu dữ liệu.
    """
    artifact = load_model_artifact(model_file_name)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    model_name = artifact["model_name"]

    input_df = build_input_dataframe(input_data, feature_names)

    prediction = model.predict(input_df)[0]

    result = {
        "model_name": model_name,
        "prediction": int(prediction) if str(prediction).isdigit() else prediction
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0].tolist()
        result["probabilities"] = probabilities

    return result


if __name__ == "__main__":
    # Ví dụ test nhanh
    sample_input = {
        "age": 45,
        "fever": 1,
        "cough": 1,
        "blood_pressure": 130,
        "glucose": 92
    }

    # Thay đúng tên model đã lưu trong outputs/models
    result = predict_one(sample_input, "random_forest_best_model.pkl")
    print(result)