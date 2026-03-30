from __future__ import annotations

from typing import Dict, Any

import joblib
import pandas as pd

from src.utils.config import MODEL_DIR


def load_model_artifact(model_file_name: str) -> Dict[str, Any]:
    """
    Load model artifact đã lưu từ thư mục outputs/models.
    Artifact dự kiến chứa:
    - model
    - feature_names
    - target_column
    - model_name
    """
    model_path = MODEL_DIR / model_file_name
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    artifact = joblib.load(model_path)

    required_keys = ["model", "feature_names", "target_column", "model_name"]
    missing_keys = [key for key in required_keys if key not in artifact]
    if missing_keys:
        raise ValueError(f"Artifact model thiếu các key bắt buộc: {missing_keys}")

    return artifact


def build_input_dataframe(input_data: Dict[str, Any], feature_names) -> pd.DataFrame:
    """
    Chuyển dữ liệu đầu vào từ dict sang DataFrame đúng thứ tự cột của model.
    Kiểm tra:
    - thiếu feature
    - thừa feature
    - ép kiểu numeric
    """
    missing_features = [col for col in feature_names if col not in input_data]
    extra_features = [col for col in input_data if col not in feature_names]

    if missing_features:
        raise ValueError(f"Thiếu các feature đầu vào: {missing_features}")

    if extra_features:
        raise ValueError(f"Có feature không hợp lệ hoặc không cần thiết: {extra_features}")

    row = {feature: input_data[feature] for feature in feature_names}
    input_df = pd.DataFrame([row])

    # Ép toàn bộ về numeric để đồng bộ với dữ liệu đã train
    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="raise")

    return input_df


def decode_prediction(prediction_value: Any) -> str:
    """
    Chuyển giá trị dự đoán thành nhãn dễ hiểu.
    Với dataset heart:
    - 0: Không mắc bệnh tim
    - 1: Có nguy cơ / mắc bệnh tim
    """
    if prediction_value == 0:
        return "Không mắc bệnh tim"
    if prediction_value == 1:
        return "Có nguy cơ / mắc bệnh tim"
    return str(prediction_value)


def predict_one(input_data: Dict[str, Any], model_file_name: str) -> Dict[str, Any]:
    """
    Dự đoán cho 1 mẫu dữ liệu đầu vào.
    """
    artifact = load_model_artifact(model_file_name)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    model_name = artifact["model_name"]

    input_df = build_input_dataframe(input_data, feature_names)

    prediction = model.predict(input_df)[0]

    result = {
        "model_name": model_name,
        "prediction": int(prediction) if str(prediction).isdigit() else prediction,
        "predicted_label": decode_prediction(prediction),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0].tolist()
        result["probabilities"] = probabilities

        # Nếu là bài toán nhị phân, lấy luôn xác suất lớp 1
        if len(probabilities) == 2:
            result["positive_class_probability"] = probabilities[1]

    return result


if __name__ == "__main__":
    # Sample input đúng theo bộ dữ liệu heart.csv / heart_processed.csv
    # Các cột đầu vào thường gồm:
    # age, sex, cp, trestbps, chol, fbs, restecg,
    # thalach, exang, oldpeak, slope, ca, thal

    sample_input = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }

    # Thay đúng tên model đã lưu trong outputs/models
    result = predict_one(sample_input, "logistic_regression_best_model.pkl")
    print("Kết quả dự đoán:")
    print(result)