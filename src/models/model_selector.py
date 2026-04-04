from __future__ import annotations

from typing import Dict, Tuple, List
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_candidate_models(random_state: int = 42) -> Dict[str, object]:
    """
    Trả về danh sách các mô hình sẽ thử nghiệm.
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
    }


def _get_average_type(y_true) -> str:
    """
    Nếu bài toán 2 lớp -> binary
    Nếu nhiều lớp -> weighted
    """
    unique_classes = pd.Series(y_true).nunique()
    return "binary" if unique_classes == 2 else "weighted"


def evaluate_model(model, X_valid, y_valid) -> Dict[str, float]:
    """
    Tính các metric cơ bản cho một mô hình.
    """
    y_pred = model.predict(X_valid)
    average_type = _get_average_type(y_valid)

    results = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred, average=average_type, zero_division=0),
        "recall": recall_score(y_valid, y_pred, average=average_type, zero_division=0),
        "f1_score": f1_score(y_valid, y_pred, average=average_type, zero_division=0),
    }
    return results


def compare_models(
    X_train,
    y_train,
    X_valid,
    y_valid,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Train tất cả mô hình và trả về:
    - bảng kết quả metrics
    - dict chứa các model đã train
    """
    models = get_candidate_models(random_state=random_state)
    trained_models: Dict[str, object] = {}
    rows: List[Dict[str, float]] = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_valid, y_valid)

        rows.append({
            "model_name": model_name,
            **metrics
        })
        trained_models[model_name] = model

    results_df = pd.DataFrame(rows).sort_values(by="recall", ascending=False).reset_index(drop=True)
    return results_df, trained_models


def select_best_model(
    results_df: pd.DataFrame,
    trained_models: Dict[str, object],
    priority_metric: str = "recall"
):
    """
    Chọn mô hình tốt nhất theo metric ưu tiên.
    """
    if priority_metric not in results_df.columns:
        raise ValueError(f"Metric ưu tiên '{priority_metric}' không tồn tại trong bảng kết quả.")

    best_row = results_df.sort_values(by=priority_metric, ascending=False).iloc[0]
    best_model_name = best_row["model_name"]
    best_model = trained_models[best_model_name]

    return best_model_name, best_model, best_row.to_dict()