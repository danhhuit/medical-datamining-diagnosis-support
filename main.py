"""
Medical Datamining Diagnosis Support
=====================================
File chạy chính - điều phối toàn bộ pipeline của dự án.

Cách sử dụng:
    python main.py                  # Chạy toàn bộ pipeline
    python main.py --preprocess     # Chỉ chạy tiền xử lý
    python main.py --train          # Chỉ chạy huấn luyện mô hình
    python main.py --evaluate       # Chỉ chạy đánh giá mô hình
    python main.py --rules          # Chỉ chạy khai phá luật kết hợp
    python main.py --app            # Chạy ứng dụng demo Streamlit
"""
from __future__ import annotations

import sys
import os
import argparse

# Fix encoding cho Windows console (hỗ trợ tiếng Việt)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')


def run_preprocessing():
    """Chạy pipeline tiền xử lý dữ liệu."""
    from src.preprocessing.preprocessing_pipeline import run_preprocessing_pipeline
    run_preprocessing_pipeline()


def run_training():
    """Chạy pipeline huấn luyện mô hình."""
    from src.models.train_model import run_training_pipeline
    run_training_pipeline()


def run_evaluation():
    """Chạy đánh giá mô hình trên tập test."""
    from src.evaluation.evaluate_model import evaluate_model
    from src.models.predict import load_model_artifact
    from src.models.train_model import load_processed_data, split_data
    from src.utils.config import PROCESSED_DATA_FILE, TARGET_COLUMN

    print("=== ĐÁNH GIÁ MÔ HÌNH ===")

    # Load dữ liệu
    X, y = load_processed_data(PROCESSED_DATA_FILE, TARGET_COLUMN)
    _, X_test, _, y_test = split_data(X, y)

    # Load model tốt nhất
    from src.models.save_model import list_saved_models
    models = list_saved_models()
    if not models:
        print("Chưa có model nào. Hãy chạy --train trước.")
        return

    model_file = models[0]
    for m in models:
        if "logistic_regression" in m:
            model_file = m
            break

    artifact = load_model_artifact(model_file)
    model = artifact["model"]
    model_name = artifact["model_name"]

    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, model_name=model_name)

    # Vẽ ROC curve nếu model hỗ trợ predict_proba
    if hasattr(model, "predict_proba"):
        from src.evaluation.plot_results import plot_roc_curve
        y_prob = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_prob, model_name=model_name)

    print("=== ĐÁNH GIÁ XONG ===")


def run_association_rules():
    """Chạy khai phá luật kết hợp."""
    from src.association_rules.apriori_rules import run_apriori
    from src.association_rules.fp_growth_rules import run_fpgrowth
    from src.association_rules.export_rules import export_rules_to_csv
    from src.preprocessing.config import PROCESSED_DATA_FILE

    data_path = str(PROCESSED_DATA_FILE)

    print("=== KHAI PHÁ LUẬT KẾT HỢP ===")

    # Apriori
    print("\n--- Apriori ---")
    apriori_rules = run_apriori(data_path, min_support=0.15, min_confidence=0.7)
    if apriori_rules is not None:
        export_rules_to_csv(apriori_rules, 'apriori_rules.csv', top_k=100, min_lift=1.2)

    # FP-Growth
    print("\n--- FP-Growth ---")
    fpgrowth_rules = run_fpgrowth(data_path, min_support=0.15, min_confidence=0.7)
    if fpgrowth_rules is not None:
        export_rules_to_csv(fpgrowth_rules, 'fp_growth_rules.csv', top_k=100, min_lift=1.2)

    print("\n=== KHAI PHÁ LUẬT XONG ===")


def run_app():
    """Khởi chạy ứng dụng demo Streamlit."""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/app/app.py",
        "--server.headless", "true"
    ])


def run_full_pipeline():
    """Chạy toàn bộ pipeline từ tiền xử lý đến khai phá luật."""
    print("=" * 60)
    print(" MEDICAL DATAMINING DIAGNOSIS SUPPORT")
    print(" Chạy toàn bộ pipeline")
    print("=" * 60)

    print("\n\n>>> BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU")
    run_preprocessing()

    print("\n\n>>> BƯỚC 2: HUẤN LUYỆN MÔ HÌNH")
    run_training()

    print("\n\n>>> BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH")
    run_evaluation()

    print("\n\n>>> BƯỚC 4: KHAI PHÁ LUẬT KẾT HỢP")
    run_association_rules()

    print("\n" + "=" * 60)
    print(" PIPELINE HOÀN TẤT!")
    print("=" * 60)
    print("\nĐể chạy ứng dụng demo:")
    print("  streamlit run src/app/app.py")


def main():
    parser = argparse.ArgumentParser(
        description="Medical Datamining Diagnosis Support - Pipeline Manager"
    )
    parser.add_argument("--preprocess", action="store_true", help="Chỉ chạy tiền xử lý")
    parser.add_argument("--train", action="store_true", help="Chỉ chạy huấn luyện")
    parser.add_argument("--evaluate", action="store_true", help="Chỉ chạy đánh giá")
    parser.add_argument("--rules", action="store_true", help="Chỉ chạy khai phá luật")
    parser.add_argument("--app", action="store_true", help="Chạy ứng dụng demo")

    args = parser.parse_args()

    # Nếu không có tham số nào -> chạy toàn bộ pipeline
    if not any([args.preprocess, args.train, args.evaluate, args.rules, args.app]):
        run_full_pipeline()
        return

    if args.preprocess:
        run_preprocessing()
    if args.train:
        run_training()
    if args.evaluate:
        run_evaluation()
    if args.rules:
        run_association_rules()
    if args.app:
        run_app()


if __name__ == "__main__":
    main()