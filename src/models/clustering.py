from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from src.utils.config import (
    PROCESSED_DATA_FILE,
    RAW_DATA_DIR,
    MODEL_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
)

def run_clustering_pipeline(n_clusters: int = 3) -> None:
    print("=== BẮT ĐẦU PHÂN CỤM BỆNH NHÂN (CLUSTERING) ===")
    
    # 1. Đọc dữ liệu đã xử lý
    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu processed: {PROCESSED_DATA_FILE}")
    
    df_processed = pd.read_csv(PROCESSED_DATA_FILE)
    
    # Tách X (loại bỏ target diagnosis)
    X = df_processed.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in df_processed.columns else df_processed
    
    # 2. Huấn luyện K-Means
    print(f"Đang chạy K-Means với n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Tính toán Silhouette Score
    sil_score = silhouette_score(X, cluster_labels)
    print(f"--> Silhouette Score: {sil_score:.4f}")
    
    # 3. Chạy PCA giảm chiều dữ liệu về 2D phục vụ trực quan hóa
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    df_pca["Cluster"] = cluster_labels
    
    # Vẽ và lưu biểu đồ phân cụm bằng matplotlib thuần
    plt.figure(figsize=(10, 7))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for cluster in sorted(df_pca["Cluster"].unique()):
        subset = df_pca[df_pca["Cluster"] == cluster]
        plt.scatter(
            subset["PCA1"], subset["PCA2"],
            label=f"Cluster {cluster}",
            color=colors[cluster % len(colors)],
            s=100, alpha=0.8, edgecolors='black', linewidths=0.5
        )
        
    plt.title(f"Phân cụm bệnh nhân bằng K-Means (Silhouette Score: {sil_score:.4f})", fontsize=12, fontweight="bold")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Các cụm")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    pca_plot_path = FIGURES_DIR / "clustering_pca.png"
    plt.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> Đã lưu biểu đồ PCA phân cụm vào: {pca_plot_path}")
    
    # 4. Phân tích cụm dựa trên dữ liệu gốc (cho dễ diễn giải y học)
    # Đọc bộ dữ liệu gốc
    from src.preprocessing.config import RAW_DATA_FILE
    from src.preprocessing.clean_data import load_raw_data
    try:
        df_raw = load_raw_data(RAW_DATA_FILE)
        df_raw = df_raw.drop_duplicates().reset_index(drop=True)
        df_raw["Cluster"] = cluster_labels
        
        # Thống kê trung bình của từng cụm
        # Lọc ra các cột số để tính mean
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
        cluster_summary = df_raw[numeric_cols].groupby("Cluster").mean()
        
        # Thêm số lượng bệnh nhân trong mỗi cụm
        cluster_counts = df_raw["Cluster"].value_counts().sort_index()
        cluster_summary.insert(0, "Patient_Count", cluster_counts)
        
        # Lưu bảng thống kê cụm
        summary_path = METRICS_DIR / "clustering_summary.csv"
        cluster_summary.to_csv(summary_path)
        print(f"--> Đã xuất bảng phân tích cụm vào: {summary_path}")
    except Exception as e:
        print(f"Cảnh báo: Không thể phân tích đặc trưng cụm từ dữ liệu gốc: {e}")
        cluster_summary = None
        
    # 5. Lưu trữ artifacts (Model và PCA)
    kmeans_path = MODEL_DIR / "kmeans_model.pkl"
    pca_path = MODEL_DIR / "pca_model.pkl"
    
    joblib.dump({"model": kmeans, "n_clusters": n_clusters, "silhouette": sil_score}, kmeans_path)
    joblib.dump(pca, pca_path)
    print(f"--> Đã lưu model K-Means tại: {kmeans_path}")
    print(f"--> Đã lưu model PCA tại: {pca_path}")
    
    # Lưu metrics Silhouette riêng
    metrics_json_path = METRICS_DIR / "clustering_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "silhouette_score": float(sil_score),
            "n_clusters": int(n_clusters),
            "cluster_counts": cluster_counts.to_dict() if 'cluster_counts' in locals() else {}
        }, f, ensure_ascii=False, indent=4)
        
    print("=== PHÂN CỤM XONG ===")

if __name__ == "__main__":
    import sys
    # Fix encoding cho Windows console
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
        
    # Đảm bảo các thư mục tồn tại
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    run_clustering_pipeline(n_clusters=3)
