"""
Script tạo lại đầy đủ tất cả biểu đồ EDA từ notebook eda_dataset.ipynb
và lưu vào outputs/figures/eda/

Biểu đồ bao gồm:
1. Phân bố biến mục tiêu (condition) - countplot
2. Histogram các thuộc tính số
3. Boxplot phát hiện outlier
4. Boxplot so sánh từng thuộc tính số theo condition (5 biểu đồ)
5. Countplot các thuộc tính phân loại theo condition
6. Heatmap tương quan
7. Barplot mức độ tương quan với condition
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

# ── Đường dẫn ──
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Đọc dữ liệu từ: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Kích thước: {df.shape}")

numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


# ═══════════════════════════════════════
# 1. Phân bố biến mục tiêu (condition)
# ═══════════════════════════════════════
def plot_target_distribution():
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        x="condition", hue="condition", data=df,
        palette="Set2", legend=False, dodge=False
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    ax.set_title("Phân bố chẩn đoán bệnh tim (condition)")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Số lượng bệnh nhân")
    ax.margins(y=0.12)
    plt.tight_layout()
    path = OUTPUT_DIR / "01_target_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ {path.name}")


# ═══════════════════════════════════════
# 2. Histogram các thuộc tính số
# ═══════════════════════════════════════
def plot_numeric_histograms():
    df[numeric_cols].hist(bins=15, figsize=(14, 8), edgecolor="black")
    plt.suptitle("Phân bố các thuộc tính số", fontsize=15)
    plt.tight_layout()
    path = OUTPUT_DIR / "02_numeric_histograms.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ {path.name}")


# ═══════════════════════════════════════
# 3. Boxplot phát hiện outlier
# ═══════════════════════════════════════
def plot_boxplot_outliers():
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df[numeric_cols])
    ax.set_title("Boxplot các thuộc tính số")
    ax.set_xlabel("Thuộc tính")
    ax.set_ylabel("Giá trị")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = OUTPUT_DIR / "03_boxplot_outliers.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ {path.name}")


# ═══════════════════════════════════════
# 4. So sánh từng thuộc tính số theo condition
# ═══════════════════════════════════════
def plot_boxplot_by_condition():
    for i, col in enumerate(numeric_cols, start=1):
        plt.figure(figsize=(7, 4))
        ax = sns.boxplot(
            x="condition", y=col, hue="condition",
            data=df, palette="Set3", legend=False
        )
        ax.set_title(f"So sánh {col} theo condition")
        ax.set_xlabel("Condition")
        ax.set_ylabel(col)
        plt.tight_layout()
        path = OUTPUT_DIR / f"04_{i}_boxplot_{col}_by_condition.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✓ {path.name}")


# ═══════════════════════════════════════
# 5. Countplot các thuộc tính phân loại theo condition
# ═══════════════════════════════════════
def plot_categorical_countplots():
    for i, col in enumerate(categorical_cols, start=1):
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(
            x=col, hue="condition", data=df,
            palette="Set1"
        )
        ax.set_title(f"Phân bố {col} theo condition")
        ax.set_xlabel(col)
        ax.set_ylabel("Số lượng")
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=3)
        ax.margins(y=0.12)
        plt.tight_layout()
        path = OUTPUT_DIR / f"05_{i}_countplot_{col}_by_condition.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✓ {path.name}")


# ═══════════════════════════════════════
# 6. Heatmap tương quan
# ═══════════════════════════════════════
def plot_correlation_heatmap():
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    ax = sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5,
        square=True, vmin=-1, vmax=1
    )
    ax.set_title("Ma trận tương quan giữa các thuộc tính")
    plt.tight_layout()
    path = OUTPUT_DIR / "06_correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ {path.name}")


# ═══════════════════════════════════════
# 7. Barplot mức độ tương quan với condition
# ═══════════════════════════════════════
def plot_correlation_with_target():
    corr = df.corr(numeric_only=True)
    if "condition" not in corr.columns:
        print("  ⚠ Cột condition không tồn tại, bỏ qua biểu đồ tương quan target.")
        return
    corr_target = corr["condition"].drop("condition").sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in corr_target.values]
    ax = plt.barh(corr_target.index, corr_target.values, color=colors)
    plt.title("Mức độ tương quan của các thuộc tính với condition")
    plt.xlabel("Hệ số tương quan Pearson")
    plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    path = OUTPUT_DIR / "07_correlation_with_target.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ {path.name}")


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
if __name__ == "__main__":
    print("\n=== TẠO BIỂU ĐỒ EDA ===\n")

    plot_target_distribution()
    plot_numeric_histograms()
    plot_boxplot_outliers()
    plot_boxplot_by_condition()
    plot_categorical_countplots()
    plot_correlation_heatmap()
    plot_correlation_with_target()

    print(f"\n=== HOÀN TẤT: Tất cả biểu đồ đã lưu vào {OUTPUT_DIR} ===\n")
