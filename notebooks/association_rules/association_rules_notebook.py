"""
NOTEBOOK: KHAI PHÁ LUẬT KẾT HỢP (Association Rules)
====================================================
Bộ dữ liệu chẩn đoán bệnh tim

Notebook này minh họa toàn bộ quy trình khai phá luật kết hợp:
1. Đọc dữ liệu đã tiền xử lý
2. Chuyển sang dạng boolean cho khai phá luật
3. Chạy thuật toán Apriori
4. Chạy thuật toán FP-Growth
5. Lọc và xuất luật kết hợp
6. Trực quan hóa kết quả

Chạy: python notebooks/association_rules/association_rules_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix encoding cho Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.association_rules.apriori_rules import run_apriori
from src.association_rules.fp_growth_rules import run_fpgrowth
from src.association_rules.export_rules import export_rules_to_csv
from src.preprocessing.config import PROCESSED_DATA_FILE

# Thư mục lưu biểu đồ
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "association_rules"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

data_path = str(PROCESSED_DATA_FILE)


# ════════════════════════════════════════
# 1. Đọc dữ liệu
# ════════════════════════════════════════
print("=" * 60)
print(" KHAI PHÁ LUẬT KẾT HỢP")
print("=" * 60)

print(f"\n[1] Đọc dữ liệu từ: {PROCESSED_DATA_FILE}")
df = pd.read_csv(data_path)
print(f"    Kích thước: {df.shape}")
print(f"    Các cột: {df.columns.tolist()}")


# ════════════════════════════════════════
# 2. Chạy Apriori
# ════════════════════════════════════════
print(f"\n[2] Chạy thuật toán Apriori...")
apriori_rules = run_apriori(data_path, min_support=0.15, min_confidence=0.7)

if apriori_rules is not None:
    print(f"    -> Tìm được {len(apriori_rules)} luật (Apriori)")
    export_rules_to_csv(apriori_rules, 'apriori_rules.csv', top_k=100, min_lift=1.2)
    print(f"    -> Đã xuất file apriori_rules.csv")
else:
    print("    -> Không tìm thấy luật nào (Apriori)")


# ════════════════════════════════════════
# 3. Chạy FP-Growth
# ════════════════════════════════════════
print(f"\n[3] Chạy thuật toán FP-Growth...")
fpgrowth_rules = run_fpgrowth(data_path, min_support=0.15, min_confidence=0.7)

if fpgrowth_rules is not None:
    print(f"    -> Tìm được {len(fpgrowth_rules)} luật (FP-Growth)")
    export_rules_to_csv(fpgrowth_rules, 'fp_growth_rules.csv', top_k=100, min_lift=1.2)
    print(f"    -> Đã xuất file fp_growth_rules.csv")
else:
    print("    -> Không tìm thấy luật nào (FP-Growth)")


# ════════════════════════════════════════
# 4. Trực quan hóa kết quả
# ════════════════════════════════════════
print(f"\n[4] Trực quan hóa kết quả...")

# Chọn rules chính (FP-Growth hoặc Apriori, ưu tiên FP-Growth)
rules = fpgrowth_rules if fpgrowth_rules is not None else apriori_rules

if rules is not None and len(rules) > 0:
    # 4a. Scatter plot: Support vs Confidence
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        rules["support"], rules["confidence"],
        c=rules["lift"], cmap="RdYlGn", alpha=0.7,
        s=50, edgecolors="black", linewidths=0.5
    )
    plt.colorbar(scatter, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Luật kết hợp: Support vs Confidence (màu = Lift)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "support_vs_confidence.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    ✓ {path.name}")

    # 4b. Histogram phân bố Lift
    plt.figure(figsize=(8, 5))
    plt.hist(rules["lift"], bins=20, color="#3498db", edgecolor="black", alpha=0.8)
    plt.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="Lift = 1.0")
    plt.xlabel("Lift")
    plt.ylabel("Số luật")
    plt.title("Phân bố giá trị Lift của các luật kết hợp")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "lift_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    ✓ {path.name}")

    # 4c. Top 15 luật theo Lift
    top_rules = rules.nlargest(15, "lift").copy()
    top_rules["rule_label"] = top_rules.apply(
        lambda r: f"{set(r['antecedents'])} → {set(r['consequents'])}", axis=1
    )
    # Rút gọn label nếu quá dài
    top_rules["rule_label"] = top_rules["rule_label"].apply(
        lambda x: x[:60] + "..." if len(x) > 60 else x
    )

    plt.figure(figsize=(12, 7))
    bars = plt.barh(top_rules["rule_label"], top_rules["lift"], color="#2ecc71", edgecolor="black")
    plt.xlabel("Lift")
    plt.ylabel("Luật")
    plt.title("Top 15 luật kết hợp có Lift cao nhất")
    for bar in bars:
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.2f}', va='center', fontsize=9)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "top15_rules_by_lift.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    ✓ {path.name}")

    # 4d. Heatmap: Support vs Confidence vs Lift (binned)
    plt.figure(figsize=(10, 6))
    if len(rules) > 5:
        rules_plot = rules.copy()
        rules_plot["support_bin"] = pd.cut(rules_plot["support"], bins=5)
        rules_plot["confidence_bin"] = pd.cut(rules_plot["confidence"], bins=5)
        pivot = rules_plot.groupby(["support_bin", "confidence_bin"])["lift"].mean().unstack()
        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5)
            plt.title("Trung bình Lift theo Support và Confidence")
            plt.xlabel("Confidence (binned)")
            plt.ylabel("Support (binned)")
            plt.tight_layout()
            path = FIGURES_DIR / "lift_heatmap.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"    ✓ {path.name}")

else:
    print("    ⚠ Không có luật nào để trực quan hóa.")


print(f"\n{'=' * 60}")
print(" KHAI PHÁ LUẬT KẾT HỢP HOÀN TẤT!")
print(f"{'=' * 60}")
print(f"  Figures: {FIGURES_DIR}")
