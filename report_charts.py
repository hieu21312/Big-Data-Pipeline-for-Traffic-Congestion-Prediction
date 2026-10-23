import os, sys, csv
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.makedirs("report/figs", exist_ok=True)

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as auc_sklearn

# ============================================================
# SPARK
# ============================================================
ADD_OPENS = " ".join([f"--add-opens=java.base/{p}=ALL-UNNAMED"
    for p in ["java.lang","java.lang.invoke","java.lang.reflect","java.io",
              "java.net","java.nio","java.util","java.util.concurrent",
              "java.util.concurrent.atomic","sun.nio.ch","sun.nio.cs",
              "sun.security.action","sun.util.calendar"]])

spark = (SparkSession.builder.appName("report-charts").master("local[2]")
    .config("spark.jars.ivy", "D:/spark-ivy")
    .config("spark.local.dir", "D:/spark-tmp")
    .config("spark.driver.memory", "2g")
    .config("spark.driver.extraJavaOptions", ADD_OPENS)
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("VE BIEU DO CHO BAO CAO")
print("=" * 60, flush=True)

# ============================================================
# 1) DOC SUMMARY
# ============================================================
summary = pd.read_csv("data/results/summary.csv")
print("\nDoc summary.csv:")
print(summary.to_string(index=False))

# ============================================================
# BIEU DO 1 — So sanh F1 va AUC giua cac model
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(summary))
w = 0.35
ax.bar(x - w/2, summary["F1"], w, label="F1-score", color="#3b82f6")
ax.bar(x + w/2, summary["AUC"], w, label="AUC-ROC", color="#f59e0b")
for i, (f1, a) in enumerate(zip(summary["F1"], summary["AUC"])):
    ax.text(i - w/2, f1 + 0.01, f"{f1:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(summary["Model"], rotation=15, ha="right")
ax.set_ylabel("Gia tri")
ax.set_title("So sanh F1-score va AUC-ROC giua cac mo hinh")
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig("report/figs/01_model_comparison.png", dpi=150)
plt.close()
print("Da luu: 01_model_comparison.png")

# ============================================================
# BIEU DO 2 — Confusion Matrix cua GBT
# ============================================================
best = "GBT"
row = summary[summary["Model"] == best].iloc[0]
tp, tn, fp, fn = int(row["TP"]), int(row["TN"]), int(row["FP"]), int(row["FN"])

cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Du doan: Khong tac", "Du doan: Tac"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Thuc te: Khong tac", "Thuc te: Tac"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
ax.set_title(f"Confusion Matrix — {best} (threshold = {row['Threshold']})")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("report/figs/02_confusion_matrix_gbt.png", dpi=150)
plt.close()
print("Da luu: 02_confusion_matrix_gbt.png")

# ============================================================
# BIEU DO 3 — ROC Curve cac model
# ============================================================
print("\nDang tinh ROC curves...", flush=True)
fig, ax = plt.subplots(figsize=(7, 7))

for name in ["LogisticRegression", "RandomForest", "GBT"]:
    path = f"data/results/pred_{name}.parquet"
    if not os.path.exists(path):
        continue
    df = spark.read.parquet(path).select("label", "p")
    # Sample de ve nhanh
    pdf = df.sample(fraction=0.02, seed=42).toPandas()
    fpr, tpr, _ = roc_curve(pdf["label"], pdf["p"])
    auc_val = auc_sklearn(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — So sanh cac mo hinh")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("report/figs/03_roc_curves.png", dpi=150)
plt.close()
print("Da luu: 03_roc_curves.png")

# ============================================================
# BIEU DO 4 — Feature Importance cua GBT
# ============================================================
print("\nDang lay feature importance...", flush=True)
FEATURES = ["speed", "speed_lag1", "speed_mean_15m", "speed_mean_60m",
            "speed_std_15m", "cur_cong", "hour", "dow", "is_weekend"]
try:
    model = PipelineModel.load("data/results/model_GBT")
    clf = model.stages[-1]
    imp = clf.featureImportances.toArray()
    pairs = sorted(zip(FEATURES, imp), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    ax.barh(names, vals, color="#10b981")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Do quan trong (Feature Importance)")
    ax.set_title("Feature Importance — GBT")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("report/figs/04_feature_importance.png", dpi=150)
    plt.close()
    print("Da luu: 04_feature_importance.png")
except Exception as e:
    print(f"Khong ve duoc feature importance: {e}")

# ============================================================
# BIEU DO 5 — EDA: Toc do trung binh theo gio
# ============================================================
print("\nDang ve EDA speed by hour...", flush=True)
feat = spark.read.parquet("data/features_h6.parquet") \
    .select("hour", "speed") \
    .groupBy("hour").agg(F.avg("speed").alias("mean_speed"),
                         F.count("speed").alias("n")) \
    .orderBy("hour").toPandas()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(feat["hour"], feat["mean_speed"], marker="o",
        lw=2, color="#6366f1")
ax.axvspan(7, 9, alpha=0.15, color="red", label="Gio cao diem sang")
ax.axvspan(16, 19, alpha=0.15, color="orange", label="Gio cao diem chieu")
ax.set_xlabel("Gio trong ngay")
ax.set_ylabel("Toc do trung binh (mph)")
ax.set_title("Toc do trung binh theo gio — EDA")
ax.grid(alpha=0.3)
ax.set_xticks(range(0, 24, 2))
ax.legend()
plt.tight_layout()
plt.savefig("report/figs/05_speed_by_hour.png", dpi=150)
plt.close()
print("Da luu: 05_speed_by_hour.png")

print("\n" + "=" * 60)
print("HOAN THANH! 5 hinh da luu trong report/figs/")
print("=" * 60)
spark.stop()