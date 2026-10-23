import os, sys, csv, json
from datetime import datetime, timedelta

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
for d in ["D:/spark-ivy", "D:/spark-tmp", "data/results"]:
    os.makedirs(d, exist_ok=True)

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ============================================================
# THAM SO
# ============================================================
H          = 6              
CONG_RATIO = 0.4            
SRC        = "data/readings_cassandra.parquet"
FEAT_PATH  = f"data/features_h{H}.parquet"
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.15, 0.2, 0.2

THRS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

# ============================================================
# CONFIG THOI GIAN
# ============================================================
cfg = json.load(open("data/config.json"))
GAP = timedelta(minutes=5 * H)     
t_train = datetime.fromisoformat(cfg["t_train"])
t_val   = datetime.fromisoformat(cfg["t_val"])

# ============================================================
# SPARK SESSION
# ============================================================
ADD_OPENS = " ".join([f"--add-opens=java.base/{p}=ALL-UNNAMED"
    for p in ["java.lang", "java.lang.invoke", "java.lang.reflect", "java.io",
              "java.net", "java.nio", "java.util", "java.util.concurrent",
              "java.util.concurrent.atomic", "sun.nio.ch", "sun.nio.cs",
              "sun.security.action", "sun.util.calendar"]])

spark = (SparkSession.builder.appName("pipeline-v6").master("local[2]")
    .config("spark.jars.ivy", "D:/spark-ivy")
    .config("spark.local.dir", "D:/spark-tmp")
    .config("spark.driver.memory", "2g")
    .config("spark.driver.extraJavaOptions", ADD_OPENS + " -Duser.timezone=UTC")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ============================================================
# 1) LAM SACH + TAO DAC TRUNG (cache ra Parquet)
# ============================================================
print("=" * 60); print("BUOC 1-3: LAM SACH + TAO DAC TRUNG"); print("=" * 60, flush=True)

if not os.path.exists(FEAT_PATH):
    print("Chua co cache. Tinh feature lan dau (~5-7 phut)...", flush=True)

    df = (spark.read.parquet(SRC)
          .select(F.col("sensor_id").cast("string").alias("sensor_id"),
                  F.col("ts").cast("timestamp").alias("ts"),
                  F.col("speed").cast("double").alias("speed")))

    # Lam sach
    df = (df.filter(F.col("speed").isNotNull() & (F.col("speed") > 0) &
                    (F.col("speed") <= 100))
            .dropDuplicates(["sensor_id", "ts"]))

    ff = (df.filter(F.col("ts") < F.lit(t_train - GAP))
            .groupBy("sensor_id")
            .agg(F.expr("percentile_approx(speed, 0.95)").alias("free_flow")))

    w = Window.partitionBy("sensor_id").orderBy("ts")
    feat = (df.join(ff, "sensor_id")
        # Thoi gian
        .withColumn("hour", F.hour("ts"))
        .withColumn("dow", F.dayofweek("ts"))
        .withColumn("is_weekend", F.when(F.col("dow").isin(1, 7), 1.0).otherwise(0.0))
        # Qua khu
        .withColumn("speed_lag1",     F.lag("speed", 1).over(w))
        .withColumn("speed_mean_15m", F.avg("speed").over(w.rowsBetween(-2, 0)))
        .withColumn("speed_mean_60m", F.avg("speed").over(w.rowsBetween(-11, 0)))
        .withColumn("speed_std_15m",  F.stddev("speed").over(w.rowsBetween(-2, 0)))
        # Ty le hien tai
        .withColumn("speed_ratio", F.col("speed") / F.col("free_flow"))
        # Tuong lai
        .withColumn("n_future",     F.count("speed").over(w.rowsBetween(1, H)))
        .withColumn("speed_future", F.avg("speed").over(w.rowsBetween(1, H)))
        .filter(F.col("n_future") == H)
        .dropna(subset=["speed_lag1", "speed_std_15m"])
        # Nhan: tac nghen trong 15 phut toi
        .withColumn("label", F.when(
            F.col("speed_future") < CONG_RATIO * F.col("free_flow"), 1.0).otherwise(0.0))
        # Trang thai hien tai: dang tac hay khong
        .withColumn("cur_cong", F.when(
            F.col("speed") < CONG_RATIO * F.col("free_flow"), 1.0).otherwise(0.0))
        .drop("n_future"))

    feat.write.mode("overwrite").parquet(FEAT_PATH)
    print(f"Da luu dac trung: {FEAT_PATH}", flush=True)
else:
    print(f"Dung cache: {FEAT_PATH}", flush=True)
    print(f"(Xoa thu muc nay de tinh lai feature)", flush=True)

# ============================================================
# 2) CHIA TRAIN / VAL / TEST THEO THOI GIAN
# ============================================================
print("\n" + "=" * 60); print("BUOC 4: CHIA TAP"); print("=" * 60, flush=True)

data = spark.read.parquet(FEAT_PATH)

train = data.filter(F.col("ts") < F.lit(t_train - GAP)).sample(TRAIN_FRAC, seed=42)
val   = data.filter((F.col("ts") >= F.lit(t_train)) &
                    (F.col("ts") < F.lit(t_val - GAP))).sample(VAL_FRAC, seed=42)
test  = data.filter(F.col("ts") >= F.lit(t_val)).sample(TEST_FRAC, seed=42)

for d in (train, val, test):
    d.persist(StorageLevel.MEMORY_AND_DISK)

cnt = {r["label"]: r["count"] for r in train.groupBy("label").count().collect()}
n_pos, n_neg = cnt.get(1.0, 0), cnt.get(0.0, 0)
n = n_pos + n_neg
print(f"Train: {n:,} dong (pos={n_pos:,}, neg={n_neg:,}, ty le pos={n_pos/max(n,1):.2%})")
print(f"Val:   {val.count():,} dong")
print(f"Test:  {test.count():,} dong", flush=True)

# Class weight cho imbalance
w_pos, w_neg = n / (2.0 * max(n_pos, 1)), n / (2.0 * max(n_neg, 1))
print(f"Class weight: pos={w_pos:.2f}, neg={w_neg:.2f}", flush=True)
train = train.withColumn("weight",
    F.when(F.col("label") == 1, F.lit(w_pos)).otherwise(F.lit(w_neg)))

# *** FEATURES: bao gom cur_cong ***
FEATURES = ["speed", "speed_lag1", "speed_mean_15m", "speed_mean_60m",
            "speed_std_15m",
            "cur_cong",
            "hour", "dow", "is_weekend"]

assembler = VectorAssembler(
    inputCols=FEATURES, outputCol="features", handleInvalid="skip")

# ============================================================
# 3) HAM DANH GIA
# ============================================================
auc_eval = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="p", metricName="areaUnderROC")


def conf_all(pred, thrs):
    """Tinh confusion matrix cho nhieu nguong cung luc (1 lan quet du lieu)."""
    aggs = []
    y = F.col("label") == 1
    for i, t in enumerate(thrs):
        pos = F.col("p") >= t
        aggs += [F.sum(F.when(y & pos, 1).otherwise(0)).alias(f"tp{i}"),
                 F.sum(F.when(~y & pos, 1).otherwise(0)).alias(f"fp{i}"),
                 F.sum(F.when(y & ~pos, 1).otherwise(0)).alias(f"fn{i}"),
                 F.sum(F.when(~y & ~pos, 1).otherwise(0)).alias(f"tn{i}")]
    r = pred.agg(*aggs).first()
    return {t: tuple(int(r[f"{k}{i}"]) for k in ("tp", "fp", "fn", "tn"))
            for i, t in enumerate(thrs)}


def prf(tp, fp, fn, tn):
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    acc  = (tp + tn) / max(tp + fp + fn + tn, 1)
    return acc, prec, rec, f1


def scored(model, df):
    """Transform + lay cot probability lop 1 lam 'p'."""
    return (model.transform(df)
            .select("sensor_id", "ts", "speed", "speed_future", "label",
                    vector_to_array("probability")[1].alias("p")))


results = []


def add_result(name, thr, cm, auc):
    acc, prec, rec, f1 = prf(*cm)
    tp, fp, fn, tn = cm
    print(f"  thr={thr:.2f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
          f"F1={f1:.4f}  AUC={auc:.4f}  (TP={tp} FP={fp} FN={fn} TN={tn})",
          flush=True)
    results.append(dict(name=name, thr=thr, acc=acc, prec=prec, rec=rec,
                        f1=f1, auc=auc, tp=tp, fp=fp, fn=fn, tn=tn))


# ============================================================
# 4) BASELINE: PERSISTENCE
# ============================================================
print("\n" + "=" * 60); print("BASELINE: PERSISTENCE"); print("=" * 60, flush=True)
base = test.select("label", F.col("cur_cong").alias("p"))
add_result("Persistence", 0.5, conf_all(base, [0.5])[0.5], auc_eval.evaluate(base))

# ============================================================
# 5) HUAN LUYEN 3 MO HINH
# ============================================================
print("\n" + "=" * 60); print("HUAN LUYEN 3 MO HINH"); print("=" * 60, flush=True)

models = {
    "LogisticRegression": LogisticRegression(
        featuresCol="features", labelCol="label", weightCol="weight",
        maxIter=20, regParam=0.01),
    "RandomForest": RandomForestClassifier(
        featuresCol="features", labelCol="label", weightCol="weight",
        numTrees=30, maxDepth=6, seed=42),
    "GBT": GBTClassifier(
        featuresCol="features", labelCol="label", weightCol="weight",
        maxIter=20, maxDepth=5, seed=42),
}

trained_models = {}

for name, clf in models.items():
    print(f"\n--- {name} ---", flush=True)
    model = Pipeline(stages=[assembler, clf]).fit(train)
    trained_models[name] = model

    # Chon nguong toi uu F1 tren VALIDATION
    cm_val = conf_all(scored(model, val), THRS)
    best = max(THRS, key=lambda t: prf(*cm_val[t])[3])
    print(f"  Nguong toi uu F1 tren val: {best}", flush=True)

    # Danh gia tren TEST
    pt = scored(model, test).persist()
    add_result(name, best, conf_all(pt, [best])[best], auc_eval.evaluate(pt))

    # Luu model + prediction
    model.write().overwrite().save(f"data/results/model_{name}")
    pt.write.mode("overwrite").parquet(f"data/results/pred_{name}.parquet")
    pt.unpersist()

# ============================================================
# 5b) FEATURE IMPORTANCE (cho RF va GBT)
# ============================================================
print("\n" + "=" * 60); print("FEATURE IMPORTANCE"); print("=" * 60, flush=True)

for name in ["RandomForest", "GBT"]:
    if name not in trained_models:
        continue
    try:
        stage = trained_models[name].stages[-1]
        imp = stage.featureImportances.toArray()
        pairs = sorted(zip(FEATURES, imp), key=lambda x: -x[1])
        print(f"\n--- {name} ---")
        for fname, val in pairs:
            print(f"  {fname:<18} {val:.4f}")
    except Exception as e:
        print(f"Khong lay duoc feature importance cua {name}: {e}")

# ============================================================
# 6) TONG KET
# ============================================================
print("\n" + "=" * 60); print("TONG KET (tap TEST)"); print("=" * 60, flush=True)
print(f"{'Model':<20} {'Thr':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("-" * 66)
for r in results:
    print(f"{r['name']:<20} {r['thr']:>5.2f} {r['acc']:>7.4f} {r['prec']:>7.4f} "
          f"{r['rec']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f}")

# Luu CSV
with open("data/results/summary.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1",
                 "AUC", "TP", "FP", "FN", "TN"])
    for r in results:
        wr.writerow([r["name"], r["thr"], f"{r['acc']:.4f}", f"{r['prec']:.4f}",
                     f"{r['rec']:.4f}", f"{r['f1']:.4f}", f"{r['auc']:.4f}",
                     r["tp"], r["fp"], r["fn"], r["tn"]])

print("\nDa luu:")
print("  data/results/summary.csv")
print("  data/results/model_*/")
print("  data/results/pred_*.parquet")
print("\nHOAN THANH!")
spark.stop()