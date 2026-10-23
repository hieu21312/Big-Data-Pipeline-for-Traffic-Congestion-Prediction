import pandas as pd, numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("report/figs", exist_ok=True)

# ---------- 1. Đọc dữ liệu ----------
wide = pd.read_hdf("data/pems-bay.h5")
wide.index.name = "ts"
wide.columns = wide.columns.astype(str)
total = wide.size
print("Tổng số ô dữ liệu:", total)

# ---------- 2. Thống kê chất lượng (đưa vào báo cáo) ----------
n_zero = int((wide == 0).sum().sum())
n_nan = int(wide.isna().sum().sum())
n_high = int((wide > 120).sum().sum())
print(f"Giá trị 0 (cảm biến lỗi): {n_zero} ({n_zero/total:.2%})")
print(f"Giá trị NaN: {n_nan}")
print(f"Giá trị > 120 mph (ngoại lệ): {n_high}")

# ---------- 3. Làm sạch ----------
wide = wide.replace(0, np.nan)             # 0 = thiếu
wide = wide.mask(wide > 120)               # ngoại lệ vật lý
missing_before = wide.isna().mean().sort_values(ascending=False)
print("Top 5 cảm biến thiếu nhiều nhất:\n", missing_before.head())

# thiếu ngắn (<= 6 mốc = 30 phút): nội suy tuyến tính
wide = wide.interpolate(limit=6, limit_direction="both")
print("Còn thiếu sau nội suy:", f"{wide.isna().mean().mean():.2%}")

# ---------- 4. Chuyển sang dạng dài ----------
long = (wide.reset_index()
            .melt(id_vars="ts", var_name="sensor_id", value_name="speed")
            .dropna())
long = long.drop_duplicates(["sensor_id", "ts"])
long["speed"] = long["speed"].astype("float32")
long.to_parquet("data/pems_bay_long.parquet", index=False)
print("Dạng dài:", long.shape)

# ---------- 5. Bảng cảm biến từ meta ----------
meta = pd.read_hdf("data/pems-bay-meta.h5").reset_index()
meta["sensor_id"] = meta["sensor_id"].astype(str)
sensors = meta.rename(columns={"Latitude": "latitude", "Longitude": "longitude",
                               "Fwy": "fwy", "Dir": "dir", "Abs_PM": "abs_pm",
                               "Lanes": "lanes"})[
    ["sensor_id", "latitude", "longitude", "fwy", "dir", "abs_pm", "lanes"]]
sensors["fwy"] = sensors["fwy"].astype(str)
sensors.to_parquet("data/sensors.parquet", index=False)
print("Sensors:", sensors.shape)

# ---------- 6. Lưu mốc cắt train/val/test ----------
n = len(wide)
cfg = {"t_train": str(wide.index[int(n * 0.7)]),
       "t_val": str(wide.index[int(n * 0.8)])}
json.dump(cfg, open("data/config.json", "w"))
print("Mốc cắt:", cfg)

# ---------- 7. Biểu đồ EDA ----------
avg = wide.mean(axis=1)
plt.figure(figsize=(8, 4))
avg.groupby(avg.index.hour).mean().plot(marker="o")
plt.title("Tốc độ trung bình theo giờ"); plt.xlabel("Giờ"); plt.ylabel("mph")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("report/figs/speed_by_hour.png", dpi=150); plt.close()

hm = avg.groupby([avg.index.dayofweek, avg.index.hour]).mean().unstack()
plt.figure(figsize=(9, 3.5))
plt.imshow(hm.values, aspect="auto", cmap="RdYlGn")
plt.colorbar(label="mph")
plt.yticks(range(7), ["T2", "T3", "T4", "T5", "T6", "T7", "CN"])
plt.xlabel("Giờ"); plt.title("Tốc độ trung bình: thứ × giờ")
plt.tight_layout(); plt.savefig("report/figs/heatmap_dow_hour.png", dpi=150); plt.close()

plt.figure(figsize=(8, 4))
missing_before.mul(100).plot(kind="hist", bins=30)
plt.title("Phân bố tỷ lệ thiếu theo cảm biến"); plt.xlabel("% thiếu")
plt.tight_layout(); plt.savefig("report/figs/missing_hist.png", dpi=150); plt.close()

vals = wide.values.ravel()
rng = np.random.default_rng(1)
plt.figure(figsize=(8, 4))
plt.hist(rng.choice(vals, 200000, replace=False), bins=60)
plt.title("Phân bố tốc độ"); plt.xlabel("mph")
plt.tight_layout(); plt.savefig("report/figs/speed_hist.png", dpi=150); plt.close()