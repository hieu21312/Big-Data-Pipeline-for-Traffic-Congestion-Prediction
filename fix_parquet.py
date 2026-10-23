import pandas as pd

df = pd.read_parquet("data/pems_bay_long.parquet")
df["ts"] = df["ts"].dt.tz_localize("UTC")        # gắn múi giờ UTC
df.to_parquet("data/pems_bay_long_ms.parquet", index=False,
              coerce_timestamps="ms", allow_truncated_timestamps=True)
print(df.shape, df.dtypes.to_dict())