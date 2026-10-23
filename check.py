import pandas as pd, pickle

speed = pd.read_hdf("data/pems-bay.h5")
print("Kích thước:", speed.shape)
print("Từ", speed.index.min(), "đến", speed.index.max())

meta = pd.read_hdf("data/pems-bay-meta.h5")
print("Cột meta:", meta.columns.tolist())
print(meta.head())

ids, idx, adj = pickle.load(open("data/adj_mx_bay.pkl", "rb"), encoding="latin1")
print("Ma trận kề:", len(ids), adj.shape)

n = len(speed)
print("Mốc cắt train:", speed.index[int(n*0.7)])
print("Mốc cắt val:  ", speed.index[int(n*0.8)])