# =====================================================================
#  第 2 个脚本：LSTM 用的逐帧特征 + 标准化
#  先跑 step1_download_split.py，这里要读它生成的 data/metadata.csv
# =====================================================================

import os
import numpy as np
import pandas as pd
import opensmile

# 路径：从这个脚本所在的 preprocessing/ 往上一层就是 AER/
AER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(AER_DIR, "data")
WAV_DIR = os.path.join(DATA_DIR, "original_data")
FEAT_DIR = os.path.join(DATA_DIR, "features")
LLD_DIR = os.path.join(FEAT_DIR, "lld")

# 读第 1 个脚本存的总表（actor 按字符串读）
df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"), dtype={"actor": str})


# ---------------------------------------------------------------------
# 第 1 步 · 提逐帧特征（给 LSTM）
# ---------------------------------------------------------------------
# 把每段录音变成一张表：每 10 毫秒一行，每行 25 个数。
# LSTM 就是这样一行一行（一帧一帧）往下读的。

smile_lld = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

n_frames = []
for i, filename in enumerate(df["filename"]):
    lld = smile_lld.process_file(os.path.join(WAV_DIR, filename + ".wav"))
    np.save(os.path.join(LLD_DIR, filename + ".npy"), lld.values)
    n_frames.append(len(lld))
df["n_frames"] = n_frames
df.to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)


# ---------------------------------------------------------------------
# 第 2 步 · 逐帧特征标准化（只用训练集）
# ---------------------------------------------------------------------
# 把每个特征都缩放到「平均 0、标准差 1」，让模型公平地看待每个特征。
# 把所有训练录音的所有帧摞在一起，每一列算一个均值、一个标准差 → 25 对

train_df = df[df["split"] == "train"]

all_frames = []
for filename in train_df["filename"]:
    all_frames.append(np.load(os.path.join(LLD_DIR, filename + ".npy")))
all_frames = np.concatenate(all_frames, axis=0)
lld_mean = all_frames.mean(axis=0)
lld_std = all_frames.std(axis=0)
np.save(os.path.join(FEAT_DIR, "lld_mean.npy"), lld_mean)
np.save(os.path.join(FEAT_DIR, "lld_std.npy"), lld_std)
