# =====================================================================
#  第 3 个脚本：SVM 用的整段特征 + 标准化
#  先跑 step1_download_split.py，这里要读它生成的 data/metadata.csv
# =====================================================================

import os
import pandas as pd
import opensmile

# 路径：从这个脚本所在的 preprocessing/ 往上一层就是 AER/
AER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(AER_DIR, "data")
WAV_DIR = os.path.join(DATA_DIR, "original_data")
FUNC_DIR = os.path.join(DATA_DIR, "features", "functionals")

# 读第 1 个脚本存的总表（actor 按字符串读）
df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"), dtype={"actor": str})


# ---------------------------------------------------------------------
# 第 1 步 · 提整段特征（给 SVM）
# ---------------------------------------------------------------------
# 把每段录音压成一行 88 个数（平均有多高、变化有多大之类的统计量），给 SVM 用。

smile_func = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

func_list = []
for i, filename in enumerate(df["filename"]):
    feat = smile_func.process_file(os.path.join(WAV_DIR, filename + ".wav"))
    func_list.append(feat)
func_df = pd.concat(func_list).reset_index(drop=True)
func_df.insert(0, "filename", df["filename"].values)
func_df.to_csv(os.path.join(FUNC_DIR, "functionals.csv"), index=False)


# ---------------------------------------------------------------------
# 第 2 步 · 整段特征标准化（只用训练集）
# ---------------------------------------------------------------------
# 把每个特征都缩放到「平均 0、标准差 1」，让模型公平地看待每个特征。
# 只取训练演员的那些行，每一列算一个均值、一个标准差 → 88 对

train_files = df[df["split"] == "train"]["filename"]
train_func = func_df[func_df["filename"].isin(train_files)].drop(columns=["filename"])
func_mean = train_func.mean()
func_std = train_func.std()
pd.DataFrame({"mean": func_mean, "std": func_std}).to_csv(
    os.path.join(FUNC_DIR, "functionals_norm.csv"))
