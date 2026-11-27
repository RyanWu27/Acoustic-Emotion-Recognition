# =====================================================================
#  第 1 个脚本：准备环境 → 下载数据 → 读标签 → 检查音频 → 按演员划分
#  跑完得到：data/metadata.csv（总表）和 data/splits/splits.txt（演员名单）
# =====================================================================


# ---------------------------------------------------------------------
# 第 0 步 · 准备环境和文件夹
# ---------------------------------------------------------------------

import os
import sys
import glob
import json
import subprocess
import urllib.request

# 没装的包自动用 pip 装上
packages = ["numpy", "pandas", "soundfile", "opensmile"]
for p in packages:
    try:
        __import__(p)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p])

import numpy as np
import pandas as pd
import soundfile as sf

# 创建路径：从这个脚本所在的 preprocessing/
AER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(AER_DIR, "data")
WAV_DIR = os.path.join(DATA_DIR, "original_data")
LLD_DIR = os.path.join(DATA_DIR, "features", "lld")
FUNC_DIR = os.path.join(DATA_DIR, "features", "functionals")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")

os.makedirs(WAV_DIR, exist_ok=True)
os.makedirs(LLD_DIR, exist_ok=True)
os.makedirs(FUNC_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 第 1 步 · 下载数据
# ---------------------------------------------------------------------
# 从官方 GitHub（CheyneyComputerScience/CREMA-D）下载 7442 段 wav，
# 放进 data/original_data/。
#
# 下载方式：
#   1. 用 GitHub API 拿到 AudioWAV/ 里所有文件名
#   2. 一个一个下载真正的 wav
# data/original_data/ 里已经有 7442 个就不下载

API_URL = "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/trees/"
WAV_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/"

wav_files = glob.glob(os.path.join(WAV_DIR, "*.wav"))
if len(wav_files) < 7442:
    root = json.load(urllib.request.urlopen(API_URL + "master"))
    for item in root["tree"]:
        if item["path"] == "AudioWAV":
            audio_sha = item["sha"]
    audio_tree = json.load(urllib.request.urlopen(API_URL + audio_sha))

    for i, item in enumerate(audio_tree["tree"]):
        filename = item["path"]
        out_path = os.path.join(WAV_DIR, filename)
        data = urllib.request.urlopen(WAV_URL + filename).read()
        with open(out_path, "wb") as f:
            f.write(data)

# 下载完重新找一遍，拿到完整的文件列表
wav_files = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")))


# ---------------------------------------------------------------------
# 第 2 步 · 从文件名读出标签
# ---------------------------------------------------------------------
# CREMA-D 没有单独的标签表，演员是谁、什么情绪，都写在文件名里。

emotion_to_label = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}

rows = []
for f in wav_files:
    filename = os.path.basename(f).replace(".wav", "")
    parts = filename.split("_")
    rows.append({
        "filename": filename,
        "actor": parts[0],
        "label": emotion_to_label[parts[2]],
    })
df = pd.DataFrame(rows)

# ---------------------------------------------------------------------
# 第 3 步 · 检查音频
# ---------------------------------------------------------------------
# 提特征之前，先确认每个文件都能读、格式都一样，免得跑到一半报错。

durations = []
bad_files = []
for filename in df["filename"]:
    f = os.path.join(WAV_DIR, filename + ".wav")
    try:
        info = sf.info(f)
        durations.append(info.frames / info.samplerate)
        if info.frames == 0 or info.samplerate != 16000 or info.channels != 1:
            bad_files.append(filename)
    except Exception:
        durations.append(0)
        bad_files.append(filename)

df["duration"] = durations
assert len(bad_files) == 0, "有坏文件，先检查一下再继续"


# ---------------------------------------------------------------------
# 第 4 步 · 按演员划分 train / val / test
# ---------------------------------------------------------------------
# 同一个演员的所有录音只能在一个组里，这样测试时模型听到的全是「没见过的人」。

actors = sorted(df["actor"].unique())
np.random.seed(42)
np.random.shuffle(actors)

# 按 70% / 15% / 15% 算每组人数
n_total = len(actors)
n_val = round(n_total * 0.15)
n_test = round(n_total * 0.15)
n_train = n_total - n_val - n_test

train_actors = actors[:n_train]
val_actors = actors[n_train:n_train + n_val]
test_actors = actors[n_train + n_val:]

# 先全部设成 train（人最多），再改 val 和 test
df["split"] = "train"
df.loc[df["actor"].isin(val_actors), "split"] = "val"
df.loc[df["actor"].isin(test_actors), "split"] = "test"

# 三组名单存在一个文件里，一组一行
with open(os.path.join(SPLIT_DIR, "splits.txt"), "w") as f:
    f.write("train: " + " ".join(train_actors) + "\n")
    f.write("val: " + " ".join(val_actors) + "\n")
    f.write("test: " + " ".join(test_actors) + "\n")

# 总表存下来，LSTM 和 SVM 两个脚本都读它
df.to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)

# ---------------------------------------------------------------------
# 目前总表 data/metadata.csv 的格式
# ---------------------------------------------------------------------
#   列名        类型     说明
#   filename    字符串   文件名，不带 .wav，例：1001_DFA_ANG_XX
#   actor       字符串   演员编号，"1001" 到 "1091"
#   label       整数     情绪编号：ANG=0, DIS=1, FEA=2, HAP=3, NEU=4, SAD=5
#   duration    小数     时长（秒），大约 1.27 到 5.00
#   split       字符串   属于哪一组："train" / "val" / "test"