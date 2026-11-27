# =====================================================================
#  按顺序运行三个预处理脚本：python preprocessing/run_all.py
#  前一个出错就停下，不会接着跑下一个。
# =====================================================================

import os
import sys
import subprocess

PRE_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    "step1_download_split.py",
    "step2_lstm_features.py",
    "step3_svm_features.py",
]

for script in scripts:
    subprocess.run([sys.executable, os.path.join(PRE_DIR, script)], check=True)
