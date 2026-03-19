# 新建check_npy.py，放在dataset/目录下执行
import numpy as np

data = np.load("crop_data/clients/train/client_0.npy", allow_pickle=True).item()
print("npy文件里的key：", list(data.keys()))  # 输出比如 ['imgs', 'labels', 'annot']