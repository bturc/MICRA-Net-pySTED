import os
import numpy as np
from matplotlib import pyplot as plt


data_path = os.path.expanduser("~/Documents/a22_docs/micranet_dataset_test")

img1 = np.load(data_path + "/2800_EXP201a_58_960_192.npz")
img2 = np.load(data_path + "/2800_EXP201a_58_960_384.npz")
img3 = np.load(data_path + "/2800_EXP201a_58_1152_384.npz")
img4 = np.load(data_path + "/2800_EXP201a_58_1344_384.npz")
img5 = np.load(data_path + "/2800_EXP201a_58_1536_384.npz")

fig, axes = plt.subplots(3, 3)

axes[0, 0].imshow(img1, cmap="hot")
axes[0, 1].imshow(img2, cmap="hot")
axes[0, 2].imshow(img3, cmap="hot")
axes[1, 0].imshow(img4, cmap="hot")
axes[1, 1].imshow(img5, cmap="hot")

plt.show()
