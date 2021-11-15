import numpy as np
import os
import torch
import logConfig # logging


logger = logConfig.getLogger("logs/images.log")

asc = 65
for i in range(26):
    images = np.load("Data/" + "OriginData/" + str(i + 1) +".npy")
    np.random.shuffle(images)
    np.save("Data/" + "ShuffledData/" + str(i + 1) +".npy", images)
    # np.save("Data/" + chr(asc + i) +".npy", images)

