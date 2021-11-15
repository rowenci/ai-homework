import numpy as np
import os
import torch
import logConfig # logging


logger = logConfig.getLogger("logs/images.log")

asc = 65
images_train = np.zeros((1, 1, 128, 128))
images_test = np.zeros((1, 1, 128, 128))
for i in range(26):
    images = np.load("Data/" + "ShuffledData/" + str(i + 1) +".npy")
    for j in range(812):
        if j == 0:
            images_train[0, :, :, :] = images[j]
        else:
            images = images[np.newaxis, :]
            images_train = np.append(images_train, images[j], axis=0)
    for k in range(204):
        if k == 0:
            images_test[0, :, :, :] = images[k + 812]
        else:
            images = images[np.newaxis, :]
            images_test = np.append(images_test, images[k + 812], axis=0)
    np.save("Data/TrainingData/" + str(i + 1) +".npy", images_train)
    np.save("Data/TestingData/" + str(i + 1) +".npy", images_test)

"""
bug
"""