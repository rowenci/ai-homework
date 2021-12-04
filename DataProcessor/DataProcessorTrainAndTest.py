import numpy as np
import os
import torch

images_train = np.zeros((1, 1, 128, 128))
images_test = np.zeros((1, 1, 128, 128))
for i in range(26):
    images = np.load("Data/" + "ShuffledData/" + str(i + 1) +".npy")
    for j in range(812):
        if j == 0:
            add_img = images[j]
            add_img = add_img[np.newaxis, :]
            images_train = add_img
        else:
            add_img = images[j]
            add_img = add_img[np.newaxis, :]
            images_train = np.append(images_train, add_img, axis=0)
    for k in range(204):
        if k == 0:
            add_img = images[k + 812]
            add_img = add_img[np.newaxis, :]
            images_test = add_img
        else:
            add_img = images[k + 812]
            add_img = add_img[np.newaxis, :]
            images_test = np.append(images_test, add_img, axis=0)
    np.save("Data/TrainingData/" + str(i + 1) +".npy", images_train)
    np.save("Data/TestingData/" + str(i + 1) +".npy", images_test)

"""
bug
"""