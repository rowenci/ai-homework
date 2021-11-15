import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboardLog')

images_sum = np.zeros((812 * 26, 1, 128, 128))
for i in range(26):
    images = np.load("Data/" + "TrainingData/" + str(i + 1) +".npy")
    for k in range(812):
        images_sum[k + 812 * i, :, :, :] = images[k, :, :, :]
writer.add_image("img", images_sum[1])