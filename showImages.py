# 1016 812 204
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboardLog')

images = np.load("Data/ShuffledData/" + str(1) +".npy")
writer.add_images("img", images)