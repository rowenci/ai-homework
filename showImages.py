# 1016 812 204
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboardLog')

images = np.load("Data/testingData.npy")
writer.add_images("img", images)