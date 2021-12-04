import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("tensorboardLog")
imgs = np.load("Data/ShuffledData/1.npy")
np.set_printoptions(threshold=np.inf)

writer.add_image("img", imgs[0])
print(imgs[0])
print(imgs[0].shape)