from PIL import Image
import numpy as np
from numpy.core.fromnumeric import shape
from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter("tensorboardLog")
img_path = "D:\\Codes\\AI\\ai-homework\\datas_original\\Sample011\\img011-00129.png"
img = Image.open(img_path)
img = np.array(img)
img = torch.from_numpy(img)
img = img[np.newaxis, :]
writer.add_image("img", img)
print(shape(img))