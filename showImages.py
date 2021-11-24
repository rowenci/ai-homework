# 1016 812 204
import numpy as np
from torch.utils.data import DataLoader
from DataSets import TrainDataSet
from DataSets import TestDataSet

from torch.utils.tensorboard import SummaryWriter, writer
writer = SummaryWriter('tensorboardLog')


train_dataset = TrainDataSet()
test_dataset = TestDataSet()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
for data in train_loader:
    imgs, labels = data
    writer.add_images("img", imgs)
    break