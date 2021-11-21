from DataSets import TrainDataSet
from torch.utils.data import DataLoader

dataloader = TrainDataSet()
train_loader = DataLoader(dataloader, batch_size=256, shuffle=True)
for data in train_loader:
        imgs, labels = data
        break
print(imgs)
print(labels)