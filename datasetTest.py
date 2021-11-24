from torch.utils.data import DataLoader
from DataSets import TrainDataSet
from DataSets import TestDataSet
import torch
from Model import AlexNet

train_dataset = TrainDataSet()
test_dataset = TestDataSet()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load("Model/trained_models/alexnet/trained_alexnet1.pth")
model = model.to(device)
for data in train_loader:
    imgs, labels = data
    print(imgs)
    imgs, labels = imgs.to(device), labels.to(device)
    print(model(imgs))
    print(labels)
    break
