from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Model import ResNet
from Model import VGG
from Model import AlexNet
from Model import BasicCNN
from Model import LeNet


transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.ToTensor()])
train_data = ImageFolder(root="Data_new/train",transform=transform)
train_loader =DataLoader(dataset=train_data,batch_size=128,
                         shuffle=True)
test_data = ImageFolder(root="Data_new/test",transform=transform)
test_loader =DataLoader(dataset=test_data,batch_size=128,
                         shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load("Model/trained_models/alexnet/trained_alexnet1.pth")
# model = ResNet.getResNet()
model = model.to(device)
# model = BasicCNN.BasicCNN()
# model = LeNet.LeNet()

for data in test_loader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        print(outputs)
        print(labels)
        break