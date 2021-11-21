import torch
from torch import nn
from torch.nn.modules.conv import Conv2d
class BasicCNN(nn.Module):

    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(13456, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 26)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x