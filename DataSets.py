import numpy as np
import torch.utils.data


class TrainDataSet(torch.utils.data.Dataset):

    def __init__(self):
        x_train = np.load("Data/trainingData.npy")
        y_train = np.load("Data/trainingLabels.npy")

        self.x = torch.from_numpy(x_train).float()
        self.y = torch.from_numpy(y_train).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data = (self.x[index], self.y[index])
        return data


class TestDataSet(torch.utils.data.Dataset):

    def __init__(self):
        x_test = np.load("Data/testingData.npy")
        y_test = np.load("Data/testingLabels.npy")
        self.x = torch.from_numpy(x_test).float()
        self.y = torch.from_numpy(y_test)
        self.y = self.y.long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data = (self.x[index], self.y[index])
        return data
