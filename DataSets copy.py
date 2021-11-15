import pandas as pd
import torch.utils.data


class TrainDataSet(torch.utils.data.Dataset):

    def __init__(self, path):
        # get x and y in training set
        train_data = pd.read_csv(path)
        x_train = train_data.drop(["label"], axis=1)
        y_train = train_data['label']
        # normalizing the data
        x_train = x_train.astype('float32')/255
        y_train = y_train.astype('float32')
        # converting the data to numpy array
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        # reshaping the array
        x_train = x_train.reshape(-1, 1, 28, 28)

        self.x = torch.from_numpy(x_train)
        self.y = torch.from_numpy(y_train)
        self.y = self.y.long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data = (self.x[index], self.y[index])
        return data


class TestDataSet(torch.utils.data.Dataset):

    def __init__(self, path):
        x_test = pd.read_csv(path)
        x_test = x_test.astype('float32')/255
        x_test = x_test.to_numpy()
        x_test = x_test.reshape(-1, 1, 28, 28)

        self.x = torch.from_numpy(x_test)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data = self.x[index]
        return data
