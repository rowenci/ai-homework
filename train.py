import torch
import torchvision
import pyttsx3 # 语音播报
import datetime # 计时
import logConfig # logging
from torchvision import transforms
from torch.utils.data import DataLoader
from DataSets import TrainDataSet
from DataSets import TestDataSet
from torch.utils.tensorboard import SummaryWriter

from Model import ResNet
from Model import VGG
from Model import AlexNet
from Model import BasicCNN
from Model import LeNet



# 1. super parameters
batch_size = 128
lr = 0.001
epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. model
# model = ResNet.getResNet()
model = AlexNet.AlexNet()
# model = BasicCNN.BasicCNN()
# model = LeNet.LeNet()
#ratio = 8
#conv_arch = ((1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio), (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio))
#fc_features = 512 * 7 * 7
#fc_hidden_units = 4096
#model = VGG.vgg(conv_arch, fc_features // ratio, fc_hidden_units // ratio)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = torch.nn.CrossEntropyLoss()

# 3. dataset
train_dataset = TrainDataSet()
test_dataset = TestDataSet()

# 4. dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# utils
writer = SummaryWriter("tensorboardLog/alexnet")
logger = logConfig.getLogger("logs/alexnet/log.txt")

# 5. states of training
train_step = 0
test_step = 0

# 6. training
logger.info("training on {}".format(device))
for epoch in range(epochs):
    begin_time = datetime.datetime.now()
    logger.info("-------epoch {}-------".format(epoch + 1))

    # training begin
    model.train()
    train_loss = 0
    for data in train_loader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)

        # get loss
        l = loss(outputs, labels)
        train_loss += l.item()
        optimizer.zero_grad()
        # optimize the model
        l.backward()
        optimizer.step()
        train_step += 1

    logger.info("train_step : {}".format(train_step))
    logger.info("train_loss : {}".format(train_loss / len(train_loader)))
    writer.add_scalar("train_loss", train_loss / len(train_loader), epoch)
    # trainging end

    # testing begin
    model.eval()
    test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            l = loss(outputs, labels)
            test_loss += l.item()
            accuracy = (outputs.argmax(1) == labels).sum() / len(outputs)
            total_accuracy += accuracy
    # testing end

    # results in one epoch
    logger.info("test_loss is : {}".format(test_loss / len(test_loader)))
    logger.info("total_accuracy is {}".format(total_accuracy / len(test_loader)))
    writer.add_scalar("test_loss", test_loss / len(test_loader), test_step)
    writer.add_scalar("test_accuracy", total_accuracy / len(test_loader), test_step)
    test_step += 1


    # save model in every epoch
    torch.save(model, "Model/trained_models/alexnet/trained_alexnet{}.pth".format(epoch + 1))
    logger.info("model has been saved")
    end_time = datetime.datetime.now()
    cost_time = (end_time - begin_time).seconds
    logger.info("time cost : {} seconds".format(cost_time))

# finish
writer.close()

# 训练完成提示
engine = pyttsx3.init() 
volume = engine.getProperty('volume')
engine.setProperty('volume', 1)
engine.say('训练完成，训练完成，训练完成')
# 等待语音播报完毕 
engine.runAndWait()
