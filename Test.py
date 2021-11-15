from PIL import Image
import numpy as np
from numpy.core.fromnumeric import shape
from torch.utils.tensorboard import SummaryWriter
import torch
import os

writer = SummaryWriter("tensorboardLog")
dir_list = []
file_list = []

def file_name(file_dir):
    i = 0
    for root, dirs, files in os.walk(file_dir):
        if i == 0 :
            i += 1
            continue
        dir_list.append(root)
        file_list.append(files)
        


file_name("D:\\Codes\\AI\\ai-homework\\datas_original")

categoryIndex = 0
data = []

i = 1
for path in dir_list:
    dataList = []
    for fileName in file_list[categoryIndex]:
        fileName = fileName.replace("img011", "img0" + str(10 + i))
        dataList.append(path + "\\" + fileName)
    i += 1
    data.append(dataList)

j = 0
for i in range(26):
    for img_path in data[i]:
        img = Image.open(img_path)
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img[np.newaxis, :]
        writer.add_image("img" + str(j), img)
        j += 1