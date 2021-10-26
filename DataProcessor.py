from logging import root
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import os
import logConfig # logging


logger = logConfig.getLogger("logs/imageProcessing.log")

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

for path in dir_list:
    dataList = []
    for fileName in file_list[categoryIndex]:
        dataList.append(path + "\\" + fileName)
    data.append(dataList)


categoryIndex = 0
asc = 65
for i in range(26):
    images = np.zeros((1016, 1, 128, 128))
    for file_path in data[i]:
        img_path = "D:\\Codes\\AI\\ai-homework\\datas_original\\Sample011\\img011-00129.png"
        img = Image.open(img_path)
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img[np.newaxis, :]
        images[i, :, :, :] = img
    np.save("Data/" + chr(asc + i) +".npy", images)


"""
img_path = "D:\\Codes\\AI\\ai-homework\\datas_original\\Sample011\\img011-00129.png"
img = Image.open(img_path)
img = np.array(img)
img = torch.from_numpy(img)
img = img[np.newaxis, :]
"""