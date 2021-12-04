from PIL import Image
import numpy as np

img = Image.open("D:\Codes\AI\\ai-homework\datas_original\Sample011\img011-00001.png")
img = np.array(img)
np.set_printoptions(threshold = np.inf)
print(img)