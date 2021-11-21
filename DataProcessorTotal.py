import numpy as np


training_sum = np.zeros((812, 1, 128, 128))
testing_sum = np.zeros((204, 1, 128, 128))
for i in range(26):
    images = np.load("Data/" + "TrainingData/" + str(i + 1) +".npy")
    if i == 0:
        training_sum = images
    else:
        training_sum = np.append(training_sum, images, axis=0)
np.save("Data/trainingData.npy", training_sum)

for j in range(26):
    images = np.load("Data/" + "TestingData/" + str(j + 1) +".npy")
    if j == 0:
        testing_sum = images
    else:
        testing_sum = np.append(testing_sum, images, axis=0)
np.save("Data/testingData.npy", testing_sum)

