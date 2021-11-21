import numpy as np

trainingLabels = np.zeros((812 * 26))
testingLabels = np.zeros((204 * 26))

for i in range(26):
    for j in range(812):
        trainingLabels[(i + 1) * j] = i

for i in range(26):
    for j in range(204):
        testingLabels[(i + 1) * j] = i

print(trainingLabels.shape)
print(testingLabels.shape)
np.save("Data/trainingLabels.npy", trainingLabels)
np.save("Data/testingLabels.npy", testingLabels)