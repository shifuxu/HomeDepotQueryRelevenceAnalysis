import numpy as np
from random import randint
from sklearn.metrics import mean_squared_error
from math import sqrt

def getSamples(percent, matrix):
    samples = np.shape(matrix)[0]
    nums = int(round(samples * percent))
    randIndexLst = []
    target = []

    while nums > 0:
        temp = randint(0, samples - 1)
        if temp not in randIndexLst:
            randIndexLst.append(temp)
            nums -= 1

    for index in randIndexLst:
        target.append(matrix[index])

    return np.array(target)

def getRMSE(yTest, yPredict):
    return sqrt(mean_squared_error(yTest, yPredict))