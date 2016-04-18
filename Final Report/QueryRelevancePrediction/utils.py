from random import randint
from sklearn.metrics import mean_squared_error , mean_absolute_error
from math import sqrt

# cal the rmse between the actual label and predict label
def getRMSE(yTest, yPredict):
    return sqrt(mean_squared_error(yTest, yPredict))

# get the rand float between start and end
def getRandFloat(start, end):
    return float(randint(start * 100, end * 100 + 1)) / 100

def getMAE(yTest, yPredict):
    return mean_absolute_error(yTest, yPredict)