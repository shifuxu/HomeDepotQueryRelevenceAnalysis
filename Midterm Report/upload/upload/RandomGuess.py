import ReadFiles
import numpy as np
import pandas as pd
import utils

# read the data that we need from the original file
trainSet, trainSetFeatures, trainSetLabels, testSet, testSetFeatures, testSetLabels = ReadFiles.readFiles()

# make the prediction on the test set by random guess
predictedLabels = []

for index in range(testSet.shape[0]):
    predictedLabels.append(utils.getRandFloat(1, 3))

predictedLabels = np.array(predictedLabels)

#output the prediction
id_test = testSet['id']
pd.DataFrame({"id": id_test, "relevance": predictedLabels}).to_csv('Random_Guess_Results.csv', index=False)

print "RMSE :\t", utils.getRMSE(testSetLabels, predictedLabels)