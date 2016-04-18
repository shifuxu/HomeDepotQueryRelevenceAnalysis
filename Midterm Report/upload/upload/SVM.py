import ReadFiles
import pandas as pd
import utils
from sklearn import svm

# read the data that we need from the original file
trainSet, trainSetFeatures, trainSetLabels, testSet, testSetFeatures, testSetLabels = ReadFiles.readFiles()

# build a svm trainning model
svmModel = svm.SVR(C=1.0, kernel="rbf", tol=0.001)

# start trainning
svmModel.fit(trainSetFeatures, trainSetLabels)

# make predictions
predictedLabels = svmModel.predict(testSetFeatures)

# output the prediction
id_test = testSet['id']
pd.DataFrame({"id": id_test, "relevance": predictedLabels}).to_csv('SVM_Results.csv', index=False)

print "RMSE :\t", utils.getRMSE(testSetLabels, predictedLabels)
