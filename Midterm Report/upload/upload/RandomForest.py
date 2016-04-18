import ReadFiles
import pandas as pd
import utils
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

# read the data that we need from the original file
trainSet, trainSetFeatures, trainSetLabels, testSet, testSetFeatures, testSetLabels = ReadFiles.readFiles()

#train the model
random_forest_regressor = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
bagging_regressor = BaggingRegressor(random_forest_regressor, n_estimators=45, max_samples=0.1, random_state=25)
bagging_regressor.fit(trainSetFeatures, trainSetLabels)

#make the prediction on the test set
predictedLabels = bagging_regressor.predict(testSetFeatures)

#output the prediction
testSetId = testSet['id']
pd.DataFrame({"id": testSetId, "relevance": predictedLabels}).to_csv('Random_Forest_Results.csv', index=False)

print "RMSE :\t", utils.getRMSE(testSetLabels, predictedLabels)
