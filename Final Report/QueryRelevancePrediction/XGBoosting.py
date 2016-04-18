import ReadFiles
import pandas as pd
import utils
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# read the data that we need from the original file
trainSet, trainSetFeatures, trainSetLabels, testSet, testSetFeatures, testSetLabels = ReadFiles.readFiles()

#train the model
# random_forest_regressor = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = GradientBoostingRegressor(warm_start=True, n_estimators=45, max_depth=6, random_state=25)
clf.fit(trainSetFeatures, trainSetLabels)

#make the prediction on the test set
predictedLabels = clf.predict(testSetFeatures)

#output the prediction
id_test = testSet['id']
pd.DataFrame({"id": id_test, "relevance": predictedLabels}).to_csv('IOFolder/XGBoosting_Results.csv', index=False)

print "RMSE :\t", utils.getRMSE(testSetLabels, predictedLabels)
print "MAE :\t", utils.getMAE(testSetLabels, predictedLabels)
