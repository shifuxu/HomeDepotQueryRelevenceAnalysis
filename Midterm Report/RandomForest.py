import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def getRMSE(yTest, yPredict):
    return sqrt(mean_squared_error(yTest, yPredict))

#randomly selected about 80% of the training data from train.csv
subTrainSet=pd.read_csv('subTrainSet.csv', encoding="ISO-8859-1")
#randomly selected about 20% of the training data from train.csv
developmentTestSet=pd.read_csv('developmentTestSet.csv', encoding="ISO-8859-1")

#read the preprocessed features from file
product_description_tfidf=pd.read_csv('product_description_tfidf.csv')
product_tittle_tfidf=pd.read_csv('product_title_tfidf.csv')


subTrainSetSize=subTrainSet.shape[0]


AllSet = pd.concat((subTrainSet, developmentTestSet), axis=0, ignore_index=True)

# add the features product_tittle_tfidf, product_description_tfidf
AllSet = pd.merge(AllSet, product_tittle_tfidf, how='left', on='id')
AllSet = pd.merge(AllSet, product_description_tfidf, how='left', on='id')

#add another feature for prediction
AllSet['search_words_nums'] = AllSet['search_term'].map(lambda x:len(x.split())).astype(np.int64)

#drop columns not used for prediction
AllSet = AllSet.drop(['product_title','search_term'], axis=1)

#use the subtrain set as the training set
trainSet = AllSet.iloc[:subTrainSetSize]
trainSetFeatures = trainSet.drop(['id', 'relevance'], axis=1).values #id and relevance is not features to use
trainSetLabels = trainSet['relevance'].values

#train the model
random_forest_regressor = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(random_forest_regressor, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(trainSetFeatures, trainSetLabels)

#define the development test set as our test set
testSet = AllSet.iloc[subTrainSetSize:]
testSetFeatures = testSet.drop(['id', 'relevance'], axis=1).values
testSetLabels = testSet['relevance'].values

#make the prediction on the test set
predictedLabels = clf.predict(testSetFeatures)


#output the prediction
id_test = testSet['id']
pd.DataFrame({"id": id_test, "relevance": predictedLabels}).to_csv('prediction_for_developmentTestSet.csv', index=False)

print "RMSE :\t",getRMSE(testSetLabels,predictedLabels)
