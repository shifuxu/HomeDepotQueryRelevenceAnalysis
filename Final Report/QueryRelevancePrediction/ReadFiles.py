import numpy as np
import pandas as pd

def readFiles():
 trainSet = pd.read_csv('IOFolder\subTrainSet-pv13.csv', encoding="ISO-8859-1")
 testSet = pd.read_csv('IOFolder\developmentSet-pv13.csv', encoding="ISO-8859-1")

 trainSet = trainSet.drop(['product_uid', 'search_term'], axis=1)
 testSet = testSet.drop(['product_uid', 'search_term'], axis=1)

 trainSetFeatures = trainSet.drop(['id', 'relevance'], axis=1).values #id and relevance is not features to use
 trainSetLabels = trainSet['relevance'].values

 testSetFeatures = testSet.drop(['id','relevance'], axis=1).values
 testSetLabels = testSet['relevance'].values

 return trainSet, trainSetFeatures, trainSetLabels, testSet, testSetFeatures,testSetLabels







