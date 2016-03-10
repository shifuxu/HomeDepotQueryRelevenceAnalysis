import numpy as np
import pandas as pd

def readFiles():
    #randomly selected about 80% of the training data from train.csv
    subTrainSet = pd.read_csv('../../myDataset/subTrainSet.csv', encoding="ISO-8859-1")
    #randomly selected about 20% of the training data from train.csv
    developmentTestSet = pd.read_csv('../../myDataset/developmentTestSet.csv', encoding="ISO-8859-1")

    #read the preprocessed features from file
    product_description_tfidf = pd.read_csv('../../myDataset/product_description_tfidf.csv')
    product_title_tfidf = pd.read_csv('../../myDataset/product_title_tfidf.csv')


    subTrainSetSize = subTrainSet.shape[0]
    developmentTestSetSize = developmentTestSet.shape[0]

    AllSet = pd.concat((subTrainSet, developmentTestSet), axis=0, ignore_index=True)

    # add the features product_title_tfidf, product_description_tfidf
    AllSet = pd.merge(AllSet, product_title_tfidf, how='left', on='id')
    AllSet = pd.merge(AllSet, product_description_tfidf, how='left', on='id')

    #add another feature for prediction
    AllSet['search_words_nums'] = AllSet['search_term'].map(lambda x: len(x.split())).astype(np.int64)

    #drop columns not used for prediction
    AllSet = AllSet.drop(['product_title', 'search_term'], axis=1)

    #use the subtrain set as the training set
    trainSet = AllSet.iloc[:subTrainSetSize]
    trainSetFeatures = trainSet.drop(['id', 'relevance'], axis=1).values #id and relevance is not features to use
    trainSetLabels = trainSet['relevance'].values

    #define the development test set as our test set
    testSet = AllSet.iloc[subTrainSetSize:]
    testSetFeatures = testSet.drop(['id', 'relevance'], axis=1).values
    testSetLabels = testSet['relevance'].values

    return trainSet, trainSetFeatures, trainSetLabels, testSet, testSetFeatures, testSetLabels
