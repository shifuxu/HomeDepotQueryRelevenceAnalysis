from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
def stem_process(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])


#read file
#randomly selected about 80% of the training data from train.csv
subTrainSet=pd.read_csv('subTrainSet.csv', encoding="ISO-8859-1")
#randomly selected about 20% of the training data from train.csv
developmentTestSet=pd.read_csv('developmentTestSet.csv', encoding="ISO-8859-1")
#testSet=pd.read_csv('test.csv', encoding="ISO-8859-1")# original test set


AllSet = pd.concat((subTrainSet, developmentTestSet), axis=0, ignore_index=True)

product_description_data = pd.read_csv('product_descriptions.csv')
AllSet = pd.merge(AllSet, product_description_data, how='left', on='product_uid')

AllSet['search_term'] = AllSet['search_term'].map(lambda x: stem_process(x))
AllSet['product_title'] = AllSet['product_title'].map(lambda x: stem_process(x))
AllSet['product_description'] = AllSet['product_description'].map(lambda x: stem_process(x))

product_title = AllSet['product_title']
product_description=AllSet['product_description']
search_term = AllSet['search_term']


#PARTI compute the tf-idf for product title

search_vect_tittle = CountVectorizer(stop_words='english', binary=True)# use binary value to indicate the frequency
search_vect_tittle.fit(product_title)#learn the vocabulary
search_tittle_fq_matrix = search_vect_tittle.transform(search_term) #get the (vocabulary)-(search term) frequency matrix


title_vect = CountVectorizer(stop_words='english')
title_vect.fit_transform(product_title)#learn the vocabulary
title_fq_matrix = title_vect.transform(product_title) #get the (vocabulary)-(product_title) frequency matrix


tfidf_transformer = TfidfTransformer(norm="l2", smooth_idf=True)
tfidf_transformer.fit(title_fq_matrix) # get idf for each vocabulary
tf_idf_title_matrix = tfidf_transformer.transform(title_fq_matrix) #get the idf matrix


tf_idf_title_result = [] #compute the result of tf-idf for product title
for index in range(tf_idf_title_matrix.shape[0]):
    tf_idf_title_result.append((np.multiply(tf_idf_title_matrix[index], search_tittle_fq_matrix[index].transpose()))[0, 0])

pd.DataFrame({"id": AllSet['id'],"product_title_tfidf": tf_idf_title_result}).to_csv('product_title_tfidf.csv', index=False)

#PARTII compute the tf-idf for product description
search_vect_descrip = CountVectorizer(stop_words='english', binary=True)# use binary value to indicate the frequency
search_vect_descrip.fit(product_description)#learn the vocabulary
search_descrip_fq_matrix = search_vect_descrip.transform(search_term) #get the (vocabulary)-(search term) frequency matrix


description_vect = CountVectorizer(stop_words ='english')
description_vect.fit_transform(product_description)#learn the vocabulary
description_fq_matrix=description_vect.transform(product_description) #get the (vocabulary)-(product_description) frequency matrix


tfidf_transformer = TfidfTransformer(norm="l2",smooth_idf=True)
tfidf_transformer.fit(description_fq_matrix) # get idf for each vocabulary
tf_idf_descrip_matrix  = tfidf_transformer.transform(description_fq_matrix) #get the idf matrix



tf_idf_descrip_result=[]#compute the result of tf-idf for product title
for index in range(tf_idf_descrip_matrix.shape[0]):
    tf_idf_descrip_result.append((np.multiply(tf_idf_descrip_matrix[index], search_descrip_fq_matrix[index].transpose()))[0, 0])

pd.DataFrame({"id":AllSet['id'],"product_description_tfidf": tf_idf_descrip_result}).to_csv('product_description_tfidf.csv', index=False)