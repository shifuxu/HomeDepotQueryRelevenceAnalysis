from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

product_title = ("The sky is blue.", "The sun is bright.")#two product title
querry_set = ("The sun in the sky is bright.", "We can see the shining sun, the bright sun.") # two search term


#learn the vocabulary
count_vect_product = CountVectorizer(stop_words ='english')
count_vect_product.fit_transform(product_title)

#produce the frequency in vocabulary for each product title
prt_term_fq_matrix=count_vect_product.transform(product_title)


print 'vocabulary learned in product title\n',count_vect_product.vocabulary_
print "\n pro_title_term_fq_matrix\n",prt_term_fq_matrix.todense()


# get idf for each vocabulary
tfidf_transformer = TfidfTransformer(norm="l2",smooth_idf=True)
tfidf_transformer.fit(prt_term_fq_matrix)
#get the idf matrix
tf_idf_matrix=tfidf_transformer.transform(prt_term_fq_matrix)


print "\n tf-idf matrix: \n",tf_idf_matrix.todense()


#count the frequency in for each product title
count_vect_querry = CountVectorizer(stop_words ='english',binary=True)
count_vect_querry.fit(product_title)
querry_set_term_fq_matrix=count_vect_querry.transform(querry_set)
print "\nquerry_set_term_fq_matrix\n",querry_set_term_fq_matrix.todense()

tf_idf_matrix=np.transpose(tf_idf_matrix)
print "\n tf-idf matrix transpose: \n",tf_idf_matrix.todense()



print "\n result: \n",np.multiply(querry_set_term_fq_matrix,tf_idf_matrix).todense()
