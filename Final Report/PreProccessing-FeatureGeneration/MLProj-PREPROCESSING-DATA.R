library(tm)
library(SnowballC)
library(dplyr)
library(tidyr)
library(slam)
library(Matrix)
library(qdap)
library(hash)

setwd("~\\MachineLearning\\Project\\dataset")
######################################################## READING FILES
camelCaseSplit <- function(x) {
  return(gsub("([a-z])([A-Z])", "\\1 \\L\\2", x, perl = TRUE))
  #return(gsub("([A-Z])", " \\1", x))
}

non_eng_remove <- function(x){
  return(gsub("[^[:alnum:]///' ]", " ", x))
}

remove_punctuations <- function(str){
  return(gsub("[[:punct:]]", " ", str) )
}

remove_numbers <- function(x){
  return(gsub('[[:digit:]]+', ' ', x))
}

remove_whiteSpaces <- function(x){
  return(gsub("\\s+", " ", x))
}

stem_doc <- function(x) {
  return(paste2(stemmer(x, capitalize = F, warn = F, rm.bracket = T), sep=" "))
}

correct_spell <- function(qs){
  qs <- remove_punctuations(qs)
  print("PUNCS REMOVED!")
  qs <- camelCaseSplit(qs)
  print("CAMEL REMOVED!")
  qs <- tolower(qs)
  print("LOWERED!")
  qs <- remove_numbers(qs)
  print("NUMS REMOVED!")
  qs <- non_eng_remove(qs)
  print("NONENGS REMOVED!")
  qs <- remove_whiteSpaces(qs)
  print("SPACES REMOVED!")
  #cspel <- check_spelling(qs)
  #qs <- unlist(lapply(qs, function(q) stem_doc(q)))
  #print("STEMMED!")
  
  return(qs)
  
}

tf_idf_scores <- function(qs, ds, d_uids){
  N.Qs <- length(qs)
  h_duids <- hash(unique(sort(d_uids)), 1:length(unique(sort(d_uids))))
  
  all_docs <- correct_spell(c(ds,qs))
  print("SPELLED!")
   #dq.corpus <- Corpus(VectorSource(c(ds,qs)))
  dq.corpus <- Corpus(VectorSource(all_docs))
  print("CORPUS CREATED!")
  dq.corpus <- tm_map(dq.corpus, removeWords, stopwords('english'))
  print("STOP WORDS REMOVED!")

  dq.corpus <- tm_map(dq.corpus, stemDocument)
   
  
  print("STEMMED!")
  tmDocMat <- TermDocumentMatrix(dq.corpus)
 # tmDocMat <- as.TermDocumentMatrix(all_docs, grouping.var = 1:length(all_docs), vowel.check = F, stopwords=tm::stopwords("english"))
  print("TFMAT CREATED!")
  
  w_pt <- weightTfIdf(tmDocMat[,1:length(ds)],normalize=T)
  w_bq <- weightBin(tmDocMat[,(length(ds)+1):ncol(tmDocMat)])
  print("TF-IDF MATS CREATED!")
  
  w_pt_sp <- sparseMatrix(w_pt$i,w_pt$j,x=w_pt$v,dims=dim(w_pt))
  w_bq_sp <- sparseMatrix(w_bq$i,w_bq$j,x=w_bq$v,dims=dim(w_bq))
  print("changed to SPARSE!")
  w_pt_sp <- w_pt_sp[,as.vector(values(h_duids,d_uids))]
  print("Matrices CREATED! Going to do main!")
  return(colSums(w_pt_sp*w_bq_sp))
}

char_length <- function(ds){
  return(unlist(lapply(ds, function(d)nchar(d))))
}

num_word <- function(ds){
  corpus <- Corpus(VectorSource(ds))
  corpus <- tm_map(corpus, content_transformer(removePunctuation))
  corpus <- tm_map(corpus, content_transformer(camelCaseSplit))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, content_transformer(removeNumbers))
  corpus <- tm_map(corpus, content_transformer(non_eng_remove))
  corpus <- tm_map(corpus, removeWords, stopwords('english'))
  corpus <- tm_map(corpus, content_transformer(stripWhitespace))
  corpus <- tm_map(corpus, stemDocument)
  
  print("CORPUS CREATED!")
  tmDocMat <- TermDocumentMatrix(corpus)
  print("TFMAT CREATED!")
  
  w_b <- weightBin(tmDocMat)
  print("TF-IDF MATS CREATED!")
  w_b_sp <- sparseMatrix(w_b$i,w_b$j,x=w_b$v,dims=dim(w_b))
  return(colSums(w_b_sp))
}

#######################################

train_set <- read.csv("newTrain.csv",stringsAsFactors = F)

test_set <- read.csv("newTest.csv",stringsAsFactors = F)
pr_info <- read.csv("products_info.csv", stringsAsFactors = F)

ch <- old_tr[,4]==train_set[,5]
which(ch==FALSE)


train_responses <- train_set$relevance

train_set <- train_set %>%
  select(-relevance, -product_title)
test_set <- test_set %>%
  select(-product_title)

df_all <- rbind(train_set,test_set)


system.time(scores_ptit <- tf_idf_scores(qs = df_all$search_term, ds = pr_info$product_title, d_uids = df_all$product_uid))
df_all$p_tit_tfidf <- scores_ptit

scores_pdesc <- tf_idf_scores(qs = df_all$search_term, ds = pr_info$product_description, d_uids = df_all$product_uid)
df_all$p_desc_tfidf <- scores_pdesc

scores_pbull <- tf_idf_scores(qs = df_all$search_term, ds = pr_info$bullet_info, d_uids = df_all$product_uid)
df_all$p_bullet_tfidf <- scores_pbull

scores_brand <- tf_idf_scores(qs = df_all$search_term, ds = pr_info$brand_info, d_uids = df_all$product_uid)
df_all$p_brand_tfidf <- scores_brand

scores_pother <- tf_idf_scores(qs = df_all$search_term, ds = pr_info$other_info, d_uids = df_all$product_uid)
df_all$p_other_tfidf <- scores_pother


df_all$nchar_query <- char_length(df_all$search_term)
df_all$nword_query <- num_word(df_all$search_term)


pr_info <- read.csv("product_info_preProcessed-V6.csv", stringsAsFactors = F)


#pr_info$nchar_tit <- char_length(pr_info$product_title)
#pr_info$nchar_desc<- char_length(pr_info$product_description)
#pr_info$nchar_bull <- char_length(pr_info$bullet_info)
#pr_info$nchar_bran <- char_length(pr_info$brand_info)
#pr_info$nchar_other <- char_length(pr_info$other_info)
#
#pr_info$nword_tit <- num_word(pr_info$product_title)
#pr_info$nword_desc<- num_word(pr_info$product_description)
#pr_info$nword_bull <- num_word(pr_info$bullet_info)
#pr_info$nword_bran <- num_word(pr_info$brand_info)
#pr_info$nword_other <- num_word(pr_info$other_info)


train_set <- df_all[1:nrow(train_set),]
test_set <- df_all[(nrow(train_set)+1):nrow(df_all),]

train_set <- merge(train_set, pr_info, by = 'product_uid', all.x = T) %>%
  select(-product_title, -product_description, -brand_info, -bullet_info, -other_info) %>%
  arrange(id)
train_set$relevance <- train_responses
train_set <- train_set[,c(2,1,3:ncol(train_set))]

test_set <- merge(test_set, pr_info, by = 'product_uid', all.x = T) %>%
  select(-product_title, -product_description, -brand_info, -bullet_info, -other_info) %>%
  arrange(id)
test_set <- test_set[,c(2,1,3:ncol(test_set))]

write.csv(train_set, file="train_preProcessed-V8.csv", row.names = F)
write.csv(test_set, file="test_preProcessed-V8.csv", row.names = F)
write.csv(pr_info, file="product_info_preProcessed-V6.csv", row.names = F)



