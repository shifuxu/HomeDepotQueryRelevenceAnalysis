library(tm)
library(SnowballC)
library(stringi)
library(dplyr)
library(tidyr)
library(slam)
library(Matrix)
library(qdap)
library(hash)
library(irlba)

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

'%nin%' <- Negate('%in%')

remove_stops <- function(x){
  stopWords <- stopwords("SMART")
  return(
    unlist(
      lapply(x, function(xi) {
                                t <- unlist(strsplit(xi, " "))
                                paste(t[t %nin% stopWords], collapse=" ")
                            })))
}

stemmer <- function(x){
  return(
  unlist(
    lapply(x, function(xi) {
      t <- unlist(strsplit(xi, " "))
      paste(wordStem(t), collapse=" ")
    })))
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
  qs <- remove_stops(qs)
  print("STOPS REMOVED!")
  qs <- stemmer(qs)
  print("STEMMED!")
  qs <- remove_whiteSpaces(qs)
  print("SPACES REMOVED!")
  
  return(qs)
  
}

tf_idf_scores <- function(qs, ds){
  N.Qs <- length(qs)

  all_docs <- correct_spell(c(ds,qs))
  print("SPELLED!")
  
  dq.corpus <- Corpus(VectorSource(all_docs))
  print("CORPUS CREATED!")
  
  #writeLines(as.character(mycorpus), con="corpus-Q-TIT.txt")
  #tmDocMat <- as.TermDocumentMatrix(all_docs, grouping.var = 1:length(all_docs), vowel.check = F)
  tmDocMat <- DocumentTermMatrix(dq.corpus)
  print("TFMAT CREATED!")
  
  w_pt <- weightTfIdf(tmDocMat[1:length(ds),])
  
  #w_pt <- weightTf(tmDocMat[,1:length(ds)])
  w_bq <- weightBin(tmDocMat[(length(ds)+1):nrow(tmDocMat),])
  print("TF-IDF MATS CREATED!")
  
  #w_pt_sp <- w_pt_sp[,as.vector(values(h_duids,d_uids))]
  print("Matrices CREATED! Going to do main!")
  new_tm_doc_mat <- w_pt*w_bq
  tmsp <- sparseMatrix(new_tm_doc_mat$i,new_tm_doc_mat$j,x=new_tm_doc_mat$v,dims=dim(new_tm_doc_mat))
  print("SParsed!")
  print(dim(tmsp))
  svd <- irlba(tmsp, nv=50, right_only = T, maxit = 500)
  
  return(tmsp %*% svd$v)
  #return(colSums(w_pt_sp*w_bq_sp))
}

#######################################

train_set <- read.csv("newTrain.csv", stringsAsFactors = F)

test_set <- read.csv("newTest.csv", stringsAsFactors = F)

pr_info <- read.csv("products_info.csv", stringsAsFactors = F)

train_set <- train_set[,-2]
train_set <- merge(train_set, pr_info, by = 'product_uid', all.x = T) %>%
  arrange(id)

test_set <- test_set[,-2]
test_set <- merge(test_set, pr_info, by = 'product_uid', all.x = T) %>%
  arrange(id)

train_responses <- train_set$relevance

train_set <- train_set %>%
  select(-relevance)

df_all <- rbind(train_set,test_set)


system.time(scores_ptit <- tf_idf_scores(qs = df_all$search_term, ds = df_all$product_title))

df <- as.data.frame(as.matrix(scores_ptit))
colnames(df) <- paste("svd_tit",1:50)

prep_tr <- read.csv("train_preProcessed-V8.csv", stringsAsFactors = F)
prep_ts <- read.csv("test_preProcessed-V8.csv", stringsAsFactors = F)

prep_tr <- cbind(prep_tr,df[1:nrow(prep_tr),])
prep_ts <- cbind(prep_ts,df[(nrow(prep_tr)+1):nrow(df),])

prep_tr <- prep_tr[,c(21,1:20,22:ncol(prep_tr))]

write.csv(prep_tr, file="train_preProcessed-V9.csv", row.names = F)
write.csv(prep_ts, file="test_preProcessed-V9.csv", row.names = F)
############################################################################################
system.time(scores_pdesc <- tf_idf_scores(qs = df_all$search_term, ds = df_all$product_description))

df <- as.data.frame(as.matrix(scores_pdesc))
colnames(df) <- paste("svd_dec",1:50)

prep_tr <- read.csv("train_preProcessed-V9.csv", stringsAsFactors = F)
prep_ts <- read.csv("test_preProcessed-V9.csv", stringsAsFactors = F)

prep_tr <- cbind(prep_tr,df[1:nrow(prep_tr),])
prep_ts <- cbind(prep_ts,df[(nrow(prep_tr)+1):nrow(df),])

write.csv(prep_tr, file="train_preProcessed-V10.csv", row.names = F)
write.csv(prep_ts, file="test_preProcessed-V10.csv", row.names = F)
###############################################################################################
system.time(scores_bull <- tf_idf_scores(qs = df_all$search_term, ds = df_all$bullet_info))

df <- as.data.frame(as.matrix(scores_bull))
colnames(df) <- paste("svd_bull",1:50)

prep_tr <- read.csv("train_preProcessed-V10.csv", stringsAsFactors = F)
prep_ts <- read.csv("test_preProcessed-V10.csv", stringsAsFactors = F)

prep_tr <- cbind(prep_tr,df[1:nrow(prep_tr),])
prep_ts <- cbind(prep_ts,df[(nrow(prep_tr)+1):nrow(df),])

write.csv(prep_tr, file="train_preProcessed-V11.csv", row.names = F)
write.csv(prep_ts, file="test_preProcessed-V11.csv", row.names = F)
###############################################################################################
system.time(scores_bran <- tf_idf_scores(qs = df_all$search_term, ds = df_all$brand_info))

df <- as.data.frame(as.matrix(scores_bran))
colnames(df) <- paste("svd_bran",1:20)

prep_tr <- read.csv("train_preProcessed-V11.csv", stringsAsFactors = F)
prep_ts <- read.csv("test_preProcessed-V11.csv", stringsAsFactors = F)

prep_tr <- cbind(prep_tr,df[1:nrow(prep_tr),])
prep_ts <- cbind(prep_ts,df[(nrow(prep_tr)+1):nrow(df),])

write.csv(prep_tr, file="train_preProcessed-V12.csv", row.names = F)
write.csv(prep_ts, file="test_preProcessed-V12.csv", row.names = F)
###############################################################################################
system.time(scores_oth <- tf_idf_scores(qs = df_all$search_term, ds = df_all$other_info))

df <- as.data.frame(as.matrix(scores_oth))
colnames(df) <- paste("svd_oth",1:50)

prep_tr <- read.csv("train_preProcessed-V12.csv", stringsAsFactors = F)
prep_ts <- read.csv("test_preProcessed-V12.csv", stringsAsFactors = F)

prep_tr <- cbind(prep_tr,df[1:nrow(prep_tr),])
prep_ts <- cbind(prep_ts,df[(nrow(prep_tr)+1):nrow(df),])

write.csv(prep_tr, file="train_preProcessed-V13.csv", row.names = F)
write.csv(prep_ts, file="test_preProcessed-V13.csv", row.names = F)
