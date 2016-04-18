library(h2o)
library(dplyr)
library(leaps)
library(Metrics)

H2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)

## Start a local cluster with 2GB RAM
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)

setwd("~/MachineLearning/Project/dataset")
test <- read.csv(file = "test_preProcessed-V13.csv", stringsAsFactors = F)
train <- read.csv(file = "train_preProcessed-V13.csv", stringsAsFactors = F)

##
train <- read.csv(file = "subTrainSet-pv13.csv", stringsAsFactors = F)
dev <- read.csv(file="developmentSet-pv13.csv", stringsAsFactors = F)
##

subTrain <- sample_frac(train,0.7)

tmp <- anti_join(train,subTrain, by='id')

subDev <- sample_frac(tmp, 0.4)

subTest <-  anti_join(tmp,subDev, by='id')

#write.csv(subTrain, file="train-v7.csv", row.names = F)
#write.csv(subTest, file="development-v7.csv", row.names = F)

tr <- subTrain[,c(1,5:ncol(subTrain))]
##
tr <-train[,c(1,5:ncol(train))]
ts_in <- dev[,5:ncol(dev)]
ts_out <- dev[,1]
##

tr_new <- tr[1]
tr_new <- cbind(tr_new, pca$x[,1:5])

td <- subDev[,c(1,5:ncol(subDev))]

ts_in <- subTest[,5:ncol(subTest)]
pca_ts <- prcomp(ts_in)
ts_in_new <- pca_ts$x[,1:5]
ts_out <- subTest[,1]


tr_h2o <- as.h2o(tr, 'tr')
td_h2o <- as.h2o(td, 'td')
ts_h2o <- as.h2o(ts_in, 'ts_in')

model <- h2o.deeplearning(x = 2:238,  # column numbers for predictors
                          y = 1,   # column number for label
                          training_frame = tr_h2o, # data in H2O format
                          #validation_frame = td_h2o, #validation frame
                          seed = 1123,
                          activation = "Tanh", # or 'Tanh'
                          max_runtime_secs = 0,
                          variable_importances = TRUE,
                          input_dropout_ratio = 0.2, # % of inputs dropout
                          hidden_dropout_ratios = rep(0.5,20), # % for nodes dropout
                          hidden = rep(20,20), # three layers of 50 nodes
                          epochs = 50) # max. no. of epochs

h2o_yhat_test <- h2o.predict(model, ts_h2o)

df_yhat_test <- as.data.frame(h2o_yhat_test)

#ts_pr <- exp(sqrt(df_yhat_test[,1]))
ts_pr <- df_yhat_test[,1]

err_rf <- rmse(ts_out,ts_pr)
err_mae <- mae(ts_out,ts_pr)
#rectifier: 0.4926
#max_out:   0.4924; (h:200 ep:150, 0.4917); (h:300 ep:150, 0.4924);; (h:220 ep:150, 0.4927)
#

##################lm
tr_lm <- rbind(tr,td) %>%
  select(-nchar_tit,-nchar_bull)

#tr_lm$lrelevance <- (log(tr_lm$relevance))^2

m <- glm(relevance~.,tr_lm, family = poisson())
summary(m)

lm_pr <- predict(m,ts_in)
#lm_pr <- exp(sqrt(lm_pr))
lm_err_rf<- rmse(ts_out,lm_pr)


submission <- test %>%
  select(id)

submission$relevance <- as.numeric(df_yhat_test[,1])

write.csv(submission, file="submission-v11.csv", row.names = F)

###############################
results <- read.csv("Project-Results.csv", stringsAsFactors = F)
library(ggplot2)
library(reshape2)
library(dplyr)

results <- results %>%
  arrange(RMSE)

d <- melt(results, id.vars="Model")

ggplot(d, aes(Model,value, col=variable, group=variable)) + 
  geom_point(size=4) + geom_line(size=2) 

