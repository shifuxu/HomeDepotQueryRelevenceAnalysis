library(dplyr)
library(tidyr)


setwd("~\\MachineLearning\\Project\\dataset")
######################################################## READING FILES
train_set <- read.csv("train.csv", stringsAsFactors = F)
test_set <- read.csv("test.csv", stringsAsFactors = F)
pr_descs <- read.csv("product_descriptions.csv", stringsAsFactors = F)
pr_attrs <- read.csv("attributes.csv", stringsAsFactors = F)

######################################################## PREPROCESSING ATTRIBUTES FILE
pr_attr_puid <- distinct(pr_attrs, product_uid) %>%
  select(product_uid)

pr_bullet_info <- pr_attrs %>%
  group_by(product_uid) %>%
  filter(grepl("Bullet*",name)) %>%
  summarize(bullet_info=paste(value,collapse=" "))

pr_brand_info <- pr_attrs %>%
  group_by(product_uid) %>%
  filter(grepl("MFG*",name)) %>%
  summarize(brand_info=value)

pr_other_info <- pr_attrs %>%
  group_by(product_uid) %>%
  filter(!grepl("MFG*",name), !grepl("Bullet*",name)) %>%
  summarize(other_info=paste(value,collapse=" "))

pr_attrs_processed <- Reduce(function(x, y) merge(x, y, all=TRUE), list(pr_attr_puid, pr_bullet_info, pr_brand_info, pr_other_info))

rm("pr_other_info","pr_brand_info","pr_bullet_info","pr_attr_puid","pr_attrs")

######################################################## MERGING ALL FILE

pr_descs_attrs <- merge(x = pr_descs, y = pr_attrs_processed, by = "product_uid", all.x = TRUE)
pr_descs_attrs[is.na(pr_descs_attrs)] <- " "

rm("pr_descs","pr_attrs_processed")

df_all <- rbind(train_set[,-ncol(train_set)],test_set)

pr_tits <- distinct(df_all, product_uid) %>%
  select(product_uid, product_title)

pr_all_info <- merge(x = pr_descs_attrs, y = pr_tits, by = "product_uid", all.x = TRUE)
pr_all_info[is.na(pr_all_info)] <- " "

rm("pr_tits","pr_descs_attrs","df_all")

pr_all_info <- pr_all_info[,c(1,6,2,4,3,5)]

write.csv(pr_all_info, file="products_info.csv", row.names = F)

