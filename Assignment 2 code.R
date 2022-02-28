# Week 2 course assignment

# Importing and aggregating data
cdata = read.csv("customer_satisfaction.csv")
View(cdata)

agg = aggregate(cdata, by = list(cdata$company_v), FUN = mean)
View(agg)

# Data for top 7 companies
top7 <- c("AMAZON", "CAROUSELL", "EBAY", "QOO10", "TAOBAO/TMALL", "ZALORA", "FAVE")
cdata_top7 = cdata[cdata$company_v %in% top7,]
agg_top7 = aggregate(cdata_top7, by = list(cdata_top7$company_v), FUN = mean)
View(agg_top7)
View(cdata_top7)

# Data on chosen individual companies
cdata_Amazon = cdata[cdata$company_v == "AMAZON",]
View(cdata_Amazon)
cdata_Qoo10 = cdata[cdata$company_v == "QOO10",]
View(cdata_Qoo10)

# Regression analysis on factors affecting customer satisfaction in the industry
lmVariables = lm(satis ~ TP01 + TP02 + TP03 + TP04 + TP05 + TP06 + TP07 + TP08 + TP09 + TP10 + TP11 + TP12 + TP13 + TP14 + TP15 + TP16 + TP17 + TP18 + TP19, data = cdata)
summary(lmVariables)

# Regression analysis on factors affecting customer satisfaction in the top 7 companies
lmVariables_top7 = lm(satis ~ TP01 + TP02 + TP03 + TP04 + TP05 + TP06 + TP07 + TP08 + TP09 + TP10 + TP11 + TP12 + TP13 + TP14 + TP15 + TP16 + TP17 + TP18 + TP19, data = cdata_top7)
summary(lmVariables_top7)

# Regression analysis for Amazon
lmAmazon = lm(satis ~ TP01 + TP02 + TP03 + TP04 + TP05 + TP06 + TP07 + TP08 + TP09 + TP10 + TP11 + TP12 + TP13 + TP14 + TP15 + TP16 + TP17 + TP18 + TP19, data = cdata_Amazon)
summary(lmAmazon)

# Regression analysis for Qoo10
lmQoo10 = lm(satis ~ TP01 + TP02 + TP03 + TP04 + TP05 + TP06 + TP07 + TP08 + TP09 + TP10 + TP11 + TP12 + TP13 + TP14 + TP15 + TP16 + TP17 + TP18 + TP19, data = cdata_Qoo10)
summary(lmQoo10)
