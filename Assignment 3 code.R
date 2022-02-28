# Week 3 course assignment

library(caret)

# Loading train and test sets
setwd(...)
hr_train <- read.csv("hr_train.csv")
View(hr_train)
hr_test <- read.csv("hr_test.csv")
View(hr_test)

# Converting columns to factors
hr_train$Attrition <- as.factor(hr_train$Attrition)
hr_train$Department <- as.factor(hr_train$Department)
hr_train$EducationField <- as.factor(hr_train$EducationField)
hr_train$Gender <- as.factor(hr_train$Gender)
hr_train$JobRole <- as.factor(hr_train$JobRole)
hr_train$OverTime <- as.factor(hr_train$OverTime)

# Fitting a model and making predictions
fit_attrition = glm(Attrition ~ . ,family=binomial(logit),data=hr_train); summary(fit_attrition)
predictions_attrition <- predict(fit_attrition,newdata=hr_test[,-2],type='response')
head(predictions_attrition)
head(hr_test$Attrition)

# Creating a confusion matrix
confusionMatrix(as.factor(predictions_attrition>.5),as.factor(hr_test$Attrition=="Yes"))


