# Week 4 course assignment

## PRE-assignment file for Friction Plate prediction prepared by your Data-Science team

#LOAD NECESSARY LIBRARIES
library(rpart); library(rpart.plot)
library(randomForest)
library(e1071)
library(neuralnet)
library(gbm)
library(pROC)
library(caret)

# READ DATA

setwd(...)

friction_plate_data <- read.csv("friction_plate_case_data.csv", sep=",")[,-1] #Read the data, note that it uses ',' (comma) as separator. The first column is number of observation, which is not needed and we drop it. 

# SPLIT DATA - YOUR TASK
set.seed(0)
train_size <- 0.7
train_index <- sample(1:nrow(friction_plate_data), train_size*nrow(friction_plate_data))

# Splitting the data into 70/30 split to training data set "train" and testing data set "test"

train <- friction_plate_data[train_index,]
test <-  friction_plate_data[-train_index,]

# NORMALIZATION - To train the Neural network model properly we need to normalize the input variables to an equal range

#Usually normalization is done to [0,1] or [-1,1] intervals, min-max can be used to scale the data to intervals [0,1]
max_vals <- apply(friction_plate_data[,-10], MARGIN=2, max)
min_vals <- apply(friction_plate_data[,-10], MARGIN=2, min)

train_nums_scaled <- as.data.frame(scale(train[,-10], center= min_vals, scale = max_vals - min_vals)) #Scaled numeric values. Scale returns a matrix, which is coerced into data.frame object
test_nums_scaled <- as.data.frame(scale(test[,-10], center= min_vals, scale = max_vals - min_vals)) #Scaled numeric values. Scale returns a matrix, which is coerced into data.frame object

train_scaled <- cbind(train_nums_scaled, train[, 10]); colnames(train_scaled)[10] <- "Test.Result" #Training data scaled ,Let's combine the output column to the scaled data set 
test_scaled <- cbind(test_nums_scaled, test[, 10]); colnames(test_scaled)[10] <- "Test.Result" #Test data scaled ,Let's combine the output column to the scaled data set 

# Other models than the Neural Network will be able to use the non-scaled input data

# MODELS FROM DATA SCIENCE TEAM
# Data science team has done some model training and tuning work and suggests using these advanced models with the given parameters for the Friction plate Test-approval prediction

# RANDOM FOREST
rf_fit <- randomForest(as.factor(Test.Result)  ~ .,data=train) #Random Forest model training with all of the variables, package randomForest
rf_predictions<-predict(rf_fit, test, type = "class") #Predictions of Random Forest model
rf_predictions <- data.frame(class = predict(rf_fit, test, type = "class"), prob = predict(rf_fit, test, type = "prob")[,2]) #In predict selection "class" gives classification TRUE/FALSE prediction and "prob" gives probability of this prediction in two columns "FALSE/TRUE"

confusionMatrix(as.factor(rf_predictions$class),as.factor(test$Test.Result), positive="TRUE")
par(pty="s") #Get rid of extra padding in the ROC plot
plot.roc(as.numeric(test$Test.Result), rf_predictions$prob, legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="red", main="Random Forest ROC")

# SUPPORT VECTOR MACHINE
#Support vector classifier: Find information in the course book ISLR chapter 9.3

svm_fit <- svm(as.factor(Test.Result)  ~ var.3 + var.7 + var.1 + var.9 + var.5, #SVM was tuned and data science team selected these variables and tuning parameters for the model
               data=train, kernel="polynomial", cost=1, degree=3, coef0=2, probability=TRUE) #kernel "polynomial" for non-linear classifiers, kernel="radial" for radial kernels, and "linear" for linear kernels
svm_fit_prob <- attr(predict(svm_fit, test, probability = TRUE), "probabilities")[, 2] #To get the SVM probabilities, we need to look into attribute-property of the predict-output and call for attribute "probabilities"
svm_predictions <- data.frame(class = predict(svm_fit, test, type = "class"), prob = svm_fit_prob)  #Support vector machine predictions for the test data

confusionMatrix(as.factor(svm_predictions$class),as.factor(test$Test.Result), positive="TRUE")
plot.roc(as.numeric(test$Test.Result), svm_predictions$prob, legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="blue", main="SVM ROC")

# NEURAL NETWORK
#neuralnet package information: https://cran.r-project.org/web/packages/neuralnet/neuralnet.pdf
nn_fit <- neuralnet(as.factor(Test.Result)  ~ ., hidden=c(3,2), data=train_scaled, linear.output = FALSE) #Training of the Neural Network with the parameters chosen in the data science teams's testing
#Parameters set for the Neural Network by the Data Science team
#hidden: vector of numbers specifying number of hidden neurons in each layer to be trained
#data: training data to be used
#linear.output: if the output is going to be smoothed
nn_predictions <- data.frame(class = predict(nn_fit, test_scaled[,-10])[,1]>0.5, prob = predict(nn_fit, test_scaled[,-10])[,1])#Neural network prediction is a probability, we use the 0.5 decision criterion to translate the probabilities to TRUE/FALSE predictions
plot(nn_fit, rep = 'best')# Plot neural network

confusionMatrix(as.factor(nn_predictions$class),as.factor(test$Test.Result), positive="TRUE")
plot.roc(as.numeric(test$Test.Result), as.numeric(nn_predictions$prob), legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="darkgreen", main="Neural Network ROC")

# BOOSTED REGRESSION TREE
#Generalized boosted models, also Gradient Boosting Machines https://cran.r-project.org/web/packages/gbm/vignettes/gbm.pdf or course book ISLR 8.2.3 or check Trevor Hastie presentation on boosting and ensemble models generally: https://www.youtube.com/watch?v=wPqtzj5VZus
#These models are very flexible and require quite a lot tuning to get a good and robust model, but can be very powerful, also prone to overfitting

#Train the best boosted model:
#The following code implements the model that the data science team found to be best through grid seach tuning
boost_fit <- gbm(as.numeric(Test.Result)  ~ .,data=train, distribution = "bernoulli", n.trees=405, interaction.depth = 3, shrinkage = 0.01, n.minobsinnode=5, bag.fraction=0.65) #Note bernoulli fit for classification, use "gaussian" for regression problems
boost_predictions <- data.frame(class = predict(boost_fit, test, n.trees = 405, type="response")>0.5, prob = predict(boost_fit, test, n.trees = 405, type="response")) #Gradient boosting model predictions, we use the decision criteria of 0.5 again
summary(boost_fit) #See the relative importance of variables for the gradient boosting machine model

confusionMatrix(as.factor(boost_predictions$class),as.factor(test$Test.Result), positive="TRUE")
plot.roc(as.numeric(test$Test.Result), as.numeric(boost_predictions$prob), legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="orange", main = "Boosted Regression Tree ROC")

# YOUR PREDICTION MODELS - your task

# LOGISTIC REGRESSION

# Fitting the model and creating predictions
lr_fit <- glm(Test.Result ~ .,family=binomial(logit),data=train); summary(lr_fit)
lr_predictions <- data.frame(class = predict(lr_fit, test, type="response")>0.5, prob = predict(lr_fit,newdata=test[,-10],type='response'))
head(lr_predictions)
head(test$Test.Result)

# Evaluating the model performance
confusionMatrix(as.factor(lr_predictions$class),as.factor(test$Test.Result), positive="TRUE")
plot.roc(as.numeric(test$Test.Result), as.numeric(lr_predictions$prob), legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="black", main= "Logistic Regression ROC")

# DECISION TREE

# Let's train some Classification Trees
tree1 = rpart(Test.Result ~ . - Test.Result, method = "class", data = train, minbucket=1)
tree2 = rpart(Test.Result ~ . - Test.Result, method = "class", data = train, minbucket=5)
tree3 = rpart(Test.Result ~ . - Test.Result, method = "class", data = train, minbucket=10)

rpart.plot(tree1) 
rpart.plot(tree3)

# Prediction accuracy
treePredict1 <- data.frame(class=predict(tree1, newdata=test, type='class'), prob=predict(tree1, newdata=test[,-10], type='prob'))
treePredict2 <- data.frame(class=predict(tree2, newdata=test, type="class"), prob=predict(tree2, newdata=test[,-10], type='prob'))
treePredict3 <- data.frame(class=predict(tree3, newdata=test, type="class"), prob=predict(tree3, newdata=test[,-10], type='prob'))

# Mean accuracy rates:
mean(test$Test.Result == treePredict1)
mean(test$Test.Result == treePredict2)
mean(test$Test.Result == treePredict3)

# Confusion matrices
confusionMatrix(treePredict1$class, as.factor(test$Test.Result), positive = "TRUE")
confusionMatrix(treePredict2$class, as.factor(test$Test.Result), positive = "TRUE")
confusionMatrix(treePredict3$class, as.factor(test$Test.Result), positive = "TRUE")

# We use proc-library to plot the ROC and AUC
par(mfrow=c(1,3))
plot.roc(as.numeric(test$Test.Result), as.numeric(treePredict1$prob.TRUE), lwd=2, type="b",print.auc=TRUE,col ="red")
plot.roc(as.numeric(test$Test.Result), as.numeric(treePredict2$prob.TRUE), lwd=2, type="b",print.auc=TRUE,col ="blue")
plot.roc(as.numeric(test$Test.Result), as.numeric(treePredict3$prob.TRUE), lwd=2, type="b",print.auc=TRUE,col ="black")
par(mfrow=c(1, 1))

#Out of these trees, I choose to use number 3 for further analysis as is has the best precision
plot.roc(as.numeric(test$Test.Result), as.numeric(treePredict3$prob.TRUE), lwd=2, type="b",print.auc=TRUE,col ="purple",main="Decision Tree ROC")

# MODEL PERFORMANCE - your task
#See assignment question 1 and 2

# Drawing ROC plots
par(mfrow=c(2,3))
plot.roc(as.numeric(test$Test.Result), rf_predictions$prob, legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="red", main="Random Forest ROC")
plot.roc(as.numeric(test$Test.Result), svm_predictions$prob, legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="blue", main="SVM ROC")
plot.roc(as.numeric(test$Test.Result), as.numeric(nn_predictions$prob), legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="darkgreen", main="Neural Network ROC")
plot.roc(as.numeric(test$Test.Result), as.numeric(boost_predictions$prob), legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="orange", main = "Boosted Regression Tree ROC")
plot.roc(as.numeric(test$Test.Result), as.numeric(lr_predictions$prob), legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="black", main= "Logistic Regression ROC")
plot.roc(as.numeric(test$Test.Result), as.numeric(treePredict3$prob.TRUE),legacy.axes=TRUE,xlab="False Positive rate", ylab="True Positive rate", lwd=2, type="b",print.auc=TRUE,col ="purple",main="Decision Tree ROC")
par(mfrow=c(1, 1))

# Accuracies
accuracy <- c(sum(test$Test.Result == rf_predictions$class)/nrow(rf_predictions),
              sum(test$Test.Result == svm_predictions$class)/nrow(svm_predictions),
              sum(test$Test.Result == nn_predictions$class)/nrow(nn_predictions),
              sum(test$Test.Result == boost_predictions$class)/nrow(boost_predictions),
              sum(test$Test.Result == lr_predictions$class)/nrow(lr_predictions),
              sum(test$Test.Result == treePredict3$class)/nrow(treePredict3))

# Precisions
precision <- c(sum(rf_predictions$class == TRUE & test$Test.Result == TRUE)/sum(rf_predictions$class == TRUE),
               sum(svm_predictions$class == TRUE & test$Test.Result == TRUE)/sum(svm_predictions$class == TRUE),
               sum(nn_predictions$class == TRUE & test$Test.Result == TRUE)/sum(nn_predictions$class == TRUE),
               sum(boost_predictions$class == TRUE & test$Test.Result == TRUE)/sum(boost_predictions$class == TRUE),
               sum(lr_predictions$class == TRUE & test$Test.Result == TRUE)/sum(lr_predictions$class == TRUE),
               sum(treePredict3$class == TRUE & test$Test.Result == TRUE)/sum(treePredict3$class == TRUE))

# Recalls
recall <- c(sum(rf_predictions$class == TRUE & test$Test.Result == TRUE)/sum(test$Test.Result == TRUE),
            sum(svm_predictions$class == TRUE & test$Test.Result == TRUE)/sum(test$Test.Result == TRUE),
            sum(nn_predictions$class == TRUE & test$Test.Result == TRUE)/sum(test$Test.Result == TRUE),
            sum(boost_predictions$class == TRUE & test$Test.Result == TRUE)/sum(test$Test.Result == TRUE),
            sum(lr_predictions$class == TRUE & test$Test.Result == TRUE)/sum(test$Test.Result == TRUE),
            sum(treePredict3$class == TRUE & test$Test.Result == TRUE)/sum(test$Test.Result == TRUE))

# F1 scores
f1_scores <- c(2*recall*precision/(recall+precision))


column_names <- c("Random Forest", "SVM", "Neural Network", "Boosted Regression Tree", "Logistic Regression", "Decision Tree")
row_names <- c("Accuracy", "Precision", "Recall", "F1 Score")
eval_values <- matrix(c(accuracy, precision, recall,f1_scores),nrow = 4, byrow = TRUE)
colnames(eval_values) <- column_names
rownames(eval_values) <- row_names
eval_values.df <- as.data.frame(eval_values)
eval_values.df

# SELECTION OF RECOMMENDED MODEL - your task
#See assignment question 2

# Based on my analysis, I recommend using an SVM model to support decision-making, as it performs well most consistently of all the models. 
# Random Forest, Boosted Regression Tree and Decision Tree outperform the SVM model in some of the aspects on some runs, but are less consistent in their performance and often worse or equal in recall. 
# Due to its high precision, a Decision Tree model could also be used if high precision is desired despite lower recall.

# Further analysis and recommendations returned as a pdf memo for the course assignment.

