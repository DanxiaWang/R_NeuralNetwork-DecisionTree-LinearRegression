setwd("G:/UMB/MSIS 672/Final Project")
save.image("FinalProject_Group")

###########################################################################
## Import Libraries
###########################################################################

# Import Decision Tree Library
library(rpart) 
library(rpart.plot)
library(RColorBrewer)
library(caret)
# Import Additional Models
library(adabag)
library(randomForest)
# Import Neural Network Library
library(neuralnet)
library(nnet)
library(e1071)
# Import ggplot2
library(ggplot2)
# Import gain
library(gains)
library(pROC)
# Import other functions
library(forecast)
library(dplyr)

###########################################################################
## Prepare Data
###########################################################################

# Import Data
fundraising.df <- read.csv("Fundraising.csv")

# Check the data: values, types, distributions
str(fundraising.df)
head(fundraising.df)
summary(fundraising.df)
apply (fundraising.df, 2, class)
apply(is.na(fundraising.df),2,sum) #sum NA values for each column

# Prepare the data
fundraising.df <- fundraising.df[,3:23]

# Set 60% training set and 40 validation set
set.seed(1)
training <- sample(c(1:dim(fundraising.df)[1]), dim(fundraising.df)[1]*0.6)
train.df <- fundraising.df[training,]
valid.df <- fundraising.df[-training,]

###########################################################################
## Decision Tree Model
###########################################################################


# Decision tree model
fundraising.ct <- rpart(TARGET_B ~ ., data = train.df, method = "class")

fundraising.deeper.ct <- rpart(TARGET_B ~ ., data = train.df, method = "class", cp = 0)
fundraising.cv.ct <- rpart(TARGET_B ~ ., data = train.df, method = "class", 
                               cp = 0.00001, minsplit = 1, xval = 5)

# Print table
printcp(fundraising.ct)
printcp(fundraising.deeper.ct)
printcp(fundraising.cv.ct)

# Prune the tree
fundraising.pruned.deeper <- prune(fundraising.deeper.ct, 
                               cp = fundraising.deeper.ct$cptable[which.min(fundraising.deeper.ct$cptable[,"xerror"]),"CP"])
printcp(fundraising.pruned.deeper)

fundraising.pruned.cv <- prune(fundraising.cv.ct, 
                               cp = fundraising.cv.ct$cptable[which.min(fundraising.cv.ct$cptable[,"xerror"]),"CP"])
printcp(fundraising.pruned.cv)

## Plot the tree
prp(fundraising.ct, type = 1, extra = 1, under=TRUE, split.font = 1, varlen = -10)
prp(fundraising.deeper.ct, type = 1, extra = 1, under=TRUE, split.font = 1, varlen = -10)
prp(fundraising.pruned.deeper, type = 1, extra = 1, under=TRUE, split.font = 1, varlen = -10)
prp(fundraising.pruned.cv, type = 1, extra = 1, under=TRUE, split.font = 1, varlen = -10)

rpart.plot(fundraising.ct)
rpart.plot(fundraising.pruned.deeper)

###########################################################################
## Decision Tree - Confusion Matrix
###########################################################################

# Predict with default tree
default.pred.train <- predict(fundraising.ct, train.df, type = "class")
head(default.pred.train)
default.pred.valid <- predict(fundraising.ct, valid.df, type = "class")
class(default.pred.train)
class(train.df$TARGET_B)
confusionMatrix(default.pred.train, as.factor(train.df$TARGET_B))
confusionMatrix(default.pred.valid, as.factor(valid.df$TARGET_B))

# Predict with pruned deeper tree
pruned.pred.deeper.train <- predict(fundraising.pruned.deeper, train.df, type = "class")
pruned.pred.deeper.valid <- predict(fundraising.pruned.deeper, valid.df, type = "class")
confusionMatrix(pruned.pred.deeper.train, as.factor(train.df$TARGET_B))
confusionMatrix(pruned.pred.deeper.valid, as.factor(valid.df$TARGET_B))

# Predict with pruned cv tree
pruned.pred.cv.train <- predict(fundraising.pruned.cv, train.df, type = "class")
pruned.pred.cv.valid <- predict(fundraising.pruned.cv, valid.df, type = "class")
confusionMatrix(pruned.pred.cv.train, as.factor(train.df$TARGET_B))
confusionMatrix(pruned.pred.cv.valid, as.factor(valid.df$TARGET_B))

#Based on the confusion matrix results, the pruned deeper tree results in the 
#highest accuracy rates

###########################################################################
## Decision Tree - Gain Chart
###########################################################################

# Default Tree Gain Chart and ROC Curve
default.train.prob <- predict(fundraising.ct,train.df)
head(default.train.prob,20)
default.train.list <- sort(default.train.prob[,2],decreasing = TRUE)
default.train.df <- as.data.frame(default.train.list)
default.train.df$V2 <- train.df$TARGET_B
colnames(default.train.df)[1:2] <- c("Probability","TARGET_B") 
View(default.train.df)

default.train.roc.curve <- roc(default.train.df$TARGET_B, default.train.df$Probability)
auc(default.train.roc.curve)
default.train.lift <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = default.train.df)
xyplot(default.train.lift, plot = "gain")

# Default Tree valid Gain Chart and ROC Curve
default.valid.prob <- predict(fundraising.ct,valid.df)
head(default.valid.prob,20)
default.valid.list <- sort(default.valid.prob[,2],decreasing = TRUE)
default.valid.df <- as.data.frame(default.valid.list)
default.valid.df$V2 <- valid.df$TARGET_B
colnames(default.valid.df)[1:2] <- c("Probability","TARGET_B") 
head(default.valid.df,20)

default.valid.roc.curve <- roc(default.valid.df$TARGET_B, default.valid.df$Probability)
auc(default.valid.roc.curve)
default.valid.lift <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = default.valid.df)
xyplot(default.valid.lift, plot = "gain")

# prunded.deeper train Gain Chart and ROC Curve
pruned.deeper.train.prob <- predict(fundraising.ct,train.df)
head(pruned.deeper.train.prob,20)
pruned.deeper.train.list <- sort(pruned.deeper.train.prob[,2],decreasing = TRUE)
pruned.deeper.train.df <- as.data.frame(pruned.deeper.train.list)
pruned.deeper.train.df$V2 <- train.df$TARGET_B
colnames(pruned.deeper.train.df)[1:2] <- c("Probability","TARGET_B") 
head(pruned.deeper.train.df,20)

pruned.deeper.train.roc.curve <- roc(pruned.deeper.train.df$TARGET_B, pruned.deeper.train.df$Probability)
auc(pruned.deeper.train.roc.curve)
pruned.deeper.train.lift <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = pruned.deeper.train.df)
xyplot(pruned.deeper.train.lift, plot = "gain")

# prunded.deeper valid Gain Chart and ROC Curve
pruned.deeper.valid.prob <- predict(fundraising.ct,valid.df)
head(pruned.deeper.valid.prob,20)
pruned.deeper.valid.list <- sort(pruned.deeper.valid.prob[,2],decreasing = TRUE)
pruned.deeper.valid.df <- as.data.frame(pruned.deeper.valid.list)
pruned.deeper.valid.df$V2 <- valid.df$TARGET_B
colnames(pruned.deeper.valid.df)[1:2] <- c("Probability","TARGET_B") 
head(pruned.deeper.valid.df,20)

pruned.deeper.valid.roc.curve <- roc(pruned.deeper.valid.df$TARGET_B, pruned.deeper.valid.df$Probability)
auc(pruned.deeper.valid.roc.curve)
pruned.deeper.valid.lift <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = pruned.deeper.valid.df)
xyplot(pruned.deeper.valid.lift, plot = "gain")

# pruned.cv train Gain Chart and ROC Curve
pruned.cv.train.prob <- predict(fundraising.ct,train.df)
head(pruned.cv.train.prob,20)
pruned.cv.train.list <- sort(pruned.cv.train.prob[,2],decreasing = TRUE)
pruned.cv.train.df <- as.data.frame(pruned.cv.train.list)
pruned.cv.train.df$V2 <- train.df$TARGET_B
colnames(pruned.cv.train.df)[1:2] <- c("Probability","TARGET_B") 
head(pruned.cv.train.df,20)

pruned.cv.train.roc.curve <- roc(pruned.cv.train.df$TARGET_B, pruned.cv.train.df$Probability)
auc(pruned.cv.train.roc.curve)
pruned.cv.train.lift <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = pruned.cv.train.df)
xyplot(pruned.cv.train.lift, plot = "gain")

# prunded.cv valid Gain Chart and ROC Curve
pruned.cv.valid.prob <- predict(fundraising.ct,valid.df)
head(pruned.cv.valid.prob,20)
pruned.cv.valid.list <- sort(pruned.cv.valid.prob[,2],decreasing = TRUE)
pruned.cv.valid.df <- as.data.frame(pruned.cv.valid.list)
pruned.cv.valid.df$V2 <- valid.df$TARGET_B
colnames(pruned.cv.valid.df)[1:2] <- c("Probability","TARGET_B") 
head(pruned.cv.valid.df,20)

pruned.cv.valid.roc.curve <- roc(pruned.cv.valid.df$TARGET_B, pruned.cv.valid.df$Probability)
auc(pruned.cv.valid.roc.curve)
pruned.cv.valid.lift <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = pruned.cv.valid.df)
xyplot(pruned.cv.valid.lift, plot = "gain")

###########################################################################
## Alternative Models - Adaboost & Random Forest
###########################################################################

# Adaboost model
train.boost <- train.df
train.boost$TARGET_B <- as.factor(train.boost$TARGET_B)
set.seed(2)
boost <- boosting(TARGET_B ~ ., data = train.boost) 

#Random Forest model
set.seed(2)
rf <- randomForest(as.factor(TARGET_B) ~ ., data = train.df, ntree = 500, 
                   mtry = 4, nodesize = 5, importance = TRUE)  
varImpPlot(rf, type = 1)

###########################################################################
## Alternative Models - Confusion Matrix
###########################################################################

# Predict with boost
boost.pred.train <- predict(boost, train.df)
boost.pred.valid <- predict(boost, valid.df)
confusionMatrix(as.factor(boost.pred.train$class), as.factor(train.df$TARGET_B))
confusionMatrix(as.factor(boost.pred.valid$class), as.factor(valid.df$TARGET_B))

# Predict with random forest
rf.pred.train <- predict(rf, train.df)
rf.pred.valid <- predict(rf, valid.df)
confusionMatrix(rf.pred.train, as.factor(train.df$TARGET_B))
confusionMatrix(rf.pred.valid, as.factor(valid.df$TARGET_B))

###########################################################################
## Alternative Models - Gain Chart
###########################################################################

# Boost train Gain Chart and ROC Curve
boost.train.prob <- predict(boost,train.df)
head(boost.train.prob,20)
boost.train.list <- sort(boost.train.prob$prob[,2],decreasing = TRUE)
boost.train.df <- as.data.frame(boost.train.list)
boost.train.df$V2 <- train.df$TARGET_B
colnames(boost.train.df)[1:2] <- c("Probability","TARGET_B") 
head(boost.train.df,20)

boost.train.roc.curve <- roc(boost.train.df$TARGET_B, boost.train.df$Probability)
auc(boost.train.roc.curve)
boost.train.lift <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = boost.train.df)
xyplot(boost.train.lift, plot = "gain")

# Boost valid Gain Chart and ROC Curve
boost.valid.prob <- predict(boost,valid.df)
head(boost.valid.prob,20)
boost.valid.list <- sort(boost.valid.prob$prob[,2],decreasing = TRUE)
boost.valid.df <- as.data.frame(boost.valid.list)
boost.valid.df$V2 <- valid.df$TARGET_B
colnames(boost.valid.df)[1:2] <- c("Probability","TARGET_B") 
head(boost.valid.df,20)

boost.valid.roc.curve <- roc(boost.valid.df$TARGET_B, boost.valid.df$Probability)
auc(boost.valid.roc.curve)
boost.valid.lift <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = boost.valid.df)
xyplot(boost.valid.lift, plot = "gain")

# Random Forest train Gain Chart and ROC Curve
rf.train.prob <- predict(rf,train.df,type="prob")
summary(rf.train.prob,20)
rf.train.list <- sort(rf.train.prob,decreasing = TRUE)
rf.train.df <- as.data.frame(as.numeric(rf.train.list))
rf.train.df$V2 <- train.df$TARGET_B
colnames(rf.train.df)[1:2] <- c("Probability","TARGET_B") 
head(rf.train.df,20)

rf.train.roc.curve <- roc(rf.train.df$TARGET_B, rf.train.df$Probability)
auc(rf.train.roc.curve)
rf.train.lift <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = rf.train.df)
xyplot(rf.train.lift, plot = "gain")


# Random Forest Gain Chart and ROC Curve
rf.valid.prob <- predict(rf,valid.df,type="prob")
head(rf.valid.prob,20)
rf.valid.list <- sort(rf.valid.prob,decreasing = TRUE)
rf.valid.df <- as.data.frame(as.numeric(rf.valid.list))
rf.valid.df$V2 <- valid.df$TARGET_B
colnames(rf.valid.df)[1:2] <- c("Probability","TARGET_B") 
head(rf.valid.df,20)

rf.valid.roc.curve <- roc(rf.valid.df$TARGET_B, rf.valid.df$Probability)
auc(rf.valid.roc.curve)
rf.valid.lift <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = rf.valid.df)
xyplot(rf.valid.lift, plot = "gain")

###########################################################################
## Neural Network Model
###########################################################################

# Choose variables
# Variables from Pruned Classification Tree
# AVGGIFT     Icmed       INCOME      MAXRAMNT    NUMPROM     RAMNTALL    totalmonths

#Variables from Random Forest
# "AVGGIFT","MAXRAMNT","LASTGIFT","totalmonths", "Income"

#Decision to use variables from Random Forest as other items are less impactful


vars <- c("AVGGIFT","MAXRAMNT","LASTGIFT","totalmonths")
fundraising.nn.df <- fundraising.df[,vars]

# Prepare the data
normalize_min_maz<- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
fundraising.nn <- cbind(normalize_min_maz(fundraising.nn.df), class.ind(fundraising.df$INCOME),
                        class.ind(fundraising.df$TARGET_B))
names(fundraising.nn) <- c(vars, "INCOME_1","INCOME_2","INCOME_3","INCOME_4","INCOME_5","INCOME_6","INCOME_7",
                           "NonDonor","Donor")

train.nn <- fundraising.nn[training,]
valid.nn <- fundraising.nn[-training,]

# Neural Network Model
set.seed(3)
nn_fd.1 <- neuralnet(NonDonor+Donor ~ AVGGIFT+MAXRAMNT+LASTGIFT+totalmonths+INCOME_1+INCOME_2+INCOME_3+INCOME_4+INCOME_5+INCOME_6+INCOME_7,data = train.nn, hidden = 3)

#Decision to use 3 hidden variables as results are slightly improved over 2.
#Also 5 hidden layers results in lower accuracy than 3 hidden layers.
# nn_fd.2 <- neuralnet(NonDonor+Donor ~.,data = train.nn, hidden = 5)
# plot (nn_fd.2, rep="best")

plot (nn_fd.1, rep="best")

###########################################################################
## Confusion Matrix and Gain Chart with Neural Network Model
###########################################################################

# Confusion Matrix for Neural Network 1
training.prediction.1 <- compute(nn_fd.1, train.nn[,-c(12:13)])
training.class.1 <- apply(training.prediction.1$net.result,1,which.max)-1
head (training.class.1)
confusionMatrix(as.factor(training.class.1), as.factor(train.df$TARGET_B))
# Accurace: 0.5919, Sensitivity: 0.5955, Specificity : 0.5882

validation.prediction.1 <- compute(nn_fd.1, valid.nn[,-c(12:13)])
validation.class.1 <- apply(validation.prediction.1$net.result,1,which.max)-1
head (validation.class.1)
confusionMatrix(as.factor(validation.class.1), as.factor(valid.df$TARGET_B))
# Accuracy : 0.5545, Sensitivity : 0.5518        Specificity : 0.5571

# NN1 Gain Chart and ROC Curve with train set
prediction.df <- as.data.frame(training.prediction.1$net.result)
lift.list <- sort(prediction.df[,2],decreasing = TRUE)
lift.df <- as.data.frame(lift.list)
lift.df$V2 <- train.df$TARGET_B
colnames(lift.df)[1:2] <- c("Probability","TARGET_B") 
head(lift.df,20)

r <- roc(lift.df$TARGET_B, lift.df$Probability)
auc(r)
lift.example <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = lift.df)
xyplot(lift.example, plot = "gain")

# NN1 Gain Chart and ROC Curve with valid set
prediction.valid.df <- as.data.frame(validation.prediction.1$net.result)
lift.valid.list <- sort(prediction.valid.df[,2],decreasing = TRUE)
lift.valid.df <- as.data.frame(lift.valid.list)
lift.valid.df$V2 <- valid.df$TARGET_B
colnames(lift.valid.df)[1:2] <- c("Probability","TARGET_B") 
head(lift.valid.df,20)

r <- roc(lift.valid.df$TARGET_B, lift.valid.df$Probability)
auc(r)
lift.valid.example <- lift(relevel(as.factor(TARGET_B), ref="1") ~ Probability, data = lift.valid.df)
xyplot(lift.valid.example, plot = "gain")

###########################################################################
## Which model is better?
###########################################################################

# Default Decision tree (Accuracy 0.5473, Sensitivity 0.5825, Specificity 0.5127)
confusionMatrix(default.pred.valid, as.factor(valid.df$TARGET_B))
# Pruned Deeper Decision tree (Accuracy 0.5521, Sensitivity 0.5210, Specificity 0.5825)
confusionMatrix(pruned.pred.deeper.valid, as.factor(valid.df$TARGET_B))
# Pruned CV Decision tree (Accuracy 0.5513, Sensitivity 0.5712, Specificity 0.5317)
confusionMatrix(pruned.pred.cv.valid, as.factor(valid.df$TARGET_B))
# Adaboost Model (Accuracy 0.5337, Sensitivity 0.5291, Specificity 0.5381)
confusionMatrix(as.factor(boost.pred.valid$class), as.factor(valid.df$TARGET_B))
# Random Forest Model (Accuracy 0.5345, Sensitivity 0.5146, Specificity 0.5540)
confusionMatrix(rf.pred.valid, as.factor(valid.df$TARGET_B))
#Neural Network Model (Accuracy 0.5545, Sensitivity 0.5518, Specificity 0.5571)
confusionMatrix(as.factor(validation.class.1), as.factor(valid.df$TARGET_B))

#Confidence interval for Neural Network Model
#95% CI : (0.5264, 0.5823).  Accuracy results for all models are within the confidence
#interval so for each random sample, the "best" model may vary.

###########################################################################
## Prediction with FutureFundraising.csv and Save data
###########################################################################

# Import dataset
FutureFundraising <- read.csv("FutureFundraising.csv")
str(FutureFundraising)
head(FutureFundraising)
summary(FutureFundraising)
apply (FutureFundraising, 2, class)
apply(is.na(FutureFundraising),2,sum) #sum NA values for each column
prediction.df <- FutureFundraising[,3:22]
head(prediction.df)



# Prepare the data

vars <- c("AVGGIFT","MAXRAMNT","LASTGIFT","totalmonths")
prediction.nn.df <- prediction.df[,vars]

prediction.nn <- cbind(normalize_min_maz(prediction.nn.df), class.ind(prediction.df$INCOME))
names(prediction.nn) <- c(vars, "INCOME_1","INCOME_2","INCOME_3","INCOME_4","INCOME_5","INCOME_6","INCOME_7")


# Predict TARGET_B
TARGET_B <- neuralnet::compute(nn_fd.1, prediction.nn)
head(TARGET_B)
FutureFundraising$TARGET_B <- apply(TARGET_B$net.result,1,which.max)-1
head(FutureFundraising)

###########################################################################
## Predict TARGET_D for TARGET_B=1
###########################################################################
D.df <- read.csv("Fundraising.csv")
D.df <- D.df[D.df$TARGET_B==1,]
D.df <- D.df[, -c(1:2,23)]
D.train <- D.df[training,]
D.valid <- D.df[-training,]
head(D.df)

# Built lm model
D.lm <- lm(TARGET_D~., data = D.train)
options(scipen = 999)
summary(D.lm)

# Use step to run stepwise regression to optimize the linear regression.
options(scipen=999, digits = 3)
D.step.lm <- step(D.lm, direction = "both")
summary(D.step.lm)
plot(D.step.lm)

pred.step.train <- predict(D.step.lm, D.train)
pred.step.valid <- predict(D.step.lm, D.valid)

accuracy(pred.step.train, D.train$TARGET_D)
accuracy(pred.step.valid, D.valid$TARGET_D)

###########################################################################
## Predict TARGET_D with FutureFundraising.csv and Save data
###########################################################################

FutureFundraising.D.df <- subset(FutureFundraising, TARGET_B==1)
FutureFundraising.D <- FutureFundraising.D.df[,-c(1:2,21)]

# Predict TARGET_D
options(digits = 2)
TARGET_D <- predict(D.step.lm, FutureFundraising.D)
head(TARGET_D)
FutureFundraising.D.df$TARGET_D <- TARGET_D

# Join TARGET_D to data
FF <- left_join(FutureFundraising[,-24], FutureFundraising.D.df[,c(1,24)], by="ï..Row.Id")
FF$TARGET_D <- ifelse(FutureFundraising$TARGET_B==0, 0, FF$TARGET_D)
View(FF)

sum(FF$TARGET_D)
DD <- read.csv("Fundraising.csv")
sum(DD$TARGET_D)

sum(FF$TARGET_B==1)

write.csv(FF,"G:/UMB/MSIS 672/Final Project/FutureFundraising_Group.csv", row.names = FALSE)
