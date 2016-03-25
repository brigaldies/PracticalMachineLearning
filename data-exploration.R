# In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 
# They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
# More information is available from the website here: http://groupware.les.inf.puc-rio.br/har 
# (see the section on the Weight Lifting Exercise Dataset). 
#
# The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 
# You may use any of the other variables to predict with. You should create a report describing how you built your model, 
# how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. 
# You will also use your prediction model to predict 20 different test cases. 

# Load some required libraries
library(plyr)
library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(gbm)
library(randomForest)

# Load the data
loadData <- function(dataFileName, isTestFile = FALSE) {
    df <- read.csv(dataFileName, stringsAsFactors = FALSE, strip.white = TRUE, na.strings = c("", "NA", "#DIV/0!"))
    # Set factors
    df$user_name <- as.factor(df$user_name)  
    if (!isTestFile) {
        df$classe <- as.factor(df$classe)    
    }
    return(df)
}

trainingDataFileName <- 'pml-training.csv'
validatingDataFileName <- 'pml-testing.csv'

df <- loadData(trainingDataFileName)
df_validating <- loadData(validatingDataFileName, TRUE)

# training
# df_full <- read.csv('WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv')
# df <- read.csv('pml-training.csv', as.is = non_factors)

dim(df)
names(df)[1] <- "row_num"
summary(df$user_name) # The test subjects
summary(df$classe)

table(df[, c("user_name", "classe")])

# remove the one "carli" record data entry record
df <- filter(df, user_name != "carli")
filter(df, user_name == "carli")
table(df[, c("user_name", "classe")])
summary(df$user_name)
unique(select(df, user_name))

# str(df)
grep(".*_z.*", names(df), value = TRUE)
grep(".*_belt.*", names(df), value = TRUE)
grep(".*_arm.*", names(df), value = TRUE)
grep(".*_forearm.*", names(df), value = TRUE)
grep(".*_dumbbell.*", names(df), value = TRUE)
grep(".*var_.*", names(df), value = TRUE)
grep(".*avg_.*", names(df), value = TRUE)
grep(".*stddev_.*", names(df), value = TRUE)
grep(".*pitch.*", names(df), value = TRUE)

# Sensors
# sensors <- c("belt", "arm", "forearm", "dumbell")

# Sensor measurement variables
measurement_vars <- grep(".*belt.*|.*arm.*|.*forearm.*|.*dumbell.*", names(df), value = TRUE)

# Exploring variables with lots of NAs
# count.na <- function(df, feature) { sum(!is.na(df[,feature]))}
# variables <- grep(".*avg_.*", names(df), value = TRUE)
measurement_vars_na_count <- data.frame(feature = measurement_vars, non_na_count = sapply(measurement_vars, function(feature) { sum(!is.na(df[,feature]))}))
measurement_non_na_vars <- as.vector(filter(measurement_vars_na_count, non_na_count == dim(df)[1])[, "feature"])

# Non-na measurements
df_measurement_non_na_vars <- select(df, one_of(append(c("classe"), measurement_non_na_vars)))
# Correlation matrix
cor_matrix <- abs(cor(df_measurement_non_na_vars))
diag(cor_matrix) <- 0
which(cor_matrix > .8, arr.ind=T)

# Training K-folds
training_adelmo <- select(filter(df, user_name == "adelmo"), one_of(append(c("classe"), measurement_non_na_vars)))
training_carlitos <- select(filter(df, user_name == "carlitos"), one_of(append(c("classe"), measurement_non_na_vars)))
training_charles <- select(filter(df, user_name == "charles"), one_of(append(c("classe"), measurement_non_na_vars)))
training_eurico <- select(filter(df, user_name == "eurico"), one_of(append(c("classe"), measurement_non_na_vars)))
training_jeremy <- select(filter(df, user_name == "jeremy"), one_of(append(c("classe"), measurement_non_na_vars)))
training_pedro <- select(filter(df, user_name == "pedro"), one_of(append(c("classe"), measurement_non_na_vars)))
training_wo_adelmo <- select(filter(df, user_name != "adelmo"), one_of(append(c("classe"), measurement_non_na_vars)))
training_wo_carlitos <- select(filter(df, user_name != "carlitos"), one_of(append(c("classe"), measurement_non_na_vars)))
training_wo_charles <- select(filter(df, user_name != "charles"), one_of(append(c("classe"), measurement_non_na_vars)))
training_wo_eurico <- select(filter(df, user_name != "eurico"), one_of(append(c("classe"), measurement_non_na_vars)))
training_wo_jeremy <- select(filter(df, user_name != "jeremy"), one_of(append(c("classe"), measurement_non_na_vars)))
training_wo_pedro <- select(filter(df, user_name != "pedro"), one_of(append(c("classe"), measurement_non_na_vars)))

# K-fold: Adelmo
summary(training_adelmo$classe)
set.seed(12345)
preProc <- preProcess(training_wo_adelmo[, -1], method="pca")
trainPC <- predict(preProc, training_wo_adelmo[,-1])
preProc

# Algorithm: Generalized Linear Modeling
# Method=glm: Not working for either raw or processed data!
modelFit <- train(training_wo_adelmo$classe ~ ., method="glm", data=trainPC, verbose=TRUE)
modelFit <- train(classe ~ ., method="glm", data=training_wo_adelmo, verbose=TRUE)

# Algorithm: Predicting with trees
# Method=rpart: Very inaccurate!
# With PCA
modelFit <- train(training_wo_adelmo$classe ~ ., method="rpart", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_adelmo[,-1])
confusionMatrix(training_adelmo$classe, predict(modelFit, testPC))
# Without PCA
modelFit <- train(classe ~ ., method="rpart", data=training_wo_adelmo, verbose=TRUE)
confusionMatrix(training_adelmo$classe, predict(modelFit, training_adelmo))

# Method=party: Not in the caret package
modelFit <- train(training_wo_adelmo$classe ~ ., method="party", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_adelmo[,-1])
confusionMatrix(training_adelmo$classe, predict(modelFit, testPC))

# Algorithm: Bagging
# Method=bagEarth: Does not converge
# Method=treebag: Execution not finishing within minutes
modelFit <- train(training_wo_adelmo$classe ~ ., method="treebag", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_adelmo[,-1])
confusionMatrix(training_adelmo$classe, predict(modelFit, testPC))

# Algorithm: Boosting
# Method=gbm: Execution not finishing within minutes
set.seed(12345)
modelFit <- train(training_wo_adelmo$classe ~ ., method="gbm", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_adelmo[,-1])
confusionMatrix(training_adelmo$classe, predict(modelFit, testPC))

# Algorithm: Random Forests
modelFit <- train(training_wo_adelmo$classe ~ ., method="rf", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_adelmo[,-1])
confusionMatrix(training_adelmo$classe, predict(modelFit, testPC))

# ------------------------------------------------------------------------------
# Function: trainWithMethod
# 
# Train with a given method, and with and without PCA pre-processing.
# ------------------------------------------------------------------------------
trainWithMethod <- function(methodName, 
                            seedNumber,                    
                            trainingData,                             
                            testingData,
                            doPCA = TRUE) {
    set.seed(seedNumber)
    modelFitPca <- NULL
    cMatrixPca <- NULL
    execTimePca <- NULL
    # ---------- With PCA ----------
    if (doPCA == TRUE) {
        preProc <- preProcess(trainingData[, -1], method="pca")
        trainPC <- predict(preProc, trainingData[,-1])    
        
        message(paste("Training with method", methodName, "with PCA ..."))
        execTimePca <- system.time({       
            try(
                if (methodName == 'gbm') {
                    modelFitPca <- train(trainingData$classe ~ ., method=methodName, data=trainPC, verbose=FALSE)
                } else {
                    modelFitPca <- train(trainingData$classe ~ ., method=methodName, data=trainPC)
                }
            )
        })
        message(paste("Training with method", methodName, "with PCA completed in", round(execTimePca["user.self"], 2), "secs"))
        
        if (is.null(modelFitPca)) { 
            message(paste("Training with method", methodName, "with PCA failed."))
        } else {
            testPC <- predict(preProc, testingData[,-1])
            cMatrixPca <- confusionMatrix(testingData$classe, predict(modelFitPca, testPC))
            message(paste("Training with method", methodName, "with PCA reached an accuracy of", cMatrixPca$overall["Accuracy"]))
        }
    } else {
        message(paste("Training with method", methodName, "with PCA skipped."))
    }
    # ---------- Without PCA ----------
    set.seed(seedNumber)
    modelFit <- NULL
    cMatrix <- NULL
    execTime <- NULL
    message(paste("Training with method", methodName, "..."))
    execTime <- system.time({
        try(
            if (methodName == 'gbm') {
                modelFit <- train(classe ~ ., method=methodName, data=trainingData, verbose=FALSE)
            } else {
                modelFit <- train(classe ~ ., method=methodName, data=trainingData)
            }
        )
    })
    message(paste("Training with method", methodName, "completed in", round(execTime["user.self"], 2), "secs"))
    
    if (is.null(modelFit)) { 
        message(paste("Training with method", methodName, "failed."))
    } else {    
        cMatrix <- confusionMatrix(testingData$classe, predict(modelFit, testingData))
        message(paste("Training with method", methodName, "reached an accuracy of", cMatrix$overall["Accuracy"]))
    }
    
    # Return the results
    result <- data.frame(
        method=methodName,        
        execTime = execTime["user.self"],
        accuracy = ifelse(is.null(modelFit), -1, cMatrix$overall["Accuracy"]),        
        execTimePCa = ifelse(is.null(execTimePca), -1, execTimePca["user.self"]),
        accuracyPca = ifelse(is.null(modelFitPca), -1, cMatrixPca$overall["Accuracy"]))
    rownames(result) <- methodName
    return(result)
}

# ------------------------------------------------------------------------------
# Model Building & Testing
# ------------------------------------------------------------------------------
results <- data.frame(model=c(), pca=c(), nonpca=c())

# ------------------------------------------------------------------------------
# With data partitioning
# ------------------------------------------------------------------------------
set.seed(1)
inTrain <- createDataPartition(df_measurement_non_na_vars$classe, p = .7, list = FALSE)
training <- df_measurement_non_na_vars[inTrain,]
testing <- df_measurement_non_na_vars[-inTrain,]

# ------------------------------------------------------------------------------
# With PCA
# ------------------------------------------------------------------------------
# set.seed(2)
#preProc <- preProcess(training[, -1], method="pca")
#trainPC <- predict(preProc, training[,-1])
#preProc

methodsToTry <- c("glm", "rpart", "gbm", "rf")
results = data.frame(method=c(), execTime=c(), accuracy=c(), execTimePca=c(), accuracyPca=c())
for (i in 1:length(methodsToTry)) {
    results <- rbind(results, trainWithMethod(methodsToTry[i], i, training, testing))
}

results
results <- rbind(results, trainWithMethod("glm", 1, training, testing))
results <- rbind(results, trainWithMethod("rpart", 2, training, testing))
results <- rbind(results, trainWithMethod("gbm", 3, training, testing))
results <- rbind(results, trainWithMethod("rf", 4, training, testing, doPCA = FALSE))


# As of 3-22-2016:
#method execTime   accuracy execTimePCa accuracyPca
#glm    glm     0.99 -1.0000000        0.79  -1.0000000
#rpart  rpart  41.42  0.4331351       35.00   0.4020391
#gbm    gbm  1153.77  0.9386576      777.07   0.7661852
#rf     rf   2411.89  0.9935429     1635.56   0.9602379

# ------------------------------------------------------------------------------
# glm - Not working
# ------------------------------------------------------------------------------
set.seed(3)
modelFitPca <- NULL
cMatrixPca <- NULL
modelFit <- NULL
cMatrix <- NULL
execTime <- system.time({
    try(modelFitPca <- train(training$classe ~ ., method="glm", data=trainPC))
})
if (is.null(modelFitPca)) { 
    message("glm with PCA: Error")    
} else {
    testPC <- predict(preProc, testing[,-1])
    cMatrixPca <- confusionMatrix(testing$classe, predict(modelFitPca, testPC))
}
# Without PCA
set.seed(3)
modelFit <- NULL
cMatrix <- NULL
execTime <- system.time({
    modelFit <- train(classe ~ ., method="glm", data=training)
})
if (is.null(modelFit)) { 
    message("glm: Error")    
} else {    
    cMatrix <- confusionMatrix(testing$classe, predict(modelFit, testing))
}
results <- rbind(results, data.frame(
    model='glm',
    pca = ifelse(is.null(modelFitPca), 'Error', 'Success'),
    nonpca = ifelse(is.null(modelFit), 'Error', 'Success')))

# ------------------------------------------------------------------------------
# Tree: rpart - Accuracy : 0.4117
# ------------------------------------------------------------------------------
set.seed(2)
modelFitPca <- NULL
cMatrixPca <- NULL
modelFit <- NULL
cMatrix <- NULL
execTime <- system.time({
    modelFitPca <- train(training$classe ~ ., method="rpart", data=trainPC)
})
if (is.null(modelFitPca)) { 
    message("rpart with PCA: Error")    
} else {
    testPC <- predict(preProc, testing[,-1])
    cMatrixPca <- confusionMatrix(testing$classe, predict(modelFitPca, testPC))
    cMatrixPca
}
# Without PCA
set.seed(4)
modelFit <- NULL
cMatrix <- NULL
execTime <- system.time({
    modelFit <- train(classe ~ ., method="rpart", data=training)
})
if (is.null(modelFit)) { 
    message("rpart: Error")    
} else {    
    cMatrix <- confusionMatrix(testing$classe, predict(modelFit, testing))
}
results <- rbind(results, data.frame(
    model='rpart',
    pca = ifelse(is.null(modelFitPca), 'Error', cMatrixPca$overall["Accuracy"]),
    nonpca = ifelse(is.null(modelFit), 'Error', cMatrix$overall["Accuracy"])))

# ------------------------------------------------------------------------------
# Bagging: bagEarth - Not converging.
# ------------------------------------------------------------------------------
set.seed(5)
modelFitPca <- NULL
cMatrixPca <- NULL
modelFit <- NULL
cMatrix <- NULL
execTime <- system.time({    
    modelFitPca <- train(training$classe ~ ., method="bagEarth", trControl = trainControl(method="oob"), data=trainPC)
})
print(execTime)
if (is.null(modelFitPca)) { 
    message("bagEarth with PCA: Error")    
} else {
    testPC <- predict(preProc, testing[,-1])
    cMatrixPca <- confusionMatrix(testing$classe, predict(modelFitPca, testPC))    
}
    
# Boosting: gbm - Accuracy : 0.7679 (user system time: 806.87)
set.seed(6)
execTime <- system.time({
    modelFit <- train(training$classe ~ ., method="gbm", data=trainPC, verbose=FALSE)
})
print(execTime)
testPC <- predict(preProc, testing[,-1])
confusionMatrix(testing$classe, predict(modelFit, testPC))

# Random Forest: rf - Accuracy : 0.9573 (user system time 1642.66)
set.seed(7)
execTime <- system.time({
    modelFit <- train(training$classe ~ ., method="rf", data=trainPC, verbose=FALSE)
})
print(execTime)
testPC <- predict(preProc, testing[,-1])
confusionMatrix(testing$classe, predict(modelFit, testPC))

# ------------------------------------------------------------------------------
# With data partitioning and no PCA
# ------------------------------------------------------------------------------

# Generalized Linear Regression
# Method=glm: Did not work!
date()
modelFit <- train(classe ~ ., method="glm", data=training, verbose=FALSE)
date()
confusionMatrix(testing$classe, predict(modelFit, testing))

# Tree
# Method=rpart: Did not work!
date()
modelFit <- train(classe ~ ., method="rpart", data=training, verbose=FALSE)
date()
confusionMatrix(testing$classe, predict(modelFit, testing))

# Bagging
# Method=bagEarth: Did not work!
date()
modelFit <- train(classe ~ ., method="bagEarth", data=training, verbose=FALSE)
date()
confusionMatrix(testing$classe, predict(modelFit, testing))

# Boosting (~20 mins model building)
# Method=gbm: 0.9295
date()
modelFit <- train(classe ~ ., method="gbm", data=training, verbose=FALSE)
date()
confusionMatrix(testing$classe, predict(modelFit, testing))

# Random Forests (~40 mins hour model building)
# Method=rf: 0.9884 *** BEST ***
set.seed(4)
modelFit.rf <- NULL
execTime <- system.time({
    modelFit.rf <- modelFit <- train(classe ~ ., method="rf", data=training)
})
print(execTime)
summary(modelFit.rf)
print(modelFit.rf, digits = 3)
modelFit.rf$finalModel
plot(modelFit.rf)
testingPred <- predict(modelFit.rf, testing)
modelFit.cm <- confusionMatrix(testing$classe, testingPred)
modelFit.cm
# out of sample error estimate
sprintf("%0.2f %%", round((1 - sum(testingPred == testing$classe)/length(testingPred)) * 100, 2))

# Test on the validation data



# filter(belt_vars, non_na_count > 19000)

# Derived stats:
# derived_stats <- grep(".*var.*|.*avg.*|.*stddev.*|.*min.*|.*max.*|.*skewness.*|.*kurtosis.*|.*amplitude.*", names(df), value = TRUE)

# When new_window == 'yes', the avg, var, stddev stats are calculated

# test
# None of the test data contain non-na for min, max, avg, var, stddev, and other stats! Can't test using these predictors!!!!!
test_df <- read.csv('pml-testing.csv')

# features with data in the test set
# Some non-na data:
filter(data.frame(feature=names(test_df), non_na_count = sapply(names(test_df), function(feature) {sum(!is.na(test_df[,feature]))})), non_na_count > 0)
# No na: Potential for predictors
# Idea: Run a PCA on them.
filter(data.frame(feature=names(test_df), na_count = sapply(names(test_df), function(feature) {sum(is.na(test_df[,feature]))})), na_count == 0)

# Plots
grep(".*_belt_z.*", names(df_measurement_non_na_vars), value = TRUE)
# Visualizing the hips thrown forward: Classe E
qplot(x = df_measurement_non_na_vars$classe, y= df_measurement_non_na_vars$gyros_belt_z, fill=df_measurement_non_na_vars$classe, geom="boxplot", xlab='Classe', ylab='gyros_belt_z', main='Title') + guides(fill=guide_legend(title="Classe"))
grep(".*_arm_z.*", names(df_measurement_non_na_vars), value = TRUE)
# visualizing the elbows moving up: Class B
qplot(x = df_measurement_non_na_vars$classe, y= df_measurement_non_na_vars$gyros_arm_y, fill=df_measurement_non_na_vars$classe, geom="boxplot")
# Visualizing the dumbell going up or down half way?
grep(".*_forearm.*", names(df_measurement_non_na_vars), value = TRUE)
qplot(x = df_measurement_non_na_vars$classe, y= df_measurement_non_na_vars$pitch_forearm, fill=df_measurement_non_na_vars$classe, geom="boxplot")

# How to make a data frame
# n = c(2, 3, 5) 
# s = c("aa", "bb", "cc") 
# b = c(TRUE, FALSE, TRUE) 
# test_df = data.frame(n, s, b)  
n = c()
s = c()
b = c()
test_df = data.frame(n, s, b)
names(test_df)
test_df <- rbind(test_df, data.frame(n = c(2), s = c('aa'), b = c(TRUE)))
test_df <- rbind(test_df, data.frame(n = c(3), s = c('bb'), b = c(FALSE)))
names(test_df)
rownames(test_df)
