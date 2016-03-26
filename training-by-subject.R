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
