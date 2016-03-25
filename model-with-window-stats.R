df_stats <- filter(df, new_window == 'yes')

# Features extraction
measurement_stats_vars <- grep(".*avg*|.*var.*|.*stddev.*|.*min.*|.*max.*", names(df), value = TRUE)
measurement_stats_na_count <- data.frame(feature = measurement_stats_vars, non_na_count = sapply(measurement_stats_vars, function(feature) { sum(!is.na(df_stats[,feature]))}))
measurement_stats_non_na_vars <- as.vector(filter(measurement_stats_na_count, non_na_count == dim(df_stats)[1])[, "feature"])

# K-folds
training_adelmo <- select(filter(df_stats, user_name == "adelmo"), one_of(append(c("classe"), measurement_stats_non_na_vars)))
training_wo_adelmo <- select(filter(df_stats, user_name != "adelmo"), one_of(append(c("classe"), measurement_stats_non_na_vars)))

training_carlitos <- select(filter(df_stats, user_name == "carlitos"), one_of(append(c("classe"), measurement_stats_non_na_vars)))
training_wo_carlitos <- select(filter(df_stats, user_name != "carlitos"), one_of(append(c("classe"), measurement_stats_non_na_vars)))

training_charles <- select(filter(df_stats, user_name == "charles"), one_of(append(c("classe"), measurement_stats_non_na_vars)))
training_wo_charles <- select(filter(df_stats, user_name != "charles"), one_of(append(c("classe"), measurement_stats_non_na_vars)))

training_eurico <- select(filter(df_stats, user_name == "eurico"), one_of(append(c("classe"), measurement_stats_non_na_vars)))
training_wo_eurico <- select(filter(df_stats, user_name != "eurico"), one_of(append(c("classe"), measurement_stats_non_na_vars)))

training_jeremy <- select(filter(df_stats, user_name == "jeremy"), one_of(append(c("classe"), measurement_stats_non_na_vars)))
training_wo_jeremy <- select(filter(df_stats, user_name != "jeremy"), one_of(append(c("classe"), measurement_stats_non_na_vars)))

training_pedro <- select(filter(df_stats, user_name == "pedro"), one_of(append(c("classe"), measurement_stats_non_na_vars)))
training_wo_pedro <- select(filter(df_stats, user_name != "pedro"), one_of(append(c("classe"), measurement_stats_non_na_vars)))

# ------------------------------------------------------------------------------
# K-fold: adelmo
# ------------------------------------------------------------------------------
# Correlation matrix
cor_matrix <- abs(cor(training_wo_adelmo[,-1]))
diag(cor_matrix) <- 0
which(cor_matrix > .8, arr.ind=T)

# PCA
set.seed(12345)
preProc <- preProcess(training_wo_adelmo[, -1], method="pca")
trainPC <- predict(preProc, training_wo_adelmo[,-1])
preProc

# Models

# glm:
set.seed(12345)
# With PCA: No working.
modelFit <- train(training_wo_adelmo$classe ~ ., method="glm", data=trainPC, verbose=TRUE)
# without PCA: No working.
modelFit <- train(classe ~ ., method="glm", data=training_wo_adelmo, verbose=TRUE)

# Algorithm: Predicting with trees
# Method=rpart:
set.seed(12345)
# With PCA: Not working
modelFit <- train(training_wo_adelmo$classe ~ ., method="rpart", data=trainPC, verbose=TRUE)
# Without PCA: Not working
modelFit <- train(classe ~ ., method="rpart", data=training_wo_adelmo, verbose=TRUE)

# Algorithm: Bagging
# Method=bagEarth
set.seed(12345)
# With PCA: Not working
modelFit <- train(training_wo_adelmo$classe ~ ., method="bagEarth", data=trainPC, verbose=TRUE)
# Without PCA: Not working
modelFit <- train(classe ~ ., method="bagEarth", data=training_wo_adelmo, verbose=TRUE)

# Algorithm: Boosting
# Method=gbm:
set.seed(12345)
# With PCA: Accuracy : 0.3735
modelFit <- train(training_wo_adelmo$classe ~ ., method="gbm", data=trainPC, verbose=TRUE)
# Without PCA: Accuracy : 0.3494
modelFit <- train(classe ~ ., method="gbm", data=training_wo_adelmo, verbose=TRUE)

# Algorithm: Random Forests
# Method=rf
# K-fold: adelmo
set.seed(12345)
# With PCA: Accuracy : 0.3855
modelFit <- train(training_wo_adelmo$classe ~ ., method="rf", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_adelmo[,-1])
confusionMatrix(training_adelmo$classe, predict(modelFit, testPC))

# Without PCA: Accuracy : 0.5422 
modelFit <- train(classe ~ ., method="rf", data=training_wo_adelmo, verbose=TRUE)
confusionMatrix(training_adelmo$classe, predict(modelFit, training_adelmo))

# ------------------------------------------------------------------------------
# K-fold: carlitos
# ------------------------------------------------------------------------------
set.seed(12345)
preProc <- preProcess(training_wo_carlitos[, -1], method="pca")
trainPC <- predict(preProc, training_wo_carlitos[,-1])
preProc

# With PCA: Accuracy : 0.5179
modelFit <- train(training_wo_carlitos$classe ~ ., method="rf", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_carlitos[,-1])
confusionMatrix(training_carlitos$classe, predict(modelFit, testPC))
# Without PCA: Accuracy : 0.5179 
modelFit <- train(classe ~ ., method="rf", data=training_wo_carlitos, verbose=TRUE)
confusionMatrix(training_carlitos$classe, predict(modelFit, training_carlitos))

# ------------------------------------------------------------------------------
# K-fold: charles
# ------------------------------------------------------------------------------
set.seed(12345)
preProc <- preProcess(training_wo_charles[, -1], method="pca")
trainPC <- predict(preProc, training_wo_charles[,-1])
preProc

# With PCA: Accuracy : 0.5062
set.seed(12345)
modelFit <- train(training_wo_charles$classe ~ ., method="rf", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_charles[,-1])
confusionMatrix(training_charles$classe, predict(modelFit, testPC))
# Without PCA: Accuracy : 0.6173
set.seed(12345)
modelFit <- train(classe ~ ., method="rf", data=training_wo_charles, verbose=TRUE)
confusionMatrix(training_charles$classe, predict(modelFit, training_charles))

# ------------------------------------------------------------------------------
# K-fold: eurico
# ------------------------------------------------------------------------------
set.seed(12345)
preProc <- preProcess(training_wo_eurico[, -1], method="pca")
trainPC <- predict(preProc, training_wo_eurico[,-1])
preProc

# With PCA: Accuracy : 0.2963
set.seed(12345)
modelFit <- train(training_wo_eurico$classe ~ ., method="rf", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_eurico[,-1])
confusionMatrix(training_eurico$classe, predict(modelFit, testPC))
# Without PCA: Accuracy : 0.3519
set.seed(12345)
modelFit <- train(classe ~ ., method="rf", data=training_wo_eurico, verbose=TRUE)
confusionMatrix(training_eurico$classe, predict(modelFit, training_eurico))

# ------------------------------------------------------------------------------
# K-fold: jeremy
# ------------------------------------------------------------------------------
set.seed(12345)
preProc <- preProcess(training_wo_jeremy[, -1], method="pca")
trainPC <- predict(preProc, training_wo_jeremy[,-1])
preProc

# With PCA: Accuracy : 0.4675
set.seed(12345)
modelFit <- train(training_wo_jeremy$classe ~ ., method="rf", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_jeremy[,-1])
confusionMatrix(training_jeremy$classe, predict(modelFit, testPC))
# Without PCA: Accuracy : 0.5195
set.seed(12345)
modelFit <- train(classe ~ ., method="rf", data=training_wo_jeremy, verbose=TRUE)
confusionMatrix(training_jeremy$classe, predict(modelFit, training_jeremy))

# ------------------------------------------------------------------------------
# K-fold: pedro
# ------------------------------------------------------------------------------
set.seed(12345)
preProc <- preProcess(training_wo_pedro[, -1], method="pca")
trainPC <- predict(preProc, training_wo_pedro[,-1])
preProc

# With PCA: Accuracy : 0.2545
set.seed(12345)
modelFit <- train(training_wo_pedro$classe ~ ., method="rf", data=trainPC, verbose=TRUE)
testPC <- predict(preProc, training_pedro[,-1])
confusionMatrix(training_pedro$classe, predict(modelFit, testPC))
# Without PCA: Accuracy : 0.4
set.seed(12345)
modelFit <- train(classe ~ ., method="rf", data=training_wo_pedro, verbose=TRUE)
confusionMatrix(training_pedro$classe, predict(modelFit, training_pedro))

# ------------------------------------------------------------------------------
# With data partitioning: 0.8235 *** THE BEST ***
# ------------------------------------------------------------------------------
# Partitioning
set.seed(12345)
inTrain <- createDataPartition(df_stats$classe, p = .7, list = FALSE)
training <- select(df_stats[inTrain,], one_of(append(c("classe"), measurement_stats_non_na_vars)))
testing <- select(df_stats[-inTrain,], one_of(append(c("classe"), measurement_stats_non_na_vars)))
modelFit <- train(classe ~ ., method="rf", data=training, verbose=TRUE)
confusionMatrix(testing$classe, predict(modelFit, testing))
