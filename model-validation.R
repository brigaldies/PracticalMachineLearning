df_validating <- loadData(validatingDataFileName, TRUE)

# Data frame with non-NA model predictors only
df_validate_measurement_non_na_vars <- select(df_validating, one_of(measurement_non_na_vars))

# Verify that there isn't any NA data
data.frame(feature = names(df_validate_measurement_non_na_vars), non_na_count = sapply(names(df_validate_measurement_non_na_vars), function(feature) { sum(is.na(df_validate_measurement_non_na_vars[,feature]))}))

# Predict
predict(modelFit.rf, df_validate_measurement_non_na_vars)
