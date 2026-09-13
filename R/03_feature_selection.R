# ============================================================
# DA2 - Stage 2: Feature Selection
# ============================================================

library(dplyr)
library(caret)
library(randomForest)

cat("============================================\n")
cat("DA2 - FEATURE SELECTION\n")
cat("============================================\n")

# ------------------------------------------------------------
# 1. Load engineered data
# ------------------------------------------------------------

hr_data <- readRDS("results/engineered_hr_data.rds")

cat("\nDataset dimensions:\n")
print(dim(hr_data))

# ------------------------------------------------------------
# 2. Remove redundant / identifier columns
# ------------------------------------------------------------

remove_cols <- c(
  "EmployeeNumber",
  "EducationLevel"
)

model_data <- hr_data %>%
  select(-any_of(remove_cols))

cat("\nRemoved columns:\n")
print(remove_cols)

# ------------------------------------------------------------
# 3. Train / test split
# ------------------------------------------------------------

set.seed(123)

train_index <- createDataPartition(
  model_data$AttritionFlag,
  p = 0.80,
  list = FALSE
)

train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

cat("\nTraining rows:", nrow(train_data), "\n")
cat("Testing rows:", nrow(test_data), "\n")

cat("\nTraining target distribution:\n")
print(prop.table(table(train_data$AttritionFlag)))

cat("\nTesting target distribution:\n")
print(prop.table(table(test_data$AttritionFlag)))

# ------------------------------------------------------------
# 4. Separate predictors and target
# ------------------------------------------------------------

train_x <- train_data %>%
  select(-AttritionFlag)

train_y <- train_data$AttritionFlag

# ------------------------------------------------------------
# 5. Dummy encode categorical predictors
# ------------------------------------------------------------

dummy_model <- dummyVars(
  ~ .,
  data = train_x,
  fullRank = TRUE
)

train_x_encoded <- predict(
  dummy_model,
  train_x
)

train_x_encoded <- as.data.frame(train_x_encoded)

# ------------------------------------------------------------
# 6. Random Forest feature importance
# ------------------------------------------------------------

set.seed(123)

rf_importance_model <- randomForest(
  x = train_x_encoded,
  y = train_y,
  ntree = 100,
  importance = TRUE
)

importance_values <- importance(
  rf_importance_model,
  type = 2
)

importance_df <- data.frame(
  Feature = rownames(importance_values),
  Importance = importance_values[, "MeanDecreaseGini"],
  row.names = NULL
)

importance_df <- importance_df %>%
  arrange(desc(Importance))

cat("\n============================================\n")
cat("TOP FEATURES\n")
cat("============================================\n")

print(head(importance_df, 20))

# ------------------------------------------------------------
# 7. Select top original/engineered features
# ------------------------------------------------------------

# Use the original feature names rather than individual
# dummy-variable levels for categorical variables.

candidate_features <- c(
  "Plant",
  "Department",
  "JobRole",
  "Shift",
  "Education",
  "EducationField",
  "DistanceFromHome",
  "DailyRate",
  "HourlyRate",
  "MonthlyIncome",
  "MonthlyRate",
  "EnvironmentSatisfaction",
  "JobSatisfaction",
  "JobInvolvement",
  "YearsAtCompany",
  "MonthlyOvertimeHours",
  "SafetyIncidents_12M",
  "EngagementScore",
  "SatisfactionIndex",
  "EngagementLevel",
  "OvertimeLevel",
  "DistanceCategory",
  "TenureGroup"
)

# Calculate variable-level importance by aggregating
# dummy-variable importance.

feature_importance <- data.frame(
  Feature = candidate_features,
  Importance = 0
)

for (i in seq_along(candidate_features)) {

  feature_name <- candidate_features[i]

  matching_cols <- grep(
    paste0("^", feature_name),
    importance_df$Feature,
    value = TRUE
  )

  if (length(matching_cols) > 0) {

    feature_importance$Importance[i] <-
      sum(
        importance_df$Importance[
          importance_df$Feature %in% matching_cols
        ]
      )
  }
}

feature_importance <- feature_importance %>%
  arrange(desc(Importance))

cat("\n============================================\n")
cat("VARIABLE-LEVEL FEATURE IMPORTANCE\n")
cat("============================================\n")

print(feature_importance)

# ------------------------------------------------------------
# 8. Select top 15 features
# ------------------------------------------------------------

selected_features <- head(
  feature_importance$Feature,
  15
)

cat("\n============================================\n")
cat("SELECTED FEATURES\n")
cat("============================================\n")

print(selected_features)

# ------------------------------------------------------------
# 9. Make sure target is not selected
# ------------------------------------------------------------

if ("AttritionFlag" %in% selected_features) {
  stop("ERROR: Target variable selected as a predictor!")
}

cat("\nTarget leakage check: PASSED\n")

# ------------------------------------------------------------
# 10. Save outputs
# ------------------------------------------------------------

write.csv(
  feature_importance,
  "results/feature_importance_ranking.csv",
  row.names = FALSE
)

write.csv(
  data.frame(SelectedFeature = selected_features),
  "results/selected_features.csv",
  row.names = FALSE
)

saveRDS(
  train_data,
  "results/train_data.rds"
)

saveRDS(
  test_data,
  "results/test_data.rds"
)

saveRDS(
  dummy_model,
  "results/dummy_encoding_model.rds"
)

cat("\n============================================\n")
cat("FEATURE SELECTION COMPLETE\n")
cat("============================================\n")

cat("\nSaved files:\n")
cat("- results/feature_importance_ranking.csv\n")
cat("- results/selected_features.csv\n")
cat("- results/train_data.rds\n")
cat("- results/test_data.rds\n")
cat("- results/dummy_encoding_model.rds\n")