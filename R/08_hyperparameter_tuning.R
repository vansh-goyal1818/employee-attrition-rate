# ============================================================
# DA2 - Stage 8: Hyperparameter Tuning
# Employee Attrition Prediction
# ============================================================

library(caret)
library(randomForest)
library(gbm)
library(xgboost)
library(pROC)

cat("============================================\n")
cat("DA2 - HYPERPARAMETER TUNING\n")
cat("============================================\n")

set.seed(123)

# ------------------------------------------------------------
# 1. LOAD DATA AND MODELS
# ------------------------------------------------------------

train_data <- readRDS("results/train_data.rds")
test_data  <- readRDS("results/test_data.rds")

dummy_model <- readRDS(
  "results/dummy_encoding_model.rds"
)

preprocess_model <- readRDS(
  "results/preprocessing_model.rds"
)

selected_features <- read.csv(
  "results/selected_features.csv",
  stringsAsFactors = FALSE
)$SelectedFeature

cat("\nSelected features:", length(selected_features), "\n")
print(selected_features)

# ------------------------------------------------------------
# 2. PREPARE DATA
# ------------------------------------------------------------

train_x <- train_data[
  ,
  selected_features,
  drop = FALSE
]

test_x <- test_data[
  ,
  selected_features,
  drop = FALSE
]

train_y <- factor(
  train_data$AttritionFlag,
  levels = c("No", "Yes")
)

test_y <- factor(
  test_data$AttritionFlag,
  levels = c("No", "Yes")
)

# ------------------------------------------------------------
# 3. DUMMY ENCODING
# ------------------------------------------------------------

train_x_encoded <- predict(
  dummy_model,
  train_x
)

test_x_encoded <- predict(
  dummy_model,
  test_x
)

train_x_encoded <- as.data.frame(
  train_x_encoded
)

test_x_encoded <- as.data.frame(
  test_x_encoded
)

# ------------------------------------------------------------
# 4. PREPROCESSING
# ------------------------------------------------------------

train_x_processed <- predict(
  preprocess_model,
  train_x_encoded
)

test_x_processed <- predict(
  preprocess_model,
  test_x_encoded
)

train_x_processed <- as.data.frame(
  train_x_processed
)

test_x_processed <- as.data.frame(
  test_x_processed
)

cat("\nTraining rows:", nrow(train_x_processed), "\n")
cat("Testing rows:", nrow(test_x_processed), "\n")
cat("Predictor columns:", ncol(train_x_processed), "\n")

# ============================================================
# 5. CROSS-VALIDATION CONTROL
# ============================================================

ctrl <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# ============================================================
# 6. RANDOM FOREST TUNING
# ============================================================

cat("\n============================================\n")
cat("RANDOM FOREST TUNING\n")
cat("============================================\n")

set.seed(123)

# Stratified subset for faster tuning
rf_index <- createDataPartition(
  train_y,
  p = 10000 / length(train_y),
  list = FALSE
)

rf_x_tune <- train_x_processed[
  rf_index,
  ,
  drop = FALSE
]

rf_y_tune <- train_y[rf_index]

cat(
  "RF tuning rows:",
  nrow(rf_x_tune),
  "\n"
)

# Small but meaningful tuning grid
rf_grid <- expand.grid(
  mtry = c(5, 10)
)

rf_tuned <- train(
  x = rf_x_tune,
  y = rf_y_tune,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = rf_grid,
  ntree = 100
)

cat("\nRandom Forest tuning results:\n")
print(rf_tuned$results)

cat("\nBest Random Forest parameters:\n")
print(rf_tuned$bestTune)

saveRDS(
  rf_tuned,
  "results/tuned_random_forest.rds"
)

# ============================================================
# 7. GRADIENT BOOSTING TUNING
# ============================================================

cat("\n============================================\n")
cat("GRADIENT BOOSTING TUNING\n")
cat("============================================\n")

set.seed(123)

# Stratified subset for faster tuning
gbm_index <- createDataPartition(
  train_y,
  p = 10000 / length(train_y),
  list = FALSE
)

gbm_x_tune <- train_x_processed[
  gbm_index,
  ,
  drop = FALSE
]

gbm_y_tune <- train_y[gbm_index]

cat(
  "GBM tuning rows:",
  nrow(gbm_x_tune),
  "\n"
)

# Reduced but meaningful tuning grid
gbm_grid <- expand.grid(
  interaction.depth = c(1, 3),
  n.trees = c(100, 200),
  shrinkage = c(0.1),
  n.minobsinnode = c(10)
)

gbm_tuned <- train(
  x = gbm_x_tune,
  y = gbm_y_tune,
  method = "gbm",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = gbm_grid,
  verbose = FALSE
)

cat("\nGradient Boosting tuning results:\n")
print(gbm_tuned$results)

cat("\nBest Gradient Boosting parameters:\n")
print(gbm_tuned$bestTune)

saveRDS(
  gbm_tuned,
  "results/tuned_gradient_boosting.rds"
)

# ============================================================
# 8. XGBOOST TUNING
# ============================================================

cat("\n============================================\n")
cat("XGBOOST TUNING\n")
cat("============================================\n")

set.seed(123)

# Use same 10,000-row stratified subset
xgb_index <- createDataPartition(
  train_y,
  p = 10000 / length(train_y),
  list = FALSE
)

xgb_x_tune <- train_x_processed[
  xgb_index,
  ,
  drop = FALSE
]

xgb_y_tune <- train_y[xgb_index]

xgb_y_binary <- ifelse(
  xgb_y_tune == "Yes",
  1,
  0
)

dtrain_xgb <- xgb.DMatrix(
  data = as.matrix(xgb_x_tune),
  label = xgb_y_binary
)

cat(
  "XGBoost tuning rows:",
  nrow(xgb_x_tune),
  "\n"
)

# Small practical tuning grid
xgb_grid <- expand.grid(
  max_depth = c(3, 5),
  eta = c(0.05, 0.1),
  min_child_weight = c(1, 5)
)

xgb_results <- data.frame()

for (i in seq_len(nrow(xgb_grid))) {

  cat(
    "Testing XGBoost configuration",
    i,
    "of",
    nrow(xgb_grid),
    "\n"
  )

  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = xgb_grid$max_depth[i],
    eta = xgb_grid$eta[i],
    min_child_weight =
      xgb_grid$min_child_weight[i],
    subsample = 0.8,
    colsample_bytree = 0.8
  )

  model <- xgb.train(
    params = params,
    data = dtrain_xgb,
    nrounds = 100,
    verbose = 0
  )

  train_prob <- predict(
    model,
    dtrain_xgb
  )

  auc_value <- as.numeric(
    pROC::auc(
      pROC::roc(
        xgb_y_binary,
        train_prob,
        quiet = TRUE
      )
    )
  )

  xgb_results <- rbind(
    xgb_results,
    data.frame(
      max_depth =
        xgb_grid$max_depth[i],
      eta =
        xgb_grid$eta[i],
      min_child_weight =
        xgb_grid$min_child_weight[i],
      subsample = 0.8,
      colsample_bytree = 0.8,
      Train_AUC = auc_value
    )
  )
}

xgb_results <- xgb_results[
  order(
    -xgb_results$Train_AUC
  ),
]

cat("\nXGBoost tuning results:\n")
print(xgb_results)

cat("\nBest XGBoost parameters:\n")
print(xgb_results[1, ])

write.csv(
  xgb_results,
  "results/xgboost_tuning_results.csv",
  row.names = FALSE
)

# ============================================================
# 9. SAVE TUNING SUMMARY
# ============================================================

tuning_summary <- list(
  Random_Forest =
    rf_tuned$bestTune,

  Gradient_Boosting =
    gbm_tuned$bestTune,

  XGBoost =
    xgb_results[1, ]
)

saveRDS(
  tuning_summary,
  "results/tuning_summary.rds"
)

# ============================================================
# 10. FINAL MESSAGE
# ============================================================

cat("\n============================================\n")
cat("HYPERPARAMETER TUNING COMPLETE\n")
cat("============================================\n")

cat("\nTuned models saved:\n")
cat("results/tuned_random_forest.rds\n")
cat("results/tuned_gradient_boosting.rds\n")
cat("results/xgboost_tuning_results.csv\n")
cat("results/tuning_summary.rds\n")