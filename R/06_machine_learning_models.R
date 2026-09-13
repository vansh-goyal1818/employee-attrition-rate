# ============================================================
# DA2 - Stage 5: Machine Learning Model Development
# Employee Attrition Prediction
# ============================================================

library(caret)
library(randomForest)
library(e1071)
library(MASS)
library(gbm)
library(xgboost)
library(glmnet)
library(kernlab)
library(naivebayes)
library(pROC)

cat("============================================\n")
cat("DA2 - MACHINE LEARNING MODEL DEVELOPMENT\n")
cat("============================================\n")

set.seed(123)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

train_data <- readRDS("results/train_data.rds")
test_data  <- readRDS("results/test_data.rds")

selected_features <- read.csv(
  "results/selected_features.csv",
  stringsAsFactors = FALSE
)$SelectedFeature



cat("\nSelected features:", length(selected_features), "\n")
print(selected_features)

# ------------------------------------------------------------
# 2. Prepare predictors and target
# ------------------------------------------------------------

train_x <- train_data[, selected_features, drop = FALSE]
test_x  <- test_data[, selected_features, drop = FALSE]

train_y <- factor(
  train_data$AttritionFlag,
  levels = c("No", "Yes")
)

test_y <- factor(
  test_data$AttritionFlag,
  levels = c("No", "Yes")
)

# ------------------------------------------------------------
# 3. Dummy encoding
# ------------------------------------------------------------

# Create dummy encoding using ONLY the selected features.
# This prevents previously saved preprocessing objects
# from introducing variables that were not selected.

dummy_model <- dummyVars(
  ~ .,
  data = train_x,
  fullRank = TRUE
)

train_x_encoded <- predict(
  dummy_model,
  train_x
)

test_x_encoded <- predict(
  dummy_model,
  test_x
)

train_x_encoded <- as.data.frame(train_x_encoded)
test_x_encoded  <- as.data.frame(test_x_encoded)

cat("\nDummy encoding completed.\n")
cat("Encoded predictor columns:",
    ncol(train_x_encoded), "\n")

# Save the correct DA2 encoding model
saveRDS(
  dummy_model,
  "results/dummy_encoding_model.rds"
)

# ------------------------------------------------------------
# 4. Preprocessing
# ------------------------------------------------------------

preprocess_model <- preProcess(
  train_x_encoded,
  method = c("center", "scale")
)

train_x_processed <- predict(
  preprocess_model,
  train_x_encoded
)

test_x_processed <- predict(
  preprocess_model,
  test_x_encoded
)

cat("\nTraining rows:", nrow(train_x_processed), "\n")
cat("Testing rows:", nrow(test_x_processed), "\n")
cat("Predictor columns:", ncol(train_x_processed), "\n")

# Save preprocessing object
saveRDS(
  preprocess_model,
  "results/preprocessing_model.rds"
)

# ------------------------------------------------------------
# Model control
# ------------------------------------------------------------

ctrl <- trainControl(
  method = "none",
  classProbs = TRUE
)

models <- list()

# ============================================================
# MODEL 1 - LOGISTIC REGRESSION
# ============================================================

cat("\n============================================\n")
cat("MODEL 1: LOGISTIC REGRESSION\n")
cat("============================================\n")

set.seed(123)

models$Logistic_Regression <- train(
  x = train_x_processed,
  y = train_y,
  method = "glm",
  family = binomial,
  trControl = ctrl
)

cat("Logistic Regression trained successfully.\n")

# ============================================================
# MODEL 2 - DECISION TREE
# ============================================================

cat("\n============================================\n")
cat("MODEL 2: DECISION TREE\n")
cat("============================================\n")

set.seed(123)

tree_grid <- expand.grid(
  cp = 0.01
)

models$Decision_Tree <- train(
  x = train_x_processed,
  y = train_y,
  method = "rpart",
  tuneGrid = tree_grid,
  trControl = ctrl
)

cat("Decision Tree trained successfully.\n")

# ============================================================
# MODEL 3 - RANDOM FOREST
# ============================================================

cat("\n============================================\n")
cat("MODEL 3: RANDOM FOREST\n")
cat("============================================\n")

set.seed(123)

models$Random_Forest <- train(
  x = train_x_processed,
  y = train_y,
  method = "rf",
  tuneGrid = expand.grid(
    mtry = max(
      1,
      floor(sqrt(ncol(train_x_processed)))
    )
  ),
  ntree = 200,
  trControl = ctrl
)

cat("Random Forest trained successfully.\n")

# ============================================================
# MODEL 4 - KNN
# ============================================================

cat("\n============================================\n")
cat("MODEL 4: KNN\n")
cat("============================================\n")

set.seed(123)

models$KNN <- train(
  x = train_x_processed,
  y = train_y,
  method = "knn",
  tuneGrid = expand.grid(
    k = 15
  ),
  trControl = ctrl
)

cat("KNN trained successfully.\n")

# ============================================================
# MODEL 5 - NAIVE BAYES
# ============================================================

cat("\n============================================\n")
cat("MODEL 5: NAIVE BAYES\n")
cat("============================================\n")

set.seed(123)

models$Naive_Bayes <- train(
  x = train_x_processed,
  y = train_y,
  method = "naive_bayes",
  tuneGrid = expand.grid(
    laplace = 1,
    usekernel = TRUE,
    adjust = 1
  ),
  trControl = ctrl
)

cat("Naive Bayes trained successfully.\n")

# ============================================================
# MODEL 6 - LDA
# ============================================================

cat("\n============================================\n")
cat("MODEL 6: LDA\n")
cat("============================================\n")

set.seed(123)

models$LDA <- train(
  x = train_x_processed,
  y = train_y,
  method = "lda",
  trControl = ctrl
)

cat("LDA trained successfully.\n")

# ============================================================
# MODEL 7 - ELASTIC NET
# ============================================================

cat("\n============================================\n")
cat("MODEL 7: ELASTIC NET\n")
cat("============================================\n")

set.seed(123)

models$Elastic_Net <- train(
  x = train_x_processed,
  y = train_y,
  method = "glmnet",
  tuneGrid = expand.grid(
    alpha = 0.5,
    lambda = 0.01
  ),
  family = "binomial",
  trControl = ctrl
)

cat("Elastic Net trained successfully.\n")

# ============================================================
# MODEL 8 - SVM
# ============================================================

cat("\n============================================\n")
cat("MODEL 8: SVM\n")
cat("============================================\n")

set.seed(123)

# SVM is computationally expensive on the full 40,001-row
# training set, so use a stratified 10,000-row sample.

svm_index <- createDataPartition(
  train_y,
  p = 10000 / length(train_y),
  list = FALSE
)

svm_x <- train_x_processed[svm_index, , drop = FALSE]
svm_y <- train_y[svm_index]

cat("SVM training rows:", nrow(svm_x), "\n")

svm_model <- e1071::svm(
  x = svm_x,
  y = svm_y,
  kernel = "linear",
  cost = 1,
  probability = TRUE,
  scale = FALSE
)

models$SVM <- svm_model

cat("SVM trained successfully.\n")
# ============================================================
# MODEL 9 - GRADIENT BOOSTING
# ============================================================

cat("\n============================================\n")
cat("MODEL 9: GRADIENT BOOSTING\n")
cat("============================================\n")

set.seed(123)

models$Gradient_Boosting <- train(
  x = train_x_processed,
  y = train_y,
  method = "gbm",
  tuneGrid = expand.grid(
    interaction.depth = 2,
    n.trees = 100,
    shrinkage = 0.1,
    n.minobsinnode = 10
  ),
  verbose = FALSE,
  trControl = ctrl
)

cat("Gradient Boosting trained successfully.\n")

# ============================================================
# MODEL 10 - XGBOOST
# ============================================================

cat("\n============================================\n")
cat("MODEL 10: XGBOOST\n")
cat("============================================\n")

set.seed(123)

train_y_xgb <- ifelse(
  train_y == "Yes",
  1,
  0
)

dtrain <- xgb.DMatrix(
  data = as.matrix(train_x_processed),
  label = train_y_xgb
)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 3,
  eta = 0.05,
  gamma = 0,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1
)

models$XGBoost <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

cat("XGBoost trained successfully.\n")

# ============================================================
# SAVE ALL MODELS
# ============================================================

saveRDS(
  models,
  "results/ml_models.rds"
)

cat("\n============================================\n")
cat("ALL MODELS TRAINED SUCCESSFULLY\n")
cat("============================================\n")

cat("\nModels saved:\n")

print(names(models))

cat("\nSaved:\n")
cat("results/ml_models.rds\n")
cat("results/preprocessing_model.rds\n")