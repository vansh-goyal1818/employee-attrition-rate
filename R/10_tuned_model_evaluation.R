# ============================================================
# DA2 - Stage 10: Tuned Model Evaluation
# ============================================================

library(caret)
library(dplyr)
library(pROC)
library(ggplot2)
library(tidyr)
library(xgboost)

cat("===== TUNED MODEL EVALUATION STARTED =====\n")

# ------------------------------------------------------------
# 1. Load objects
# ------------------------------------------------------------

train_data <- readRDS("results/train_data.rds")
test_data  <- readRDS("results/test_data.rds")

selected_features <- read.csv(
  "results/selected_features.csv",
  stringsAsFactors = FALSE
)$SelectedFeature

dummy_model <- readRDS(
  "results/dummy_encoding_model.rds"
)

preprocess_model <- readRDS(
  "results/preprocessing_model.rds"
)

tuned_rf <- readRDS(
  "results/tuned_random_forest.rds"
)

tuned_gbm <- readRDS(
  "results/tuned_gradient_boosting.rds"
)

cat("Train rows:", nrow(train_data), "\n")
cat("Test rows :", nrow(test_data), "\n")
cat("Selected features:", length(selected_features), "\n")

# ------------------------------------------------------------
# 2. Prepare predictors
# ------------------------------------------------------------

train_x <- train_data[, selected_features, drop = FALSE]
test_x  <- test_data[, selected_features, drop = FALSE]

train_dummy <- predict(
  dummy_model,
  newdata = train_x
)

test_dummy <- predict(
  dummy_model,
  newdata = test_x
)

train_processed <- predict(
  preprocess_model,
  train_dummy
)

test_processed <- predict(
  preprocess_model,
  test_dummy
)

cat(
  "Processed dimensions:",
  nrow(test_processed),
  "x",
  ncol(test_processed),
  "\n"
)

# ------------------------------------------------------------
# 3. Actual target
# ------------------------------------------------------------

actual <- factor(
  as.character(test_data$AttritionFlag),
  levels = c("No", "Yes")
)

cat("\nActual test distribution:\n")
print(table(actual))

# ------------------------------------------------------------
# 4. Generic caret model evaluation
# ------------------------------------------------------------

evaluate_caret <- function(model, model_name) {

  cat("\n----------------------------------------\n")
  cat("Evaluating:", model_name, "\n")

  # Class prediction
  pred_class <- predict(
    model,
    newdata = test_processed
  )

  pred_class <- factor(
    as.character(pred_class),
    levels = c("No", "Yes")
  )

  # Probability prediction
  prob_df <- predict(
    model,
    newdata = test_processed,
    type = "prob"
  )

  cat("Probability columns:",
      paste(colnames(prob_df), collapse = ", "),
      "\n")

  # Explicitly select Yes probability
  pred_prob <- prob_df[, "Yes"]

  # Convert to numeric safely
  pred_prob <- as.numeric(pred_prob)

  cat(
    "Probability range:",
    min(pred_prob, na.rm = TRUE),
    "to",
    max(pred_prob, na.rm = TRUE),
    "\n"
  )

  # Confusion matrix
  cm <- confusionMatrix(
    pred_class,
    actual,
    positive = "Yes"
  )

  # ROC
  roc_obj <- pROC::roc(
    response = actual,
    predictor = pred_prob,
    levels = c("No", "Yes"),
    direction = "<",
    quiet = TRUE
  )

  data.frame(
    Model = model_name,
    Accuracy = unname(
      cm$overall["Accuracy"]
    ),
    Kappa = unname(
      cm$overall["Kappa"]
    ),
    Precision = unname(
      cm$byClass["Precision"]
    ),
    Recall = unname(
      cm$byClass["Sensitivity"]
    ),
    F1_Score = unname(
      cm$byClass["F1"]
    ),
    Sensitivity = unname(
      cm$byClass["Sensitivity"]
    ),
    Specificity = unname(
      cm$byClass["Specificity"]
    ),
    ROC_AUC = as.numeric(
      pROC::auc(roc_obj)
    )
  )
}

# ------------------------------------------------------------
# 5. Tuned Random Forest
# ------------------------------------------------------------

rf_results <- evaluate_caret(
  tuned_rf,
  "Tuned_Random_Forest"
)

print(rf_results)

# ------------------------------------------------------------
# 6. Tuned Gradient Boosting
# ------------------------------------------------------------

gbm_results <- evaluate_caret(
  tuned_gbm,
  "Tuned_Gradient_Boosting"
)

print(gbm_results)

# ------------------------------------------------------------
# 7. XGBoost tuning results
# ------------------------------------------------------------

cat("\nChecking XGBoost tuning results...\n")

xgb_tuning <- read.csv(
  "results/xgboost_tuning_results.csv",
  stringsAsFactors = FALSE
)

print(xgb_tuning)

best_xgb <- xgb_tuning %>%
  arrange(desc(Train_AUC)) %>%
  slice(1)

cat("\nBest XGBoost configuration:\n")
print(best_xgb)

# ------------------------------------------------------------
# 8. Prepare XGBoost labels
# ------------------------------------------------------------

xgb_train_labels <- ifelse(
  as.character(train_data$AttritionFlag) == "Yes",
  1,
  0
)

xgb_test_labels <- ifelse(
  as.character(test_data$AttritionFlag) == "Yes",
  1,
  0
)

cat("\nXGBoost train labels:\n")
print(table(xgb_train_labels))

cat("\nXGBoost test labels:\n")
print(table(xgb_test_labels))

# ------------------------------------------------------------
# 9. Train tuned XGBoost
# ------------------------------------------------------------

set.seed(42)

xgb_train <- xgb.DMatrix(
  data = as.matrix(train_processed),
  label = xgb_train_labels
)

xgb_test <- xgb.DMatrix(
  data = as.matrix(test_processed),
  label = xgb_test_labels
)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = as.integer(best_xgb$max_depth),
  eta = best_xgb$eta,
  min_child_weight = best_xgb$min_child_weight,
  subsample = 0.8,
  colsample_bytree = 0.8
)

tuned_xgb <- xgb.train(
  params = xgb_params,
  data = xgb_train,
  nrounds = 100,
  verbose = 0
)

saveRDS(
  tuned_xgb,
  "results/tuned_xgboost_model.rds"
)

cat("\nTuned XGBoost model saved.\n")

# ------------------------------------------------------------
# 10. Evaluate XGBoost
# ------------------------------------------------------------

xgb_prob <- predict(
  tuned_xgb,
  newdata = xgb_test
)

xgb_class <- factor(
  ifelse(
    xgb_prob >= 0.5,
    "Yes",
    "No"
  ),
  levels = c("No", "Yes")
)

cat("\nXGBoost predicted distribution:\n")
print(table(xgb_class))

cat(
  "XGBoost probability range:",
  min(xgb_prob),
  "to",
  max(xgb_prob),
  "\n"
)

xgb_cm <- confusionMatrix(
  xgb_class,
  actual,
  positive = "Yes"
)

xgb_roc <- pROC::roc(
  response = actual,
  predictor = as.numeric(xgb_prob),
  levels = c("No", "Yes"),
  direction = "<",
  quiet = TRUE
)

xgb_results <- data.frame(
  Model = "Tuned_XGBoost",
  Accuracy = unname(
    xgb_cm$overall["Accuracy"]
  ),
  Kappa = unname(
    xgb_cm$overall["Kappa"]
  ),
  Precision = unname(
    xgb_cm$byClass["Precision"]
  ),
  Recall = unname(
    xgb_cm$byClass["Sensitivity"]
  ),
  F1_Score = unname(
    xgb_cm$byClass["F1"]
  ),
  Sensitivity = unname(
    xgb_cm$byClass["Sensitivity"]
  ),
  Specificity = unname(
    xgb_cm$byClass["Specificity"]
  ),
  ROC_AUC = as.numeric(
    pROC::auc(xgb_roc)
  )
)

print(xgb_results)

# ------------------------------------------------------------
# 11. Final comparison
# ------------------------------------------------------------

tuned_comparison <- bind_rows(
  rf_results,
  gbm_results,
  xgb_results
) %>%
  arrange(desc(ROC_AUC))

cat("\n========================================\n")
cat("===== TUNED MODEL COMPARISON =====\n")
cat("========================================\n\n")

print(tuned_comparison)

write.csv(
  tuned_comparison,
  "results/tuned_model_comparison.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 12. Best model
# ------------------------------------------------------------

best_model <- tuned_comparison %>%
  slice(1)

cat("\n========================================\n")
cat("===== BEST TUNED MODEL =====\n")
cat("========================================\n\n")

print(best_model)

write.csv(
  best_model,
  "results/final_tuned_model.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 13. Visualization
# ------------------------------------------------------------

plot_data <- tuned_comparison %>%
  select(
    Model,
    ROC_AUC,
    F1_Score,
    Accuracy
  ) %>%
  pivot_longer(
    cols = c(
      ROC_AUC,
      F1_Score,
      Accuracy
    ),
    names_to = "Metric",
    values_to = "Score"
  )

p <- ggplot(
  plot_data,
  aes(
    x = reorder(Model, Score),
    y = Score,
    fill = Metric
  )
) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Tuned Model Performance Comparison",
    x = "Model",
    y = "Score"
  ) +
  theme_minimal()

ggsave(
  "results/tuned_model_comparison.png",
  p,
  width = 10,
  height = 6,
  dpi = 300
)

cat("\nComparison plot saved.\n")
cat("\n===== STAGE 10 COMPLETED =====\n")