# ============================================================
# DA2 - Stage 7: Model Evaluation & Comparative Analysis
# ============================================================

library(caret)
library(pROC)
library(dplyr)
library(ggplot2)
library(tidyr)

cat("============================================\n")
cat("DA2 - MODEL EVALUATION & COMPARATIVE ANALYSIS\n")
cat("============================================\n")

# ------------------------------------------------------------
# 1. LOAD DATA, FEATURES AND MODELS
# ------------------------------------------------------------

train_data <- readRDS("results/train_data.rds")
test_data  <- readRDS("results/test_data.rds")
models     <- readRDS("results/ml_models.rds")

selected_features <- read.csv(
  "results/selected_features.csv",
  stringsAsFactors = FALSE
)$SelectedFeature

cat("\nModels loaded:", length(models), "\n")
cat("Selected features:", length(selected_features), "\n")

# ------------------------------------------------------------
# 2. RECREATE EXACT SAME ENCODING AS STAGE 6
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

dummy_model <- readRDS(
  "results/dummy_encoding_model.rds"
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

preprocess_model <- readRDS(
  "results/preprocessing_model.rds"
)

train_x_processed <- predict(
  preprocess_model,
  train_x_encoded
)

test_x_processed <- predict(
  preprocess_model,
  test_x_encoded
)

train_x_processed <- as.data.frame(train_x_processed)
test_x_processed  <- as.data.frame(test_x_processed)

cat("Test rows:", nrow(test_x_processed), "\n")
cat("Test columns:", ncol(test_x_processed), "\n")

# ------------------------------------------------------------
# 3. STORAGE FOR RESULTS
# ------------------------------------------------------------

results_list <- list()
confusion_list <- list()

# ------------------------------------------------------------
# 4. EVALUATE CARET MODELS
# ------------------------------------------------------------

caret_models <- c(
  "Logistic_Regression",
  "Decision_Tree",
  "Random_Forest",
  "KNN",
  "Naive_Bayes",
  "LDA",
  "Elastic_Net",
  "Gradient_Boosting"
)

for (model_name in caret_models) {

  cat("\nEvaluating:", model_name, "\n")

  model <- models[[model_name]]

  # Class predictions
  predictions <- predict(
    model,
    newdata = test_x_processed
  )

  predictions <- factor(
    predictions,
    levels = c("No", "Yes")
  )

  # Confusion matrix
  cm <- confusionMatrix(
    predictions,
    test_y,
    positive = "Yes"
  )

  accuracy <- as.numeric(
    cm$overall["Accuracy"]
  )

  precision <- as.numeric(
    cm$byClass["Precision"]
  )

  recall <- as.numeric(
    cm$byClass["Recall"]
  )

  f1 <- as.numeric(
    cm$byClass["F1"]
  )

  # Probability predictions
  probabilities <- tryCatch(
    {
      prob <- predict(
        model,
        newdata = test_x_processed,
        type = "prob"
      )

      if ("Yes" %in% colnames(prob)) {
        as.numeric(prob[, "Yes"])
      } else {
        rep(NA_real_, nrow(test_x_processed))
      }

    },
    error = function(e) {
      rep(NA_real_, nrow(test_x_processed))
    }
  )

  # ROC-AUC
  roc_auc <- NA_real_

  if (!all(is.na(probabilities))) {

    roc_obj <- roc(
      response = test_y,
      predictor = probabilities,
      levels = c("No", "Yes"),
      direction = "<",
      quiet = TRUE
    )

    roc_auc <- as.numeric(
      auc(roc_obj)
    )
  }

  results_list[[model_name]] <- data.frame(
    Model = model_name,
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1,
    ROC_AUC = roc_auc
  )

  confusion_list[[model_name]] <- cm

  cat(
    "Accuracy:",
    round(accuracy, 4),
    "| F1:",
    round(f1, 4),
    "| ROC-AUC:",
    round(roc_auc, 4),
    "\n"
  )
}

# ------------------------------------------------------------
# 5. EVALUATE SVM
# ------------------------------------------------------------

cat("\nEvaluating: SVM\n")

svm_model <- models$SVM

svm_predictions <- predict(
  svm_model,
  test_x_processed
)

svm_predictions <- factor(
  svm_predictions,
  levels = c("No", "Yes")
)

svm_cm <- confusionMatrix(
  svm_predictions,
  test_y,
  positive = "Yes"
)

svm_accuracy <- as.numeric(
  svm_cm$overall["Accuracy"]
)

svm_precision <- as.numeric(
  svm_cm$byClass["Precision"]
)

svm_recall <- as.numeric(
  svm_cm$byClass["Recall"]
)

svm_f1 <- as.numeric(
  svm_cm$byClass["F1"]
)

svm_probabilities <- tryCatch(
  {
    svm_prob <- attr(
      predict(
        svm_model,
        test_x_processed,
        probability = TRUE
      ),
      "probabilities"
    )

    if (!is.null(svm_prob) &&
        "Yes" %in% colnames(svm_prob)) {

      as.numeric(svm_prob[, "Yes"])

    } else {
      rep(NA_real_, nrow(test_x_processed))
    }

  },
  error = function(e) {
    rep(NA_real_, nrow(test_x_processed))
  }
)

svm_auc <- NA_real_

if (!all(is.na(svm_probabilities))) {

  svm_roc <- roc(
    test_y,
    svm_probabilities,
    levels = c("No", "Yes"),
    direction = "<",
    quiet = TRUE
  )

  svm_auc <- as.numeric(
    auc(svm_roc)
  )
}

results_list[["SVM"]] <- data.frame(
  Model = "SVM",
  Accuracy = svm_accuracy,
  Precision = svm_precision,
  Recall = svm_recall,
  F1_Score = svm_f1,
  ROC_AUC = svm_auc
)

confusion_list[["SVM"]] <- svm_cm

cat(
  "Accuracy:",
  round(svm_accuracy, 4),
  "| F1:",
  round(svm_f1, 4),
  "| ROC-AUC:",
  round(svm_auc, 4),
  "\n"
)

# ------------------------------------------------------------
# 6. EVALUATE XGBOOST
# ------------------------------------------------------------

cat("\nEvaluating: XGBoost\n")

xgb_model <- models$XGBoost

xgb_probabilities <- predict(
  xgb_model,
  newdata = as.matrix(test_x_processed)
)

xgb_predictions <- ifelse(
  xgb_probabilities >= 0.5,
  "Yes",
  "No"
)

xgb_predictions <- factor(
  xgb_predictions,
  levels = c("No", "Yes")
)

xgb_cm <- confusionMatrix(
  xgb_predictions,
  test_y,
  positive = "Yes"
)

xgb_accuracy <- as.numeric(
  xgb_cm$overall["Accuracy"]
)

xgb_precision <- as.numeric(
  xgb_cm$byClass["Precision"]
)

xgb_recall <- as.numeric(
  xgb_cm$byClass["Recall"]
)

xgb_f1 <- as.numeric(
  xgb_cm$byClass["F1"]
)

xgb_roc <- roc(
  test_y,
  xgb_probabilities,
  levels = c("No", "Yes"),
  direction = "<",
  quiet = TRUE
)

xgb_auc <- as.numeric(
  auc(xgb_roc)
)

results_list[["XGBoost"]] <- data.frame(
  Model = "XGBoost",
  Accuracy = xgb_accuracy,
  Precision = xgb_precision,
  Recall = xgb_recall,
  F1_Score = xgb_f1,
  ROC_AUC = xgb_auc
)

confusion_list[["XGBoost"]] <- xgb_cm

cat(
  "Accuracy:",
  round(xgb_accuracy, 4),
  "| F1:",
  round(xgb_f1, 4),
  "| ROC-AUC:",
  round(xgb_auc, 4),
  "\n"
)

# ------------------------------------------------------------
# 7. COMBINE RESULTS
# ------------------------------------------------------------

evaluation_results <- bind_rows(
  results_list
)

evaluation_results <- evaluation_results %>%
  arrange(desc(F1_Score))

cat("\n============================================\n")
cat("MODEL COMPARISON\n")
cat("============================================\n")

print(evaluation_results)

# ------------------------------------------------------------
# 8. SAVE EVALUATION TABLE
# ------------------------------------------------------------

write.csv(
  evaluation_results,
  "results/model_evaluation_results.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 9. BEST MODEL
# ------------------------------------------------------------

best_model <- evaluation_results %>%
  slice(1)

cat("\n============================================\n")
cat("BEST MODEL BASED ON F1 SCORE\n")
cat("============================================\n")

print(best_model)

# ------------------------------------------------------------
# 10. MODEL COMPARISON PLOT
# ------------------------------------------------------------

plot_data <- evaluation_results %>%
  pivot_longer(
    cols = c(
      Accuracy,
      Precision,
      Recall,
      F1_Score,
      ROC_AUC
    ),
    names_to = "Metric",
    values_to = "Score"
  )

comparison_plot <- ggplot(
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
    title = "Comparison of Machine Learning Models",
    x = "Model",
    y = "Score"
  ) +
  theme_minimal()

ggsave(
  "results/model_comparison.png",
  comparison_plot,
  width = 10,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# 11. SAVE CONFUSION MATRICES
# ------------------------------------------------------------

png(
  "results/confusion_matrices_stage7.png",
  width = 1800,
  height = 1400,
  res = 180
)

par(
  mfrow = c(2, 5),
  mar = c(3, 3, 3, 1)
)

for (model_name in names(confusion_list)) {

  cm_table <- confusion_list[[model_name]]$table

  fourfoldplot(
    cm_table,
    color = c("white", "grey80"),
    main = model_name
  )
}

dev.off()

cat("\n============================================\n")
cat("STAGE 7 COMPLETED SUCCESSFULLY\n")
cat("============================================\n")

cat("\nSaved files:\n")
cat("results/model_evaluation_results.csv\n")
cat("results/model_comparison.png\n")
cat("results/confusion_matrices_stage7.png\n")