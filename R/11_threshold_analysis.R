# ============================================================
# DA2 - Stage 11: Threshold Analysis
# ROC + Precision-Recall + Optimal Threshold
# ============================================================

library(caret)
library(dplyr)
library(pROC)
library(ggplot2)

cat("===== THRESHOLD ANALYSIS STARTED =====\n")

# ------------------------------------------------------------
# 1. Load data and models
# ------------------------------------------------------------

train_data <- readRDS(
  "results/train_data.rds"
)

test_data <- readRDS(
  "results/test_data.rds"
)

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

tuned_xgb <- readRDS(
  "results/tuned_xgboost_model.rds"
)

# ------------------------------------------------------------
# 2. Prepare test data
# ------------------------------------------------------------

test_x <- test_data[
  ,
  selected_features,
  drop = FALSE
]

test_dummy <- predict(
  dummy_model,
  newdata = test_x
)

test_processed <- predict(
  preprocess_model,
  test_dummy
)

actual <- factor(
  as.character(test_data$AttritionFlag),
  levels = c("No", "Yes")
)

cat("\nActual test distribution:\n")
print(table(actual))

# ------------------------------------------------------------
# 3. Get probabilities from all three tuned models
# ------------------------------------------------------------

rf_prob <- predict(
  tuned_rf,
  newdata = test_processed,
  type = "prob"
)[, "Yes"]

gbm_prob <- predict(
  tuned_gbm,
  newdata = test_processed,
  type = "prob"
)[, "Yes"]

xgb_prob <- predict(
  tuned_xgb,
  newdata = as.matrix(test_processed)
)

# ------------------------------------------------------------
# 4. ROC curves
# ------------------------------------------------------------

rf_roc <- roc(
  actual,
  rf_prob,
  levels = c("No", "Yes"),
  direction = "<",
  quiet = TRUE
)

gbm_roc <- roc(
  actual,
  gbm_prob,
  levels = c("No", "Yes"),
  direction = "<",
  quiet = TRUE
)

xgb_roc <- roc(
  actual,
  xgb_prob,
  levels = c("No", "Yes"),
  direction = "<",
  quiet = TRUE
)

cat("\n===== ROC-AUC =====\n")

cat(
  "Random Forest:",
  round(as.numeric(auc(rf_roc)), 4),
  "\n"
)

cat(
  "Gradient Boosting:",
  round(as.numeric(auc(gbm_roc)), 4),
  "\n"
)

cat(
  "XGBoost:",
  round(as.numeric(auc(xgb_roc)), 4),
  "\n"
)

# ------------------------------------------------------------
# 5. ROC curve visualization
# ------------------------------------------------------------

roc_data <- bind_rows(
  data.frame(
    FPR = 1 - rf_roc$specificities,
    TPR = rf_roc$sensitivities,
    Model = "Random Forest"
  ),
  data.frame(
    FPR = 1 - gbm_roc$specificities,
    TPR = gbm_roc$sensitivities,
    Model = "Gradient Boosting"
  ),
  data.frame(
    FPR = 1 - xgb_roc$specificities,
    TPR = xgb_roc$sensitivities,
    Model = "XGBoost"
  )
)

roc_plot <- ggplot(
  roc_data,
  aes(
    x = FPR,
    y = TPR,
    linetype = Model
  )
) +
  geom_line(linewidth = 1) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "ROC Curve - Tuned Models",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal()

ggsave(
  "results/tuned_roc_curves.png",
  roc_plot,
  width = 9,
  height = 6,
  dpi = 300
)

# ------------------------------------------------------------
# 6. Threshold analysis function
# ------------------------------------------------------------

threshold_analysis <- function(
  actual,
  probabilities,
  model_name
) {

  thresholds <- seq(
    0.10,
    0.90,
    by = 0.05
  )

  results <- data.frame()

  for (threshold in thresholds) {

    predictions <- factor(
      ifelse(
        probabilities >= threshold,
        "Yes",
        "No"
      ),
      levels = c("No", "Yes")
    )

    cm <- confusionMatrix(
      predictions,
      actual,
      positive = "Yes"
    )

    results <- rbind(
      results,
      data.frame(
        Model = model_name,
        Threshold = threshold,
        Accuracy = as.numeric(
          cm$overall["Accuracy"]
        ),
        Precision = as.numeric(
          cm$byClass["Precision"]
        ),
        Recall = as.numeric(
          cm$byClass["Sensitivity"]
        ),
        F1 = as.numeric(
          cm$byClass["F1"]
        ),
        Specificity = as.numeric(
          cm$byClass["Specificity"]
        )
      )
    )
  }

  results
}

# ------------------------------------------------------------
# 7. Run threshold analysis
# ------------------------------------------------------------

rf_thresholds <- threshold_analysis(
  actual,
  rf_prob,
  "Random Forest"
)

gbm_thresholds <- threshold_analysis(
  actual,
  gbm_prob,
  "Gradient Boosting"
)

xgb_thresholds <- threshold_analysis(
  actual,
  xgb_prob,
  "XGBoost"
)

all_thresholds <- bind_rows(
  rf_thresholds,
  gbm_thresholds,
  xgb_thresholds
)

write.csv(
  all_thresholds,
  "results/threshold_analysis.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 8. Find best F1 threshold
# ------------------------------------------------------------

best_thresholds <- all_thresholds %>%
  group_by(Model) %>%
  arrange(desc(F1)) %>%
  slice(1) %>%
  ungroup()

cat("\n========================================\n")
cat("===== BEST THRESHOLD BY F1 =====\n")
cat("========================================\n\n")

print(best_thresholds)

write.csv(
  best_thresholds,
  "results/best_thresholds.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 9. Precision-Recall data
# ------------------------------------------------------------

pr_curve <- function(
  actual,
  probabilities,
  model_name
) {

  thresholds <- seq(
    0.05,
    0.95,
    by = 0.01
  )

  output <- data.frame()

  for (threshold in thresholds) {

    predictions <- ifelse(
      probabilities >= threshold,
      "Yes",
      "No"
    )

    tp <- sum(
      predictions == "Yes" &
      actual == "Yes"
    )

    fp <- sum(
      predictions == "Yes" &
      actual == "No"
    )

    fn <- sum(
      predictions == "No" &
      actual == "Yes"
    )

    precision <- ifelse(
      tp + fp == 0,
      0,
      tp / (tp + fp)
    )

    recall <- ifelse(
      tp + fn == 0,
      0,
      tp / (tp + fn)
    )

    output <- rbind(
      output,
      data.frame(
        Precision = precision,
        Recall = recall,
        Threshold = threshold,
        Model = model_name
      )
    )
  }

  output
}

pr_data <- bind_rows(
  pr_curve(
    actual,
    rf_prob,
    "Random Forest"
  ),
  pr_curve(
    actual,
    gbm_prob,
    "Gradient Boosting"
  ),
  pr_curve(
    actual,
    xgb_prob,
    "XGBoost"
  )
)

write.csv(
  pr_data,
  "results/precision_recall_data.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 10. Precision-Recall visualization
# ------------------------------------------------------------

pr_plot <- ggplot(
  pr_data,
  aes(
    x = Recall,
    y = Precision,
    linetype = Model
  )
) +
  geom_line(linewidth = 1) +
  labs(
    title = "Precision-Recall Curve - Tuned Models",
    x = "Recall",
    y = "Precision"
  ) +
  theme_minimal()

ggsave(
  "results/tuned_precision_recall_curves.png",
  pr_plot,
  width = 9,
  height = 6,
  dpi = 300
)

# ------------------------------------------------------------
# 11. Final summary
# ------------------------------------------------------------

cat("\n========================================\n")
cat("===== THRESHOLD ANALYSIS COMPLETED =====\n")
cat("========================================\n")

cat("\nFiles created:\n")
cat("1. tuned_roc_curves.png\n")
cat("2. tuned_precision_recall_curves.png\n")
cat("3. threshold_analysis.csv\n")
cat("4. best_thresholds.csv\n")
cat("5. precision_recall_data.csv\n")

cat("\n===== STAGE 11 COMPLETED =====\n")