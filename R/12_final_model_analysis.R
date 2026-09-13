# ============================================================
# DA2 - Stage 12: Final Model Analysis
# Final Model: Tuned XGBoost
# Classification Threshold: 0.20
# ============================================================

library(caret)
library(dplyr)
library(ggplot2)
library(pROC)
library(xgboost)

cat("===== FINAL MODEL ANALYSIS STARTED =====\n")

# ------------------------------------------------------------
# 1. Load data and model
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

final_model <- readRDS(
  "results/tuned_xgboost_model.rds"
)

# Final classification threshold
final_threshold <- 0.20

cat("Final model: Tuned XGBoost\n")
cat("Classification threshold:", final_threshold, "\n")
cat("Selected features:", length(selected_features), "\n")

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

# ------------------------------------------------------------
# 3. Generate probabilities
# ------------------------------------------------------------

test_matrix <- as.matrix(
  test_processed
)

test_dmatrix <- xgb.DMatrix(
  data = test_matrix
)

probabilities <- predict(
  final_model,
  newdata = test_dmatrix
)

# ------------------------------------------------------------
# 4. Apply final threshold
# ------------------------------------------------------------

predictions <- factor(
  ifelse(
    probabilities >= final_threshold,
    "Yes",
    "No"
  ),
  levels = c("No", "Yes")
)

cat("\nActual distribution:\n")
print(table(actual))

cat("\nPredicted distribution:\n")
print(table(predictions))

# ------------------------------------------------------------
# 5. Confusion matrix
# ------------------------------------------------------------

cm <- confusionMatrix(
  predictions,
  actual,
  positive = "Yes"
)

cat("\n========================================\n")
cat("===== FINAL CONFUSION MATRIX =====\n")
cat("========================================\n\n")

print(cm$table)

# ------------------------------------------------------------
# 6. Final performance metrics
# ------------------------------------------------------------

final_metrics <- data.frame(
  Model = "Tuned XGBoost",
  Threshold = final_threshold,
  Accuracy = as.numeric(
    cm$overall["Accuracy"]
  ),
  Kappa = as.numeric(
    cm$overall["Kappa"]
  ),
  Precision = as.numeric(
    cm$byClass["Precision"]
  ),
  Recall = as.numeric(
    cm$byClass["Sensitivity"]
  ),
  F1_Score = as.numeric(
    cm$byClass["F1"]
  ),
  Sensitivity = as.numeric(
    cm$byClass["Sensitivity"]
  ),
  Specificity = as.numeric(
    cm$byClass["Specificity"]
  ),
  stringsAsFactors = FALSE
)

# ROC-AUC
roc_obj <- pROC::roc(
  actual,
  probabilities,
  levels = c("No", "Yes"),
  direction = "<",
  quiet = TRUE
)

final_metrics$ROC_AUC <- as.numeric(
  pROC::auc(roc_obj)
)

cat("\n========================================\n")
cat("===== FINAL MODEL METRICS =====\n")
cat("========================================\n\n")

print(final_metrics)

write.csv(
  final_metrics,
  "results/final_model_metrics.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. ROC Curve
# ------------------------------------------------------------

roc_df <- data.frame(
  FalsePositiveRate =
    1 - roc_obj$specificities,
  TruePositiveRate =
    roc_obj$sensitivities
)

roc_plot <- ggplot(
  roc_df,
  aes(
    x = FalsePositiveRate,
    y = TruePositiveRate
  )
) +
  geom_line(linewidth = 1) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = paste0(
      "ROC Curve - Tuned XGBoost (AUC = ",
      round(
        as.numeric(auc(roc_obj)),
        3
      ),
      ")"
    ),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal()

ggsave(
  "results/final_roc_curve.png",
  roc_plot,
  width = 8,
  height = 6,
  dpi = 300
)

# ------------------------------------------------------------
# 8. Confusion Matrix Plot
# ------------------------------------------------------------

cm_df <- as.data.frame(
  cm$table
)

cm_plot <- ggplot(
  cm_df,
  aes(
    x = Reference,
    y = Prediction,
    fill = Freq
  )
) +
  geom_tile() +
  geom_text(
    aes(label = Freq),
    size = 6
  ) +
  labs(
    title = paste0(
      "Final Confusion Matrix - Threshold ",
      final_threshold
    ),
    x = "Actual",
    y = "Predicted"
  ) +
  theme_minimal()

ggsave(
  "results/final_confusion_matrix.png",
  cm_plot,
  width = 7,
  height = 6,
  dpi = 300
)

# ------------------------------------------------------------
# 9. Feature Importance
# ------------------------------------------------------------

importance_matrix <- xgb.importance(
  model = final_model
)

cat("\n========================================\n")
cat("===== FINAL MODEL FEATURE IMPORTANCE =====\n")
cat("========================================\n\n")

print(
  head(
    importance_matrix,
    15
  )
)

write.csv(
  importance_matrix,
  "results/final_feature_importance.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 10. Feature Importance Plot
# ------------------------------------------------------------

importance_top <- importance_matrix %>%
  head(15) %>%
  mutate(
    Feature = factor(
      Feature,
      levels = rev(Feature)
    )
  )

importance_plot <- ggplot(
  importance_top,
  aes(
    x = Feature,
    y = Gain
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 15 Features - Tuned XGBoost",
    x = "Feature",
    y = "Importance Gain"
  ) +
  theme_minimal()

ggsave(
  "results/final_feature_importance.png",
  importance_plot,
  width = 9,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# 11. Save final model information
# ------------------------------------------------------------

final_model_info <- data.frame(
  Model = "Tuned XGBoost",
  Threshold = final_threshold,
  ROC_AUC = final_metrics$ROC_AUC,
  Accuracy = final_metrics$Accuracy,
  Precision = final_metrics$Precision,
  Recall = final_metrics$Recall,
  F1_Score = final_metrics$F1_Score,
  Specificity = final_metrics$Specificity,
  stringsAsFactors = FALSE
)

write.csv(
  final_model_info,
  "results/final_model_summary.csv",
  row.names = FALSE
)

cat("\n========================================\n")
cat("===== STAGE 12 COMPLETED =====\n")
cat("========================================\n\n")

cat("Files created:\n")
cat("- final_model_metrics.csv\n")
cat("- final_model_summary.csv\n")
cat("- final_roc_curve.png\n")
cat("- final_confusion_matrix.png\n")
cat("- final_feature_importance.csv\n")
cat("- final_feature_importance.png\n")