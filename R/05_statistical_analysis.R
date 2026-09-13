# ============================================================
# DA2 - Stage 4: Statistical Analysis
# ============================================================

library(dplyr)

cat("============================================\n")
cat("DA2 - STATISTICAL ANALYSIS\n")
cat("============================================\n")

# ------------------------------------------------------------
# 1. Load engineered dataset
# ------------------------------------------------------------

hr_data <- readRDS("results/engineered_hr_data.rds")

cat("\nDataset:", nrow(hr_data), "rows x", ncol(hr_data), "columns\n")

# ------------------------------------------------------------
# 2. Wilcoxon tests
# ------------------------------------------------------------

numeric_features <- c(
  "EngagementScore",
  "MonthlyOvertimeHours",
  "EnvironmentSatisfaction",
  "JobSatisfaction",
  "JobInvolvement",
  "DistanceFromHome",
  "MonthlyIncome",
  "YearsAtCompany",
  "SatisfactionIndex"
)

wilcoxon_results <- data.frame()

for (feature in numeric_features) {

  formula <- as.formula(
    paste(feature, "~ AttritionFlag")
  )

  test <- wilcox.test(
    formula,
    data = hr_data
  )

  wilcoxon_results <- rbind(
    wilcoxon_results,
    data.frame(
      Feature = feature,
      W_Statistic = as.numeric(test$statistic),
      P_Value = test$p.value,
      Significant = ifelse(
        test$p.value < 0.05,
        "Yes",
        "No"
      )
    )
  )
}

cat("\n============================================\n")
cat("WILCOXON TEST RESULTS\n")
cat("============================================\n")

print(wilcoxon_results)

write.csv(
  wilcoxon_results,
  "results/wilcoxon_results.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 3. Chi-square tests for categorical variables
# ------------------------------------------------------------

categorical_features <- c(
  "Plant",
  "Department",
  "JobRole",
  "Shift",
  "Education",
  "EducationField",
  "EngagementLevel",
  "OvertimeLevel",
  "DistanceCategory",
  "TenureGroup"
)

chi_square_results <- data.frame()

for (feature in categorical_features) {

  contingency_table <- table(
    hr_data[[feature]],
    hr_data$AttritionFlag
  )

  test <- suppressWarnings(
    chisq.test(contingency_table)
  )

  chi_square_results <- rbind(
    chi_square_results,
    data.frame(
      Feature = feature,
      Chi_Square = as.numeric(test$statistic),
      Degrees_of_Freedom = as.numeric(test$parameter),
      P_Value = test$p.value,
      Significant = ifelse(
        test$p.value < 0.05,
        "Yes",
        "No"
      )
    )
  )
}

cat("\n============================================\n")
cat("CHI-SQUARE TEST RESULTS\n")
cat("============================================\n")

print(chi_square_results)

write.csv(
  chi_square_results,
  "results/chi_square_results.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 4. Correlation with attrition probability
# ------------------------------------------------------------

hr_data_numeric <- hr_data %>%
  mutate(
    AttritionBinary = ifelse(
      AttritionFlag == "Yes",
      1,
      0
    )
  )

correlation_results <- data.frame()

for (feature in numeric_features) {

  correlation_value <- cor(
    hr_data_numeric[[feature]],
    hr_data_numeric$AttritionBinary,
    use = "complete.obs"
  )

  correlation_results <- rbind(
    correlation_results,
    data.frame(
      Feature = feature,
      Correlation = correlation_value
    )
  )
}

correlation_results <- correlation_results %>%
  arrange(desc(abs(Correlation)))

cat("\n============================================\n")
cat("CORRELATION WITH ATTRITION\n")
cat("============================================\n")

print(correlation_results)

write.csv(
  correlation_results,
  "results/attrition_correlations.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 5. Summary of significant variables
# ------------------------------------------------------------

cat("\n============================================\n")
cat("SIGNIFICANT WILCOXON FEATURES\n")
cat("============================================\n")

print(
  wilcoxon_results %>%
    filter(Significant == "Yes")
)

cat("\n============================================\n")
cat("SIGNIFICANT CHI-SQUARE FEATURES\n")
cat("============================================\n")

print(
  chi_square_results %>%
    filter(Significant == "Yes")
)

cat("\n============================================\n")
cat("STATISTICAL ANALYSIS COMPLETE\n")
cat("============================================\n")

cat("\nSaved files:\n")
cat("- results/wilcoxon_results.csv\n")
cat("- results/chi_square_results.csv\n")
cat("- results/attrition_correlations.csv\n")