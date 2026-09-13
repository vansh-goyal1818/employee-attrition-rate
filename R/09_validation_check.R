# ============================================================
# DA2 - Stage 9: Validation and Data Leakage Check
# Employee Attrition Prediction
# ============================================================

cat("============================================\n")
cat("VALIDATION AND DATA LEAKAGE CHECK\n")
cat("============================================\n")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

train_data <- readRDS(
  "results/train_data.rds"
)

test_data <- readRDS(
  "results/test_data.rds"
)

engineered_data <- readRDS(
  "results/engineered_hr_data.rds"
)

selected_features <- read.csv(
  "results/selected_features.csv",
  stringsAsFactors = FALSE
)$SelectedFeature

cat("\nSelected features:\n")
print(selected_features)

# ------------------------------------------------------------
# 2. TARGET LEAKAGE CHECK
# ------------------------------------------------------------

cat("\n============================================\n")
cat("1. TARGET LEAKAGE CHECK\n")
cat("============================================\n")

if ("AttritionFlag" %in% selected_features) {

  cat(
    "WARNING: AttritionFlag is included as a predictor!\n"
  )

} else {

  cat(
    "PASS: AttritionFlag is NOT included as a predictor.\n"
  )
}

# Also check for obvious target-derived variables

target_derived <- c(
  "Attrition",
  "AttritionFlag"
)

leakage_variables <- intersect(
  selected_features,
  target_derived
)

if (length(leakage_variables) == 0) {

  cat(
    "PASS: No direct target-derived variables detected.\n"
  )

} else {

  cat(
    "WARNING: Target-derived variables detected:\n"
  )

  print(leakage_variables)
}

# ------------------------------------------------------------
# 3. FEATURE-TARGET RELATIONSHIP CHECK
# ------------------------------------------------------------

cat("\n============================================\n")
cat("2. FEATURE-TARGET CHECK\n")
cat("============================================\n")

target_numeric <- ifelse(
  train_data$AttritionFlag == "Yes",
  1,
  0
)

for (feature in selected_features) {

  if (feature %in% names(train_data) &&
      is.numeric(train_data[[feature]])) {

    correlation <- cor(
      train_data[[feature]],
      target_numeric,
      use = "complete.obs"
    )

    cat(
      feature,
      ": correlation =",
      round(correlation, 4),
      "\n"
    )
  }
}

# ------------------------------------------------------------
# 4. TRAIN / TEST OVERLAP CHECK
# ------------------------------------------------------------

cat("\n============================================\n")
cat("3. TRAIN / TEST OVERLAP CHECK\n")
cat("============================================\n")

# EmployeeNumber was not selected as a model feature,
# so compare complete selected-feature combinations.

train_keys <- do.call(
  paste,
  c(
    train_data[, selected_features, drop = FALSE],
    sep = "_"
  )
)

test_keys <- do.call(
  paste,
  c(
    test_data[, selected_features, drop = FALSE],
    sep = "_"
  )
)

overlap <- length(
  intersect(train_keys, test_keys)
)

cat(
  "Exact overlapping selected-feature combinations:",
  overlap,
  "\n"
)

if (overlap == 0) {

  cat(
    "PASS: No exact selected-feature combinations overlap.\n"
  )

} else {

  cat(
    "NOTE: Some identical feature combinations exist.\n"
  )
}

# ------------------------------------------------------------
# 5. CLASS DISTRIBUTION
# ------------------------------------------------------------

cat("\n============================================\n")
cat("4. CLASS DISTRIBUTION\n")
cat("============================================\n")

cat("\nTraining set:\n")
print(
  table(train_data$AttritionFlag)
)

print(
  prop.table(
    table(train_data$AttritionFlag)
  )
)

cat("\nTesting set:\n")
print(
  table(test_data$AttritionFlag)
)

print(
  prop.table(
    table(test_data$AttritionFlag)
  )
)

# ------------------------------------------------------------
# 6. ENGINEERED FEATURE VALIDATION
# ------------------------------------------------------------

cat("\n============================================\n")
cat("5. ENGINEERED FEATURE VALIDATION\n")
cat("============================================\n")

# Satisfaction Index

calculated_satisfaction <- rowMeans(
  engineered_data[
    ,
    c(
      "EnvironmentSatisfaction",
      "JobSatisfaction",
      "JobInvolvement"
    )
  ]
)

satisfaction_match <- isTRUE(
  all.equal(
    calculated_satisfaction,
    engineered_data$SatisfactionIndex
  )
)

cat(
  "SatisfactionIndex matches calculation:",
  satisfaction_match,
  "\n"
)

# Engagement Level

calculated_engagement <- ifelse(
  engineered_data$EngagementScore < 50,
  "Low",
  ifelse(
    engineered_data$EngagementScore < 70,
    "Medium",
    "High"
  )
)

engagement_match <- isTRUE(
  all.equal(
    calculated_engagement,
    engineered_data$EngagementLevel
  )
)

cat(
  "EngagementLevel matches calculation:",
  engagement_match,
  "\n"
)

# Overtime Level

calculated_overtime <- ifelse(
  engineered_data$MonthlyOvertimeHours <= 20,
  "Low",
  ifelse(
    engineered_data$MonthlyOvertimeHours <= 40,
    "Medium",
    "High"
  )
)

overtime_match <- isTRUE(
  all.equal(
    calculated_overtime,
    engineered_data$OvertimeLevel
  )
)

cat(
  "OvertimeLevel matches calculation:",
  overtime_match,
  "\n"
)

# Distance Category

calculated_distance <- ifelse(
  engineered_data$DistanceFromHome <= 10,
  "Near",
  ifelse(
    engineered_data$DistanceFromHome <= 30,
    "Moderate",
    "Far"
  )
)

distance_match <- isTRUE(
  all.equal(
    calculated_distance,
    engineered_data$DistanceCategory
  )
)

cat(
  "DistanceCategory matches calculation:",
  distance_match,
  "\n"
)

# Tenure Group

calculated_tenure <- ifelse(
  engineered_data$YearsAtCompany <= 2,
  "New",
  ifelse(
    engineered_data$YearsAtCompany <= 5,
    "Early",
    ifelse(
      engineered_data$YearsAtCompany <= 10,
      "Established",
      "Long_Term"
    )
  )
)

tenure_match <- isTRUE(
  all.equal(
    calculated_tenure,
    engineered_data$TenureGroup
  )
)

cat(
  "TenureGroup matches calculation:",
  tenure_match,
  "\n"
)

# ------------------------------------------------------------
# 7. DATA DIMENSION CHECK
# ------------------------------------------------------------

cat("\n============================================\n")
cat("6. DATA DIMENSION CHECK\n")
cat("============================================\n")

cat(
  "Training rows:",
  nrow(train_data),
  "\n"
)

cat(
  "Testing rows:",
  nrow(test_data),
  "\n"
)

cat(
  "Selected predictor count:",
  length(selected_features),
  "\n"
)

if (
  nrow(train_data) > 0 &&
  nrow(test_data) > 0 &&
  length(selected_features) > 0
) {

  cat(
    "PASS: Training, testing and selected-feature data are available.\n"
  )

} else {

  cat(
    "WARNING: One or more required datasets are empty.\n"
  )
}

# ------------------------------------------------------------
# 8. FINAL VALIDATION STATUS
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FINAL VALIDATION STATUS\n")
cat("============================================\n")

all_checks <- all(
  !("AttritionFlag" %in% selected_features),
  length(leakage_variables) == 0,
  satisfaction_match,
  engagement_match,
  overtime_match,
  distance_match,
  tenure_match
)

if (all_checks) {

  cat(
    "PASS: Core validation and leakage checks completed successfully.\n"
  )

} else {

  cat(
    "REVIEW REQUIRED: One or more validation checks need attention.\n"
  )
}

cat("\n============================================\n")
cat("VALIDATION CHECK COMPLETED\n")
cat("============================================\n")