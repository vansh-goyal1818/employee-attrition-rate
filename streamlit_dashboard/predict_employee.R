# ==============================================================================
# R PREDICTION BRIDGE FOR STREAMLIT DECISION SUPPORT DASHBOARD
# Project: Employee Attrition Prediction & Workforce Analytics
# Course:  Programming for Data Science | SCOPE, VIT Chennai
# ==============================================================================
# Evaluates a single employee profile against the trained Tuned XGBoost model.
# Ingests JSON input (file path or raw JSON string) and outputs structured JSON.
# ==============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(caret)
  library(xgboost)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cat('{"error": "No input provided. Pass JSON file path or raw JSON string as first argument."}\n')
  quit(status = 1)
}

# 1. Parse Input JSON
input_raw <- args[1]
if (file.exists(input_raw)) {
  input_data <- fromJSON(input_raw)
} else {
  input_data <- fromJSON(input_raw)
}

# 2. Selected Predictive Features
selected_features <- c(
  "EngagementScore", "MonthlyIncome", "MonthlyRate", "DailyRate", 
  "Education", "HourlyRate", "YearsAtCompany", "DistanceFromHome", 
  "MonthlyOvertimeHours", "Department", "Plant", "EducationField", 
  "SatisfactionIndex", "JobRole", "JobInvolvement"
)

# 3. Load Trained Artifacts
train_data       <- readRDS("results/train_data.rds")
dummy_model      <- readRDS("results/dummy_encoding_model.rds")
preprocess_model <- readRDS("results/preprocessing_model.rds")
final_model      <- readRDS("results/tuned_xgboost_model.rds")

# Extract known factor levels
cat_cols <- c("Department", "Plant", "EducationField", "JobRole")
factor_levels <- list()
for (col in cat_cols) {
  factor_levels[[col]] <- sort(unique(as.character(train_data[[col]])))
}

# 4. Compute Derived Features & Build Input Row
env_sat <- as.numeric(input_data$EnvironmentSatisfaction)
job_sat <- as.numeric(input_data$JobSatisfaction)
job_inv <- as.numeric(input_data$JobInvolvement)
sat_idx <- (env_sat + job_sat + job_inv) / 3

row_df <- data.frame(
  EngagementScore      = as.numeric(input_data$EngagementScore),
  MonthlyIncome        = as.numeric(input_data$MonthlyIncome),
  MonthlyRate          = as.numeric(input_data$MonthlyRate),
  DailyRate            = as.numeric(input_data$DailyRate),
  Education            = as.numeric(input_data$Education),
  HourlyRate           = as.numeric(input_data$HourlyRate),
  YearsAtCompany       = as.numeric(input_data$YearsAtCompany),
  DistanceFromHome     = as.numeric(input_data$DistanceFromHome),
  MonthlyOvertimeHours = as.numeric(input_data$MonthlyOvertimeHours),
  Department           = factor(as.character(input_data$Department), levels = factor_levels[["Department"]]),
  Plant                = factor(as.character(input_data$Plant), levels = factor_levels[["Plant"]]),
  EducationField       = factor(as.character(input_data$EducationField), levels = factor_levels[["EducationField"]]),
  SatisfactionIndex    = as.numeric(sat_idx),
  JobRole              = factor(as.character(input_data$JobRole), levels = factor_levels[["JobRole"]]),
  JobInvolvement       = as.numeric(job_inv),
  stringsAsFactors     = FALSE
)

row_df <- row_df[, selected_features, drop = FALSE]

# 5. Apply Transformation Pipeline & Score
d_enc  <- predict(dummy_model, newdata = row_df)
p_proc <- predict(preprocess_model, newdata = d_enc)
dmat   <- xgb.DMatrix(data = as.matrix(p_proc))
prob   <- as.numeric(predict(final_model, newdata = dmat))

# 6. Format Response JSON
out_list <- list(
  probability = prob,
  satisfaction_index = sat_idx,
  status = "success"
)

# Output only valid JSON to stdout
cat(toJSON(out_list, auto_unbox = TRUE))
cat("\n")
