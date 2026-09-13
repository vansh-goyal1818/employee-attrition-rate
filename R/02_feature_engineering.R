# ============================================================
# DA2 - Stage 1: Feature Engineering
# Employee Attrition Prediction
# ============================================================

library(readxl)
library(dplyr)

cat("============================================\n")
cat("DA2 - FEATURE ENGINEERING\n")
cat("============================================\n")

# ------------------------------------------------------------
# 1. Load new realistic dataset
# ------------------------------------------------------------

data_path <- "data/Uno_Minda_Multi_Plant_HR_Data_50000_Realistic.xlsx"

hr_data <- read_excel(data_path)

cat("\nDataset loaded successfully.\n")
cat("Rows:", nrow(hr_data), "\n")
cat("Columns:", ncol(hr_data), "\n")

# ------------------------------------------------------------
# 2. Basic target preparation
# ------------------------------------------------------------

hr_data <- hr_data %>%
  mutate(
    AttritionFlag = factor(
      AttritionFlag,
      levels = c("No", "Yes")
    )
  )

# ------------------------------------------------------------
# 3. Feature Engineering
# ------------------------------------------------------------

# Satisfaction Index:
# Average of environment satisfaction,
# job satisfaction and job involvement.

hr_data <- hr_data %>%
  mutate(
    SatisfactionIndex = round(
      (
        EnvironmentSatisfaction +
        JobSatisfaction +
        JobInvolvement
      ) / 3,
      2
    )
  )

# Engagement level

hr_data <- hr_data %>%
  mutate(
    EngagementLevel = case_when(
      EngagementScore < 50 ~ "Low",
      EngagementScore < 70 ~ "Medium",
      TRUE ~ "High"
    )
  )

# Overtime level

hr_data <- hr_data %>%
  mutate(
    OvertimeLevel = case_when(
      MonthlyOvertimeHours <= 20 ~ "Low",
      MonthlyOvertimeHours <= 40 ~ "Medium",
      TRUE ~ "High"
    )
  )

# Distance category

hr_data <- hr_data %>%
  mutate(
    DistanceCategory = case_when(
      DistanceFromHome <= 10 ~ "Near",
      DistanceFromHome <= 30 ~ "Moderate",
      TRUE ~ "Far"
    )
  )

# Tenure group

hr_data <- hr_data %>%
  mutate(
    TenureGroup = case_when(
      YearsAtCompany <= 2 ~ "New",
      YearsAtCompany <= 5 ~ "Early",
      YearsAtCompany <= 10 ~ "Established",
      TRUE ~ "Long_Term"
    )
  )

# ------------------------------------------------------------
# 4. Convert engineered categorical variables to factors
# ------------------------------------------------------------

hr_data <- hr_data %>%
  mutate(
    EngagementLevel = factor(
      EngagementLevel,
      levels = c("Low", "Medium", "High")
    ),
    OvertimeLevel = factor(
      OvertimeLevel,
      levels = c("Low", "Medium", "High")
    ),
    DistanceCategory = factor(
      DistanceCategory,
      levels = c("Near", "Moderate", "Far")
    ),
    TenureGroup = factor(
      TenureGroup,
      levels = c("New", "Early", "Established", "Long_Term")
    )
  )

# ------------------------------------------------------------
# 5. Display engineered features
# ------------------------------------------------------------

cat("\n============================================\n")
cat("ENGINEERED FEATURES\n")
cat("============================================\n")

cat("\nSatisfactionIndex summary:\n")
print(summary(hr_data$SatisfactionIndex))

cat("\nEngagementLevel distribution:\n")
print(table(hr_data$EngagementLevel))

cat("\nOvertimeLevel distribution:\n")
print(table(hr_data$OvertimeLevel))

cat("\nDistanceCategory distribution:\n")
print(table(hr_data$DistanceCategory))

cat("\nTenureGroup distribution:\n")
print(table(hr_data$TenureGroup))

# ------------------------------------------------------------
# 6. Check target distribution
# ------------------------------------------------------------

cat("\n============================================\n")
cat("ATTRITION DISTRIBUTION\n")
cat("============================================\n")

print(table(hr_data$AttritionFlag))

print(
  round(
    prop.table(table(hr_data$AttritionFlag)) * 100,
    2
  )
)

# ------------------------------------------------------------
# 7. Save engineered dataset
# ------------------------------------------------------------

saveRDS(
  hr_data,
  "results/engineered_hr_data.rds"
)

cat("\n============================================\n")
cat("FEATURE ENGINEERING COMPLETE\n")
cat("============================================\n")

cat("\nSaved:\n")
cat("results/engineered_hr_data.rds\n")