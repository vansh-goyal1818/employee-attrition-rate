# ============================================================
# Employee Attrition Prediction Project
# Step 01: Data Import and Validation
# ============================================================

library(readxl)
library(dplyr)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

data_path <- "data/Uno_Minda_Multi_Plant_HR_Data_50000_Realistic.xlsx"
hr_data <- read_excel(data_path)

# ------------------------------------------------------------
# 2. Basic dataset information
# ------------------------------------------------------------

cat("Dataset loaded successfully.\n")
cat("Number of rows:", nrow(hr_data), "\n")
cat("Number of columns:", ncol(hr_data), "\n\n")

# ------------------------------------------------------------
# 3. Display column names
# ------------------------------------------------------------

cat("Column names:\n")
print(names(hr_data))

# ------------------------------------------------------------
# 4. Check missing values
# ------------------------------------------------------------

cat("\nMissing values by column:\n")
print(colSums(is.na(hr_data)))

# ------------------------------------------------------------
# 5. Check duplicate rows
# ------------------------------------------------------------

cat("\nDuplicate rows:", sum(duplicated(hr_data)), "\n")

# ------------------------------------------------------------
# 6. Check target variable
# ------------------------------------------------------------

cat("\nAttrition distribution:\n")
print(table(hr_data$AttritionFlag))

cat("\nAttrition percentages:\n")
print(round(prop.table(table(hr_data$AttritionFlag)) * 100, 2))

# ------------------------------------------------------------
# 7. Dataset structure
# ------------------------------------------------------------

cat("\nDataset structure:\n")
str(hr_data)

# ------------------------------------------------------------
# 8. First few records
# ------------------------------------------------------------

cat("\nFirst 6 records:\n")
print(head(hr_data))