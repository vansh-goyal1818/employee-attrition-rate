# ============================================================
# DA2 - Stage 3: Database Connectivity & SQL Analysis
# ============================================================

library(DBI)
library(RSQLite)
library(dplyr)

cat("============================================\n")
cat("DA2 - DATABASE CONNECTIVITY\n")
cat("============================================\n")

# ------------------------------------------------------------
# 1. Load engineered dataset
# ------------------------------------------------------------

hr_data <- readRDS("results/engineered_hr_data.rds")

cat("\nDataset loaded:\n")
cat("Rows:", nrow(hr_data), "\n")
cat("Columns:", ncol(hr_data), "\n")

# ------------------------------------------------------------
# 2. Create SQLite database
# ------------------------------------------------------------

db_path <- "results/hr_attrition.db"

if (file.exists(db_path)) {
  file.remove(db_path)
}

con <- dbConnect(
  SQLite(),
  db_path
)

# ------------------------------------------------------------
# 3. Write dataset to database
# ------------------------------------------------------------

dbWriteTable(
  con,
  "employee_data",
  hr_data,
  overwrite = TRUE
)

cat("\nDatabase table created: employee_data\n")

# ------------------------------------------------------------
# 4. Verify table
# ------------------------------------------------------------

tables <- dbListTables(con)

cat("\nDatabase tables:\n")
print(tables)

cat("\nNumber of rows in database:\n")

row_count <- dbGetQuery(
  con,
  "SELECT COUNT(*) AS total_rows FROM employee_data"
)

print(row_count)

# ------------------------------------------------------------
# 5. Attrition distribution
# ------------------------------------------------------------

attrition_distribution <- dbGetQuery(
  con,
  "
  SELECT
      AttritionFlag,
      COUNT(*) AS EmployeeCount,
      ROUND(
        COUNT(*) * 100.0 /
        (SELECT COUNT(*) FROM employee_data),
        2
      ) AS Percentage
  FROM employee_data
  GROUP BY AttritionFlag
  ORDER BY AttritionFlag
  "
)

cat("\n============================================\n")
cat("SQL - ATTRITION DISTRIBUTION\n")
cat("============================================\n")

print(attrition_distribution)

write.csv(
  attrition_distribution,
  "results/sql_attrition_distribution.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 6. Department-level analysis
# ------------------------------------------------------------

department_analysis <- dbGetQuery(
  con,
  "
  SELECT
      Department,
      COUNT(*) AS EmployeeCount,
      SUM(
        CASE WHEN AttritionFlag = 'Yes'
        THEN 1 ELSE 0 END
      ) AS AttritionCount,
      ROUND(
        AVG(
          CASE WHEN AttritionFlag = 'Yes'
          THEN 1.0 ELSE 0.0 END
        ) * 100,
        2
      ) AS AttritionRate
  FROM employee_data
  GROUP BY Department
  ORDER BY AttritionRate DESC
  "
)

cat("\n============================================\n")
cat("SQL - DEPARTMENT ANALYSIS\n")
cat("============================================\n")

print(department_analysis)

write.csv(
  department_analysis,
  "results/sql_department_analysis.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. Attrition vs key numerical variables
# ------------------------------------------------------------

key_variable_analysis <- dbGetQuery(
  con,
  "
  SELECT
      AttritionFlag,
      ROUND(AVG(EngagementScore), 2) AS AvgEngagement,
      ROUND(AVG(MonthlyOvertimeHours), 2) AS AvgOvertimeHours,
      ROUND(AVG(EnvironmentSatisfaction), 2) AS AvgEnvironmentSatisfaction,
      ROUND(AVG(JobSatisfaction), 2) AS AvgJobSatisfaction,
      ROUND(AVG(DistanceFromHome), 2) AS AvgDistance,
      ROUND(AVG(MonthlyIncome), 2) AS AvgMonthlyIncome,
      ROUND(AVG(YearsAtCompany), 2) AS AvgYearsAtCompany
  FROM employee_data
  GROUP BY AttritionFlag
  "
)

cat("\n============================================\n")
cat("SQL - ATTRITION VS KEY VARIABLES\n")
cat("============================================\n")

print(key_variable_analysis)

write.csv(
  key_variable_analysis,
  "results/sql_key_variable_analysis.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 8. Job role analysis
# ------------------------------------------------------------

jobrole_analysis <- dbGetQuery(
  con,
  "
  SELECT
      JobRole,
      COUNT(*) AS EmployeeCount,
      SUM(
        CASE WHEN AttritionFlag = 'Yes'
        THEN 1 ELSE 0 END
      ) AS AttritionCount,
      ROUND(
        AVG(
          CASE WHEN AttritionFlag = 'Yes'
          THEN 1.0 ELSE 0.0 END
        ) * 100,
        2
      ) AS AttritionRate
  FROM employee_data
  GROUP BY JobRole
  ORDER BY AttritionRate DESC
  "
)

cat("\n============================================\n")
cat("SQL - JOB ROLE ANALYSIS\n")
cat("============================================\n")

print(jobrole_analysis)

write.csv(
  jobrole_analysis,
  "results/sql_jobrole_analysis.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 9. Retrieve ML dataset using SQL
# ------------------------------------------------------------

selected_features <- read.csv(
  "results/selected_features.csv",
  stringsAsFactors = FALSE
)$SelectedFeature

ml_columns <- paste(
  c(selected_features, "AttritionFlag"),
  collapse = ", "
)

ml_query <- paste(
  "SELECT",
  ml_columns,
  "FROM employee_data"
)

ml_data_sql <- dbGetQuery(
  con,
  ml_query
)

cat("\n============================================\n")
cat("SQL - ML DATA RETRIEVAL\n")
cat("============================================\n")

cat(
  "Retrieved rows:",
  nrow(ml_data_sql),
  "\n"
)

cat(
  "Retrieved columns:",
  ncol(ml_data_sql),
  "\n"
)

# ------------------------------------------------------------
# 10. Close database
# ------------------------------------------------------------

dbDisconnect(con)

cat("\n============================================\n")
cat("DATABASE CONNECTIVITY COMPLETE\n")
cat("============================================\n")

cat("\nSaved files:\n")
cat("- results/hr_attrition.db\n")
cat("- results/sql_attrition_distribution.csv\n")
cat("- results/sql_department_analysis.csv\n")
cat("- results/sql_key_variable_analysis.csv\n")
cat("- results/sql_jobrole_analysis.csv\n")