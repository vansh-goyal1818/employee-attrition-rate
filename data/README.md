# Dataset Information

## Overview
This project investigates employee attrition and workforce dynamics using an industrial manufacturing dataset:
- **Source File**: `Uno_Minda_Multi_Plant_HR_Data_50000_Realistic.xlsx`
- **Volume**: 50,000 employee records
- **Dimensionality**: 21 raw columns spanning demographic, compensation, operational, and psychometric dimensions
- **Target Variable**: `AttritionFlag` (`No`: 41,668 records [83.34%], `Yes`: 8,332 records [16.66%])
- **Plants & Operational Context**: Manufacturing plants located in major industrial clusters (Bawal, Manesar, Pune)

---

## Exclusion from GitHub Repository
In accordance with institutional data management practices and repository hygiene:
- **The raw Excel dataset is intentionally excluded from this GitHub repository.**
- Exclusion is strictly enforced via the root `.gitignore` file (`data/*.xlsx`).
- The Excel file is maintained locally for script execution and model training.

---

## Reproducibility & Repository Artifacts
While the raw Excel dataset is excluded from version control:
1. **Analysis & Pipeline Scripts**: All 12 R scripts in [`R/`](../R/) contain the complete, transparent data processing, feature engineering, statistical testing, and machine learning logic.
2. **Processed Data & Model RDS Files**: Saved serialized objects (e.g., [`results/engineered_hr_data.rds`](../results/), [`results/train_data.rds`](../results/), [`results/test_data.rds`](../results/), and [`results/tuned_xgboost_model.rds`](../results/)) are included in the repository.
3. **Database & Results**: The SQLite database ([`results/hr_attrition.db`](../results/)) and all empirical CSV metrics and PNG plots are committed to provide complete verification of results.
