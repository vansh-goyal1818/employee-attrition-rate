# Employee Attrition Prediction and Workforce Analytics Using Machine Learning

[![R](https://img.shields.io/badge/Language-R%20%3E%3D%204.0-blue.svg)](https://www.r-project.org/)
[![XGBoost](https://img.shields.io/badge/Model-Tuned%20XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)](https://www.sqlite.org/)
[![Status](https://img.shields.io/badge/Project%20Status-Complete%20(DA1%20%2B%20DA2)-brightgreen.svg)]()

> **Course**: Programming for Data Science  
> **Institution**: School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Chennai  
> **Project Scope**: End-to-End Predictive Modeling, Relational SQL Analytics, Hyperparameter Optimization, and Operational Decision Thresholding for Multi-Plant Manufacturing

---

## 👥 Team Members

| Name | Registration Number | Primary Focus Areas |
| :--- | :---: | :--- |
| **Vansh Goyal** | `24BDS1156` | Lead Machine Learning Modeling, Hyperparameter Tuning & Threshold Optimization |
| **Ritabrata Dey** | `24BDS1129` | Feature Engineering, SQLite Database Architecture & SQL Analytics |
| **Chetan Pareek** | `24BDS1125` | Non-Parametric Hypothesis Testing, Model Validation & Documentation |

---

## 📌 Project Overview & Problem Statement

Voluntary employee turnover imposes severe operational friction on continuous multi-plant manufacturing environments. In precision automotive component production (such as the Uno Minda industrial facilities), unplanned shop-floor departures destabilize assembly lines, increase quality defect rates, necessitate expensive overtime from remaining workers, and trigger costly replacement cycles costing 50% to 150% of annual employee compensation.

Traditional human resource management operates reactively—documenting departure rationales through exit interviews only after valuable human capital has resigned. This project formulates, evaluates, and deploys an end-to-end predictive machine learning and workforce analytics pipeline that calculates individualized employee departure risk probabilities months in advance, enabling targeted, data-driven retention interventions.

---

## 🎯 Project Objectives

1. **Multi-Dimensional Feature Engineering**: Construct behavioral composite indicators capturing workplace sentiment, physical fatigue, and commuting burden.
2. **Leakage-Free Feature Selection**: Rank predictors using Random Forest Gini impurity reduction with variable-level aggregation, selecting the top 15 features while enforcing strict target-leakage prevention.
3. **Relational Database Connectivity**: Implement an embedded SQLite database (`results/hr_attrition.db`) with production-ready SQL querying across organizational hierarchies.
4. **Statistical Hypothesis Testing**: Quantify the statistical significance of turnover drivers via Wilcoxon Rank-Sum tests, Pearson's Chi-Square tests of independence, and point-biserial correlations.
5. **Multi-Model Benchmarking**: Train and compare 10 diverse machine learning algorithms under unified preprocessing.
6. **Hyperparameter Tuning**: Perform 3-fold cross-validation on stratified subsets for Random Forest, Gradient Boosting, and XGBoost.
7. **Classification Threshold Optimization**: Address severe class imbalance (83.34% retention vs. 16.66% attrition) by optimizing the decision threshold from default 0.50 to 0.20 to maximize F1-score and operational recall.
8. **Explainable Workforce Analytics**: Attribute turnover drivers via XGBoost tree split gain, formulating actionable industrial human resource policies.

---

## 📊 Dataset Description

The empirical foundation of this study is an authentic industrial workforce dataset:
- **Source**: `Uno_Minda_Multi_Plant_HR_Data_50000_Realistic.xlsx` (multi-plant manufacturing facilities in Bawal, Manesar, and Pune).
- **Scale**: Exactly **50,000 individual employee records** across **21 raw attributes**.
- **Target Variable**: `AttritionFlag` (`No`: 41,668 records [**83.34%**], `Yes`: 8,332 records [**16.66%**]).
- **Quality Audits**: 0 missing values across all 1,050,000 cells; 0 duplicate rows or duplicate employee identifiers.
- **Predictor Dimensions**:
  - *Demographic & Education*: `Education` (ordinal 1–5), `EducationField` (6 levels).
  - *Compensation*: `MonthlyIncome`, `MonthlyRate`, `DailyRate`, `HourlyRate`.
  - *Operational & Logistics*: `Department` (7 levels), `JobRole` (5 levels), `Plant` (6 sites), `Shift` (3 levels), `DistanceFromHome` (km), `MonthlyOvertimeHours`, `YearsAtCompany`, `SafetyIncidents_12M`.
  - *Psychometric & Sentiment*: `EnvironmentSatisfaction`, `JobSatisfaction`, `JobInvolvement`, `EngagementScore`.

> **Note on Dataset Availability**: The raw Excel dataset (`data/*.xlsx`) is intentionally excluded from this GitHub repository via `.gitignore`. The complete analysis scripts, intermediate data representations, trained model RDS files, SQLite database, and empirical outputs are fully preserved within the repository.

---

## 🔄 DA1 vs. DA2 Continuity

- **Digital Assignment 1 (DA1)**: Completed foundational exploratory data analysis (EDA), univariate and bivariate statistical visualization, preliminary data cleaning, and academic literature review.
- **Digital Assignment 2 (DA2)**: Ingested the validated dataset to execute advanced computational modeling, relational database connectivity, non-parametric statistical testing, 10-algorithm benchmarking, cross-validated hyperparameter tuning, threshold calibration, and business impact interpretation.

---

## 🛠️ DA2 Technical Methodology

```
┌────────────────────────────────────────────────────────────────────────────┐
│                             DA2 WORKFLOW PIPELINE                          │
└────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
[01. Data Import & Validation] ──────────► 50,000 records, 0 missing, 0 duplicates
        │
        ▼
[02. Feature Engineering] ──────────────► SatisfactionIndex, EngagementLevel,
        │                                 OvertimeLevel, DistanceCategory, TenureGroup
        ▼
[03. Feature Selection] ────────────────► RF Gini Importance -> Top 15 Predictors
        │                                 (Zero target leakage verified)
        ▼
[04. SQLite Database & SQL] ────────────► results/hr_attrition.db -> SQL Analytics
        │
        ▼
[05. Statistical Hypothesis Testing] ───► Wilcoxon tests, Chi-Square tests, Correlations
        │
        ▼
[06. 10 ML Baseline Models] ────────────► Logistic, CART, RF, KNN, NB, LDA,
        │                                 Elastic Net, SVM, GBM, XGBoost
        ▼
[07. Baseline Model Evaluation] ────────► Default threshold 0.50 failure mode analysis
        │
        ▼
[08. Hyperparameter Tuning] ────────────► 3-Fold CV on 10k stratified sample
        │                                 (Random Forest, GBM, XGBoost)
        ▼
[09. Validation & Leakage Checks] ──────► Dimensional integrity and mathematical audit
        │
        ▼
[10. Tuned Model Evaluation] ───────────► Tuned RF vs. Tuned GBM vs. Tuned XGBoost
        │
        ▼
[11. Threshold Optimization] ───────────► Sweep [0.10, 0.90] -> Optimal Threshold = 0.20
        │
        ▼
[12. Final Model Analysis] ─────────────► Tuned XGBoost @ 0.20 -> Scorecard & CM
```

---

## ⚙️ Feature Engineering

Five domain-guided behavioral indicators were constructed in `R/02_feature_engineering.R`:

1. **`SatisfactionIndex`** (Continuous, 1.00 – 4.00):
   $$\text{SatisfactionIndex} = \frac{\text{EnvironmentSatisfaction} + \text{JobSatisfaction} + \text{JobInvolvement}}{3}$$
   Combines multi-attribute psychometric ratings to reduce survey measurement noise.
2. **`EngagementLevel`** (Ordinal Factor: `Low`, `Medium`, `High`):
   Discretized from raw `EngagementScore`: `Low` (<50), `Medium` (50–69), `High` (≥70).
3. **`OvertimeLevel`** (Ordinal Factor: `Low`, `Medium`, `High`):
   Discretized from `MonthlyOvertimeHours`: `Low` (≤20h), `Medium` (21–40h), `High` (>40h fatigue/burnout zone).
4. **`DistanceCategory`** (Ordinal Factor: `Near`, `Moderate`, `Far`):
   Commute tiers from `DistanceFromHome`: `Near` (≤10 km), `Moderate` (11–30 km), `Far` (>30 km transit burden).
5. **`TenureGroup`** (Ordinal Factor: `New`, `Early`, `Established`, `Long_Term`):
   Organizational lifecycle stages: `New` (≤2 yrs), `Early` (2.1–5 yrs), `Established` (5.1–10 yrs), `Long_Term` (>10 yrs).

---

## 🔍 Feature Selection & Leakage Prevention

In `R/03_feature_selection.R`:
- **Protocol**: Removed `EmployeeNumber` (arbitrary identifier) and `EducationLevel` (duplicate text column).
- **Partitioning**: 80% train (40,001 rows) and 20% test (9,999 rows) via `caret::createDataPartition(p = 0.80, seed = 123)`.
- **Ranking**: Trained a Random Forest (100 trees) on one-hot encoded training predictors. Gini importance scores across dummy levels were aggregated back to original parent variables.
- **Leakage Check (PASSED)**: Verified that `AttritionFlag` and derived proxies were strictly excluded. Encoding and scaling parameters were fitted strictly on the training partition.

### Top 15 Selected Features

| Rank | Feature Name | Total Gini Importance | Domain Category |
| :---: | :--- | :---: | :--- |
| **1** | `EngagementScore` | **882.96** | Psychometric / Behavioral |
| **2** | `MonthlyIncome` | **880.37** | Financial / Base Compensation |
| **3** | `MonthlyRate` | **872.39** | Financial / Variable Compensation |
| **4** | `DailyRate` | **850.42** | Financial / Production Compensation |
| **5** | `Education` | **839.04** | Demographic / Qualification |
| **6** | `HourlyRate` | **787.18** | Financial / Base Rate |
| **7** | `YearsAtCompany` | **758.51** | Operational / Tenure |
| **8** | `DistanceFromHome` | **724.05** | Operational / Commute Burden |
| **9** | `MonthlyOvertimeHours` | **706.18** | Operational / Workload Fatigue |
| **10** | `Department` | **537.30** | Organizational / Functional Unit |
| **11** | `Plant` | **509.56** | Organizational / Manufacturing Site |
| **12** | `EducationField` | **506.42** | Demographic / Academic Background |
| **13** | `SatisfactionIndex` | **370.11** | Psychometric / Composite Sentiment |
| **14** | `JobRole` | **363.26** | Organizational / Position Tier |
| **15** | `JobInvolvement` | **256.09** | Psychometric / Role Investment |

---

## 🗄️ Relational Database & SQL Analytics

In `R/04_database_connectivity.R`, an embedded SQLite database (`results/hr_attrition.db`) was constructed. Key SQL aggregation queries revealed significant workforce disparities:

### Department-Level Turnover (`results/sql_department_analysis.csv`)
- **Logistics**: 17.71% attrition (1,330 departed / 7,511 employees) — **Highest turnover department**
- **Assembly A**: 17.16% attrition (1,878 departed / 10,947 employees)
- **Maintenance**: 16.82% attrition (988 departed / 5,874 employees)
- **Assembly B**: 16.32% attrition (1,479 departed / 9,065 employees)
- **Quality Control**: 16.12% attrition (824 departed / 5,111 employees)
- **Packaging**: 16.07% attrition (1,200 departed / 7,468 employees)
- **Tool Room**: 15.73% attrition (633 departed / 4,024 employees) — **Lowest turnover department**

### Job Role Turnover (`results/sql_jobrole_analysis.csv`)
- **Operator**: 17.17% attrition (4,671 departed / 27,204 headcount) — Largest cohort and highest departure volume
- **Technician**: 17.03% attrition (1,736 departed / 10,196 headcount)
- **Inspector**: 16.06% attrition (574 departed / 3,573 headcount)
- **Engineer**: 15.28% attrition (620 departed / 4,058 headcount)
- **Supervisor**: 14.71% attrition (731 departed / 4,969 headcount) — Lowest departure rate

### Average Continuous Profiles (`results/sql_key_variable_analysis.csv`)
- **Engagement Score**: 61.89 (Departed) vs. 68.62 (Retained)
- **Monthly Overtime**: 27.70 hours (Departed) vs. 25.16 hours (Retained)
- **Commute Distance**: 33.37 km (Departed) vs. 31.37 km (Retained)
- **Monthly Income**: INR 30,852.19 (Departed) vs. INR 31,577.14 (Retained)

---

## 📈 Statistical Hypothesis Testing

In `R/05_statistical_analysis.R`, formal non-parametric and independence tests were executed:

### Wilcoxon Rank-Sum Tests (`results/wilcoxon_results.csv`)
All 9 tested continuous variables exhibited statistically significant differences ($p < 0.01$) between retained and departed employees:
- `EngagementScore`: $W = 229,466,352.5$, $p < 0.0001$ (Significant)
- `SatisfactionIndex`: $W = 197,928,623.5$, $p < 0.0001$ (Significant)
- `EnvironmentSatisfaction`: $W = 195,474,699.5$, $p < 0.0001$ (Significant)
- `MonthlyOvertimeHours`: $W = 152,468,614.5$, $p = 4.51 \times 10^{-69}$ (Significant)
- `DistanceFromHome`: $W = 160,016,905.0$, $p = 1.53 \times 10^{-29}$ (Significant)
- `MonthlyIncome`: $W = 182,900,305.5$, $p = 9.77 \times 10^{-15}$ (Significant)
- `YearsAtCompany`: $W = 176,829,143.0$, $p = 0.0071$ (Significant)

### Chi-Square Tests of Independence (`results/chi_square_results.csv`)
- **Significant**: `EngagementLevel` ($\chi^2 = 1870.27$, $p < 0.0001$), `OvertimeLevel` ($\chi^2 = 273.58$, $p < 0.0001$), `DistanceCategory` ($\chi^2 = 94.29$, $p < 0.0001$), `JobRole` ($\chi^2 = 26.16$, $p < 0.0001$), `Department` ($\chi^2 = 14.20$, $p = 0.0275$), `Shift` ($\chi^2 = 9.52$, $p = 0.0086$).
- **Non-Significant**: `Plant` ($p = 0.8641$), `Education` ($p = 0.6710$), `EducationField` ($p = 0.1267$), `TenureGroup` ($p = 0.1275$).

### Point-Biserial Correlation (`results/attrition_correlations.csv`)
- `EngagementScore`: $r = -0.2115$ (Strongest protective factor)
- `SatisfactionIndex`: $r = -0.0932$
- `MonthlyOvertimeHours`: $r = +0.0800$ (Strongest risk factor)
- `DistanceFromHome`: $r = +0.0510$

---

## 🤖 10 Machine Learning Baseline Models

In `R/06_machine_learning_models.R` and `R/07_model_evaluation.R`, 10 baseline algorithms were evaluated on the 9,999-record test partition under the default 0.50 probability cutoff:

### Baseline Performance Table (`results/model_evaluation_results.csv`)

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Discriminant Analysis (LDA)** | 83.47% | 0.5802 | 0.0282 | 0.0538 | **0.6847** |
| **Logistic Regression (GLM)** | 83.45% | 0.5663 | 0.0282 | 0.0537 | **0.6846** |
| **Gradient Boosting Machine (GBM)** | 83.34% | 0.5000 | 0.0114 | 0.0223 | **0.6807** |
| **Elastic Net (GLMNET)** | 83.38% | 0.5833 | 0.0084 | 0.0166 | **0.6826** |
| **K-Nearest Neighbors (KNN)** | 83.21% | 0.3415 | 0.0084 | 0.0164 | 0.5662 |
| **XGBoost (Baseline)** | 83.38% | **0.8333** | 0.0030 | 0.0060 | **0.6789** |
| **Random Forest (Baseline)** | 83.32% | 0.4000 | 0.0024 | 0.0048 | 0.6516 |
| **Decision Tree (CART)** | 83.34% | NA | 0.0000 | NA | 0.5000 |
| **Naive Bayes** | 83.34% | NA | 0.0000 | NA | 0.6707 |
| **Support Vector Machine (SVM)** | 83.34% | NA | 0.0000 | NA | 0.5331 |

> ⚠️ **Default Threshold Failure Mode**: Every model achieves ~83.3% accuracy simply by predicting the majority class (`No`). At threshold 0.50, Decision Tree, Naive Bayes, and SVM predict 0 positive cases (Recall = 0). XGBoost and Random Forest capture fewer than 6 out of 1,666 true attritions. This empirical collapse necessitates hyperparameter tuning and threshold calibration.

---

## 🔧 Hyperparameter Tuning

In `R/08_hyperparameter_tuning.R`, a representative 10,000-row stratified training subset was evaluated using 3-fold cross-validation optimizing ROC-AUC:
- **Random Forest**: Grid `mtry` $\in \{5, 10\}$ with 100 trees. Optimal: `mtry = 5`.
- **Gradient Boosting**: Grid `depth` $\in \{1, 3\}$, `n.trees` $\in \{100, 200\}$, `shrinkage = 0.1`. Optimal: `depth = 3`, `n.trees = 100`.
- **XGBoost**: 8-combination grid on `max_depth` $\in \{3, 5\}$, `eta` $\in \{0.05, 0.10\}$, `min_child_weight` $\in \{1, 5\}$ with `subsample = 0.8`, `colsample_bytree = 0.8`. Optimal: `max_depth = 5`, `eta = 0.10`, `min_child_weight = 1` (Cross-Validated Train AUC = **0.8868**).

### Tuned Model Comparison at Default Threshold 0.50 (`results/tuned_model_comparison.csv`)

| Tuned Architecture | Accuracy | Kappa | Precision | Recall | F1-Score | Specificity | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned Gradient Boosting** | 83.41% | 0.0101 | 0.7333 | 0.0066 | 0.0131 | 99.95% | **0.6790** |
| **Tuned XGBoost** | 83.24% | 0.0161 | 0.4107 | 0.0138 | 0.0267 | 99.60% | **0.6716** |
| **Tuned Random Forest** | 83.33% | 0.0062 | 0.4706 | 0.0048 | 0.0095 | 99.89% | 0.6412 |

---

## 🎯 Classification Threshold Optimization

In `R/11_threshold_analysis.R`, probability thresholds were swept across $[0.10, 0.90]$ in increments of 0.05. Across all three architectures, the optimal threshold maximizing F1-score converged at **`0.20`**:

### Optimal Threshold Comparison (`results/best_thresholds.csv`)

| Model Architecture | Optimal Threshold | Accuracy | Precision | Recall | F1-Score | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **0.20** | **69.19%** | **27.43%** | **51.62%** | **0.3583** | **72.70%** |
| **Gradient Boosting** | **0.20** | 70.06% | 27.78% | 49.82% | 0.3567 | 74.10% |
| **Random Forest** | **0.20** | 63.03% | 23.79% | 55.34% | 0.3328 | 64.56% |

### XGBoost Metric Progression Across Thresholds (`results/threshold_analysis.csv`)
- **Threshold 0.10**: Recall = 86.43%, Precision = 20.60%, F1 = 0.3327, Specificity = 33.39%
- **Threshold 0.15**: Recall = 67.29%, Precision = 23.55%, F1 = 0.3488, Specificity = 56.32%
- **Threshold 0.20 (Optimal)**: **Recall = 51.62%, Precision = 27.43%, F1 = 0.3583, Specificity = 72.70%**
- **Threshold 0.25**: Recall = 35.83%, Precision = 30.60%, F1 = 0.3301, Specificity = 83.75%
- **Threshold 0.30**: Recall = 22.93%, Precision = 32.82%, F1 = 0.2700, Specificity = 90.62%
- **Threshold 0.50 (Default)**: Recall = 1.38%, Precision = 41.07%, F1 = 0.0267, Specificity = 99.60%

---

## 🏆 Final Champion Model & Operational Results

The final champion model selected for operational deployment is **Tuned XGBoost** operating at a **classification threshold of 0.20**.

### Complete Performance Scorecard (`results/final_model_metrics.csv`)

| Metric Parameter | Observed Value | Operational Significance |
| :--- | :---: | :--- |
| **Model Algorithm** | **Tuned XGBoost** | Gradient boosted regression trees with L1/L2 regularization |
| **Decision Threshold** | **0.20** | Calibrated to population base attrition rate (~16.66%) |
| **Area Under ROC Curve (ROC-AUC)** | **0.6716** | Statistically solid discriminative separation |
| **Overall Classification Accuracy** | **69.19%** | 6,918 correct classifications out of 9,999 test records |
| **Recall / Sensitivity (TPR)** | **51.62%** | **Intercepts 860 out of 1,666 departing employees** |
| **Precision (PPV)** | **27.43%** | Nearly 1 in every 3.6 flagged employees is an authentic flight risk |
| **F1-Score** | **0.3583** | Peak harmonic balance between recall and precision |
| **Specificity (TNR)** | **72.70%** | 6,058 retained employees correctly left undisturbed |
| **Cohen's Kappa ($\kappa$)** | **0.1798** | Statistically significant performance beyond chance agreement |

### Final Confusion Matrix ($N = 9,999$ Test Records)

```
                              ACTUAL ATTRITION
                         Yes (1,666)     No (8,333)       Total Predicted
PREDICTED  Yes (Risk)      860 (TP)       2,275 (FP)          3,135
           No (Stay)       806 (FN)       6,058 (TN)          6,864
Total Actual              1,666           8,333               9,999
```

- **True Positives (TP = 860)**: 860 departing workers correctly identified for retention outreach.
- **False Negatives (FN = 806)**: 806 departures missed (down from 1,643 false negatives at default 0.50).
- **False Positives (FP = 2,275)**: 2,275 retained workers flagged. In HR operations, false positives carry benign costs—conducting career development dialogues strengthens engagement.
- **True Negatives (TN = 6,058)**: 6,058 loyal employees correctly identified as stable.

---

## 🔑 Feature Importance & Workforce Drivers

In `R/12_final_model_analysis.R`, feature importance was decomposed via `xgboost::xgb.importance`:

### Top 15 Predictor Gains (`results/final_feature_importance.csv`)

| Predictor Variable | Importance Gain | Observation Cover | Tree Frequency | Primary Operational Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **`EngagementScore`** | **35.71%** | 17.23% | 9.59% | Dominant master catalyst of voluntary turnover |
| **`SatisfactionIndex`** | **9.26%** | 11.41% | 6.89% | Validates engineered composite sentiment proxy |
| **`MonthlyOvertimeHours`** | **8.93%** | 10.23% | 8.80% | Chronic shop-floor fatigue; burnout threshold >40h |
| **`DistanceFromHome`** | **8.21%** | 11.37% | 10.71% | Commute burden (avg 33.4 km departed vs 31.4 km stayed) |
| **`MonthlyIncome`** | **6.94%** | 8.22% | 10.13% | Base financial compensation |
| **`MonthlyRate`** | **6.45%** | 9.01% | 11.18% | Variable monthly compensation |
| **`DailyRate`** | **5.80%** | 7.54% | 9.74% | Daily production compensation |
| **`HourlyRate`** | **4.94%** | 4.69% | 8.73% | Base hourly compensation |
| **`YearsAtCompany`** | **4.77%** | 7.98% | 8.62% | Institutional tenure and assimilation tier |
| **`Education`** | **1.17%** | 0.96% | 2.27% | Formal educational qualification code |
| **`JobInvolvement`** | **1.03%** | 2.49% | 1.91% | Psychological role investment |
| **`JobRoleOperator`** | **0.65%** | 0.94% | 1.19% | Frontline operator risk factor |
| **`DepartmentLogistics`** | **0.57%** | 1.36% | 0.90% | High-turnover department marker |
| **`JobRoleTechnician`** | **0.51%** | 0.71% | 0.79% | Technical maintenance role marker |
| **`EducationFieldLife Sciences`** | **0.51%** | 0.77% | 0.83% | Academic background specialization |

---

## 💡 Strategic Business Recommendations

1. **Overtime Capping & Rest Periods**: Monthly overtime accounts for 8.93% of model gain and correlates positively with attrition ($p < 0.0001$). Implement a hard cap of 25 overtime hours/month and mandate compensatory recovery rest.
2. **Digital Engagement Pulse Surveys**: Worker engagement drives 35.71% of model gain. Implement bi-weekly micro-pulse digital surveys to identify sentiment dips before formal resignation.
3. **Plant Transit Shuttles**: Commute distance accounts for 8.21% of model gain. Provide plant-sponsored transit shuttles connecting major residential corridors to Bawal and Manesar industrial facilities.
4. **Targeted Frontline Retention**: Operators represent 56% of all plant turnover. Direct retention bonuses, supervisor leadership training, and career progression frameworks toward shop-floor operators.

---

## 📁 Repository Structure

```
Employee Attrition Rate/
├── .gitignore                                # Excludes raw dataset and oversized model file
├── README.md                                 # Comprehensive project documentation
│
├── data/
│   └── README.md                             # Dataset details & exclusion explanation
│
├── R/                                        # Complete 12-stage R technical pipeline
│   ├── 01_data_import_check.R                # Data import, dimension & missing value validation
│   ├── 02_feature_engineering.R              # 5 composite feature engineering formulations
│   ├── 03_feature_selection.R                # Random Forest Gini ranking & leakage check
│   ├── 04_database_connectivity.R            # SQLite DB instantiation & structured SQL queries
│   ├── 05_statistical_analysis.R             # Wilcoxon tests, Chi-Square tests, correlations
│   ├── 06_machine_learning_models.R          # Training of 10 diverse machine learning algorithms
│   ├── 07_model_evaluation.R                # Baseline evaluation & threshold 0.50 analysis
│   ├── 08_hyperparameter_tuning.R            # 3-Fold CV tuning for RF, GBM, and XGBoost
│   ├── 09_validation_check.R                 # Automated data leakage & mathematical audits
│   ├── 10_tuned_model_evaluation.R           # Comparative benchmarking of tuned architectures
│   ├── 11_threshold_analysis.R               # Systematic sweep [0.10, 0.90] for optimal F1
│   └── 12_final_model_analysis.R             # Tuned XGBoost @ 0.20 scorecard & feature gains
│
├── results/                                  # Empirical CSV metrics, PNG plots, and RDS models
│   ├── hr_attrition.db                       # Embedded SQLite database (7.8 MB)
│   ├── tuned_xgboost_model.rds               # Champion Tuned XGBoost model object (78.6 KB)
│   ├── tuned_gradient_boosting.rds           # Tuned Gradient Boosting model object (1.68 MB)
│   ├── tuned_random_forest.rds               # Tuned Random Forest model object (2.23 MB)
│   ├── engineered_hr_data.rds                # Full engineered dataset object (1.23 MB)
│   ├── train_data.rds / test_data.rds        # 80/20 train/test partition objects
│   ├── final_model_metrics.csv               # Champion model evaluation scorecard
│   ├── final_model_summary.csv               # Champion model summary
│   ├── model_evaluation_results.csv          # 10 baseline models benchmark table
│   ├── tuned_model_comparison.csv            # Tuned models comparative metrics
│   ├── best_thresholds.csv                   # Optimal F1 threshold comparison
│   ├── threshold_analysis.csv                # Complete threshold sensitivity sweep [0.10, 0.90]
│   ├── final_feature_importance.csv          # Gain, Cover, Frequency feature importance
│   ├── selected_features.csv                 # Top 15 selected predictor names
│   ├── feature_importance_ranking.csv        # Variable-level Random Forest Gini rankings
│   ├── wilcoxon_results.csv                  # Wilcoxon Rank-Sum test statistics and p-values
│   ├── chi_square_results.csv                # Chi-Square independence test statistics
│   ├── attrition_correlations.csv            # Point-biserial correlations with attrition
│   ├── sql_department_analysis.csv           # SQL query: Department headcount and turnover rates
│   ├── sql_jobrole_analysis.csv              # SQL query: Job role headcount and turnover rates
│   ├── sql_key_variable_analysis.csv         # SQL query: Continuous variable averages by cohort
│   ├── sql_attrition_distribution.csv        # SQL query: Population retention/attrition counts
│   ├── final_confusion_matrix.png            # Confusion matrix plot (Threshold 0.20)
│   ├── final_roc_curve.png                   # Final ROC curve (AUC = 0.672)
│   ├── final_feature_importance.png          # Top 15 feature importance gain plot
│   ├── tuned_model_comparison.png            # Tuned architectures comparative bar plot
│   ├── tuned_roc_curves.png                  # Comparative ROC curves for tuned models
│   └── tuned_precision_recall_curves.png     # Precision-Recall trade-off curves
│
├── docs/                                     # Formal academic project reports
│   ├── DA1_Report_Formatted.docx             # Digital Assignment 1 Report (Preserved)
│   └── DA2_Final_Report.docx                 # Digital Assignment 2 Final Report (24 Sections)
│
└── presentation/                             # Professional presentation slide decks
    ├── DA1_Presentation (1).pptx             # Digital Assignment 1 Presentation (Preserved)
    └── DA2_Final_Presentation.pptx           # Digital Assignment 2 Presentation (20 Slides)
```

---

## 💻 Technologies & Packages Used

- **Language**: R (v4.0+)
- **Data Manipulation**: `dplyr`, `tidyr`, `readxl`
- **Machine Learning**: `caret`, `xgboost`, `randomForest`, `gbm`, `e1071`, `glmnet`, `MASS`, `naivebayes`, `kernlab`
- **Model Evaluation**: `pROC`
- **Visualization**: `ggplot2`
- **Database Connectivity**: `DBI`, `RSQLite`
- **Document & Slide Generation**: `python-docx`, `python-pptx`

---

## 🔬 Reproducibility / How to Run the Analysis

1. **Prerequisites**: Install R and the required packages:
   ```r
   install.packages(c("readxl", "dplyr", "tidyr", "caret", "randomForest", 
                      "gbm", "xgboost", "glmnet", "e1071", "MASS", 
                      "naivebayes", "pROC", "ggplot2", "DBI", "RSQLite"))
   ```
2. **Execution Order**:
   - The analysis scripts are designed to run sequentially from `01` to `12`:
     ```r
     source("R/01_data_import_check.R")
     source("R/02_feature_engineering.R")
     source("R/03_feature_selection.R")
     source("R/04_database_connectivity.R")
     source("R/05_statistical_analysis.R")
     source("R/06_machine_learning_models.R")
     source("R/07_model_evaluation.R")
     source("R/08_hyperparameter_tuning.R")
     source("R/09_validation_check.R")
     source("R/10_tuned_model_evaluation.R")
     source("R/11_threshold_analysis.R")
     source("R/12_final_model_analysis.R")
     ```
   - *Note*: Running script `01` requires the local raw Excel dataset `Uno_Minda_Multi_Plant_HR_Data_50000_Realistic.xlsx` in `data/`. For environments without the raw Excel file, all serialized intermediate objects (`results/engineered_hr_data.rds`, `results/train_data.rds`, `results/test_data.rds`, and model `.rds` files) and SQLite database are included, enabling execution and evaluation from stage `07` onward without retraining.

---

## ⚠️ Limitations & Threats to Validity

1. **Cross-Sectional Observation Window**: The dataset captures a cross-sectional snapshot rather than longitudinal panel data, precluding dynamic tracking of consecutive quarterly engagement drops.
2. **Synthetic / Anonymized Artifacts**: Uniform distributions across certain categorical variables (such as education field and plant sites) represent anonymized generation benchmarks that dampen site-specific variance.
3. **Precision-Recall Operational Trade-off**: Deploying at threshold 0.20 yields a Precision of 27.43% (FP = 2,275). While highly justified in human resources to capture 51.62% of departures, it requires managerial tolerance for false alarms.
4. **Absence of Qualitative Feedback**: The modeling framework currently operates on structured tabular data, lacking unstructured exit interview narratives or macroeconomic regional unemployment indices.

---

## 🚀 Future Enhancements

1. **Survival Analysis & Time-to-Event Modeling**: Formulate turnover using Cox Proportional Hazards and Kaplan-Meier estimators to model the temporal probability curve of when an employee will depart.
2. **Explainable AI (XAI) Dashboards**: Deploy an interactive Streamlit or R Shiny web dashboard integrating SHAP (Shapley Additive exPlanations) watermarks for individualized risk factor inspection by plant HR managers.
3. **Natural Language Processing (NLP)**: Incorporate unstructured textual feedback from employee reviews and peer appraisals via transformer embeddings (BERT/RoBERTa).
4. **Cost-Sensitive Objective Functions**: Embed asymmetric financial replacement matrices directly into XGBoost's custom gradient loss function.
