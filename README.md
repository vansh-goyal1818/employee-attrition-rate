# Employee Attrition Prediction and Workforce Analytics

An end-to-end machine learning, statistical modeling, and interactive decision-support system designed to identify employee turnover risk and provide evidence-based workforce insights.

---

## 📌 Project Overview

Voluntary employee turnover creates operational and workforce-planning challenges across industrial organizations. When skilled operators, technicians, and team leaders depart unexpectedly, operations face productivity bottlenecks, scheduling disruptions, line balancing challenges, and substantial replacement costs.

Traditional human resource management often addresses attrition reactively through exit interviews conducted after an employee has already decided to leave. This project develops an end-to-end data science and machine learning pipeline that shifts workforce management toward proactive, data-informed intervention. By analyzing demographic, compensation, operational, and psychometric signals, the system estimates individual employee attrition risk and translates model outputs into actionable decision support.

---

## 🎯 Key Objectives

- **Analyze Workforce & Attrition Patterns**: Conduct comprehensive exploratory data analysis and statistical evaluation across a multi-plant workforce.
- **Engineer Meaningful HR Indicators**: Construct domain-specific composite features (such as `SatisfactionIndex` and operational workload tiers) to capture nuanced behavioral signals.
- **Identify Predictive Drivers**: Uncover empirical relationships and rank predictive features contributing to voluntary turnover without making unsupported causal claims.
- **Benchmark Multiple Machine Learning Models**: Evaluate 10 diverse supervised algorithms spanning linear, instance-based, kernel, and tree ensemble paradigms under standardized preprocessing.
- **Address Class Imbalance via Threshold Optimization**: Calibrate decision boundaries beyond default accuracy metrics to prioritize early leaver detection and maximize the F1-score.
- **Deliver an Interactive Decision-Support Interface**: Deploy an intuitive Streamlit web dashboard featuring real-time risk scoring, diagnostic analytics, and tailored retention guidelines.

---

## 📊 Dataset

The project utilizes an enterprise workforce dataset comprising **50,000 employee records** across 6 regional manufacturing facilities and 7 functional departments.

- **Total Records**: 50,000 individual employee rows
- **Feature Dimensions**: 21 original attributes
- **Target Variable**: `AttritionFlag` (`No` = 41,668 records [83.34%], `Yes` = 8,332 records [16.66%])
- **Class Ratio**: Moderate class imbalance of approximately 5:1

### Major Feature Categories

| Category | Primary Attributes | Description |
| :--- | :--- | :--- |
| **Organizational** | `Plant`, `Department`, `JobRole`, `Shift` | Factory facility, functional department, operational role, and shift schedule. |
| **Demographic & Educational** | `Education`, `EducationField`, `DistanceFromHome` | Education tier (1–5), field of study, and one-way commute distance in kilometers. |
| **Compensation** | `MonthlyIncome`, `MonthlyRate`, `DailyRate`, `HourlyRate` | Standardized compensation across hourly, daily, and monthly pay schedules. |
| **Psychometric & Sentiment** | `EnvironmentSatisfaction`, `JobSatisfaction`, `JobInvolvement`, `EngagementScore` | Standardized survey ratings measuring workplace climate, role satisfaction, and engagement. |
| **Operational & Tenure** | `YearsAtCompany`, `MonthlyOvertimeHours`, `SafetyIncidents_12M` | Organizational tenure, logged monthly overtime hours, and trailing 12-month safety incidents. |

> **Note on Data Access**: Raw enterprise spreadsheet files (`data/*.xlsx`) are excluded from this public repository via `.gitignore`. The complete analysis scripts, intermediate data objects, SQLite database, and empirical outputs are preserved within the repository.

---

## ⚙️ Analytics & Feature Engineering

To transform raw survey ratings and operational logs into expressive predictive signals, five domain-specific indicators were engineered:

1. **`SatisfactionIndex`** (Continuous, range [1.00, 4.00]):
   $$\text{SatisfactionIndex} = \frac{\text{EnvironmentSatisfaction} + \text{JobSatisfaction} + \text{JobInvolvement}}{3}$$
   Consolidates three collinear ordinal survey metrics into a unified continuous psychometric indicator reflecting overall workplace satisfaction.
2. **`EngagementLevel`** (Categorical: `Low`, `Medium`, `High`):
   Discretizes continuous engagement scores (<60.0: Low, 60.0–75.0: Medium, >75.0: High) to capture non-linear disengagement thresholds ($\chi^2 = 1870.27, p < 0.001$).
3. **`OvertimeLevel`** (Categorical: `Low`, `Moderate`, `High`):
   Partitions monthly overtime into operational workload tiers (<20 hrs: Low, 20–35 hrs: Moderate, >35 hrs: High) to isolate potential burnout zones ($\chi^2 = 273.58, p < 0.001$).
4. **`DistanceCategory`** (Categorical: `Near`, `Moderate`, `Far`):
   Segments one-way commute distances (<15 km: Near, 15–35 km: Moderate, >35 km: Far) to evaluate commute-related fatigue ($\chi^2 = 94.29, p < 0.001$).
5. **`TenureGroup`** (Categorical: `New`, `Developing`, `Established`, `Veteran`):
   Categorizes organizational tenure (<1 yr, 1–3 yrs, 3–7 yrs, >7 yrs) across career lifecycle stages.

---

## 🤖 Machine Learning

The modeling pipeline utilized an 80/20 stratified split (40,001 training records; 9,999 holdout test records: 8,333 retained, 1,666 leavers). All preprocessing transformations (dummy encoding of nominal variables and Z-score standardization of continuous features) were fitted strictly on the training partition to prevent data leakage.

### Evaluated Algorithms

Ten baseline supervised classification models were benchmarked:
1. **Logistic Regression** (Generalized Linear Model)
2. **Linear Discriminant Analysis (LDA)**
3. **Elastic Net** (L1/L2 Regularized Generalized Linear Model)
4. **Naive Bayes** (Probabilistic Classifier)
5. **K-Nearest Neighbors (KNN)** (Distance-Based Classifier)
6. **Support Vector Machines (SVM)** (Radial Basis Function Kernel)
7. **Decision Tree (CART)** (Recursive Partitioning)
8. **Random Forest** (Bagged Ensemble)
9. **Gradient Boosting Machine (GBM)** (Sequential Boosting)
10. **Extreme Gradient Boosting (XGBoost)** (Regularized Tree Boosting)

### Hyperparameter Tuning & Threshold Optimization

Hyperparameter tuning used 3-fold cross-validation for the tuned Random Forest and Gradient Boosting models, while XGBoost tuning evaluated the specified parameter configurations. While tuned Gradient Boosting achieved the highest overall discriminatory capability (ROC-AUC = 0.6790), all models evaluated at the standard default threshold of 0.50 suffered from the **Accuracy Paradox**: achieving ~83.3% accuracy while capturing less than 1.4% of actual leavers due to class imbalance.

Systematic decision-threshold sweeps across $T \in [0.10, 0.90]$ were conducted to align the decision boundary with operational retention priorities.

---

## 🏆 Final Model & Results

The final champion model selected for operational decision support is **Tuned XGBoost** operating at a calibrated classification threshold of **`0.20`**.

### Performance Scorecard (Holdout Test Set, $N = 9,999$)

| Metric Parameter | Observed Score | Practical Significance |
| :--- | :---: | :--- |
| **Model Algorithm** | **Tuned XGBoost** | Regularized gradient boosted decision tree ensemble |
| **Operational Threshold** | **0.20** | Calibrated to prioritize recall over naive majority-class accuracy |
| **ROC-AUC** | **0.6716** | Statistically solid discriminatory capability |
| **Recall / Sensitivity** | **51.62%** | **Correctly identifies 860 out of 1,666 departing employees** |
| **Precision** | **27.43%** | Approximately 1 in every 3.6 flagged employees is an authentic flight risk |
| **F1-Score** | **0.3583** | **Peak harmonic balance achieved across all evaluated models** |
| **Specificity** | **72.70%** | Correctly clears 6,058 out of 8,333 retained employees |
| **Overall Accuracy** | **69.19%** | Realistic balanced accuracy reflecting calibrated threshold trade-offs |
| **Cohen's Kappa ($\kappa$)** | **0.1798** | Statistically significant classification beyond chance agreement |

### Confusion Matrix Diagnostics ($T = 0.20$)

```
                         ACTUAL RETENTION (0)   ACTUAL ATTRITION (1)      TOTAL
PREDICTED RETENTION (0)       6,058 (TN)              806 (FN)            6,864
PREDICTED ATTRITION (1)       2,275 (FP)              860 (TP)            3,135
TOTAL                         8,333                 1,666                 9,999
```

- **Threshold Selection Rationale**: The decision boundary of 0.20 was chosen because it achieved the **highest F1-score (0.3583)** among all tuned models, whereas the default 0.50 threshold caught only 23 leavers (Recall = 1.38%). Identifying over 51% of potential departures provides actionable lead time for human resource interventions, while the false positive volume (2,275 across 10,000 records) represents low-risk opportunities for supportive check-ins and career development dialogues.

---

## 🔑 Key Findings

Feature importance was evaluated by computing the Information Gain contributed by each feature across the trained XGBoost decision trees.

### Top Predictive Signals

1. **`EngagementScore`** (Gain: 882.96): Primary predictor; workers exhibiting scores below 60.0 represent the highest concentration of voluntary turnover.
2. **`SatisfactionIndex`** (Gain: 370.11): Validates the composite sentiment indicator; lower scores associate consistently with departure propensity.
3. **`MonthlyOvertimeHours`** (Gain: 706.18): Primary physical workload signal; an empirical inflection point is observed at approximately 28 overtime hours/month.
4. **`DistanceFromHome`** (Gain: 724.05): Captures daily transit fatigue, with workers commuting >30 km exhibiting elevated departure rates.
5. **Compensation Metrics** (`MonthlyIncome`, `MonthlyRate`, `DailyRate`, `HourlyRate`): Contribute to tree split stratification across organizational tiers, though baseline pay displays minimal linear correlation with turnover in this standardized manufacturing pay structure.
6. **`YearsAtCompany`** (Gain: 758.51): Captures organizational tenure and career progression milestones.

> **Important Note on Interpretation**: Feature importance reflects mathematical split contributions within the decision trees and statistical associations within the dataset. These signals should be interpreted as predictive indicators rather than direct causal conclusions.

---

## 💻 Interactive Dashboard

The project includes a full-featured, mobile-responsive web dashboard built using **Python Streamlit** and **Plotly**, connected to an underlying R inference bridge:

```
[ Streamlit Web UI ] ──► [ Python Controller ] ──► [ R Prediction Bridge ] ──► [ Tuned XGBoost ]
 (8 Functional Tabs)      (NumPy Vectorization)     (predict_employee.R)        (Real-Time Score)
```

### Dashboard Modules

1. **Executive Overview**: High-level workforce KPIs (50,000 workforce, 16.66% turnover), champion model scorecard, and class imbalance explanation.
2. **Model Comparison**: Interactive benchmarking table and comparative bar charts across all 10 baseline algorithms and 3 tuned ensembles.
3. **Threshold Analysis**: Interactive slider ($0.10 \le T \le 0.90$) with real-time recalculation of confusion matrices, precision, recall, and F1 curves.
4. **Workforce Analytics**: Multi-dimensional population segmentation filtered by Plant, Department, Shift, and Overtime ranges.
5. **Employee Risk Calculator**: Interactive inference engine allowing HR managers to input 16 employee parameters, compute `SatisfactionIndex` dynamically, and receive a real-time predicted attrition probability and prescriptive recommendations.
6. **Feature Importance**: Interactive Plotly horizontal bar charts for Information Gain, Cover, and Frequency across predictors.
7. **Confusion Matrix & Diagnostics**: Interactive contingency heatmap, sensitivity-specificity trade-off curves, and classification error breakdowns.
8. **Business Insights**: Synthesis of empirical workforce findings and strategic HR retention action pillars.

---

## 📁 Project Structure

```
Employee Attrition Rate/
├── streamlit_app.py                      # Main 8-tab Streamlit dashboard application
├── streamlit_dashboard/                  # Modular dashboard backend and assets
│   ├── model_utils.py                    # Metric computations & NumPy threshold engine
│   ├── utils.py                          # SQLite database query & recommendation generator
│   ├── predict_employee.R                # Headless R prediction bridge script
│   ├── test_predictions.csv              # Precomputed test predictions for sub-ms slider response
│   ├── requirements.txt                  # Python dependencies for the dashboard
│   └── README.md                         # Dashboard technical documentation
├── R/                                    # End-to-end 12-stage R analytical pipeline
│   ├── 01_data_import_check.R            # Data import, dimensional audits, missing-value check
│   ├── 02_feature_engineering.R          # Derivation of 5 domain composite indicators
│   ├── 03_feature_selection.R            # Random Forest Gini ranking & leakage audit
│   ├── 04_database_connectivity.R        # SQLite DB creation & structured SQL queries
│   ├── 05_statistical_analysis.R         # Wilcoxon tests, Chi-Square tests, correlations
│   ├── 06_machine_learning_models.R      # Training 10 diverse machine learning baselines
│   ├── 07_model_evaluation.R             # Baseline evaluation & threshold 0.50 analysis
│   ├── 08_hyperparameter_tuning.R        # 3-Fold cross-validation tuning for tree ensembles
│   ├── 09_validation_check.R             # Mathematical checks & leakage validation
│   ├── 10_tuned_model_evaluation.R       # Benchmark comparison across tuned architectures
│   ├── 11_threshold_analysis.R           # Comprehensive threshold sensitivity sweep [0.10, 0.90]
│   └── 12_final_model_analysis.R         # Final XGBoost @ 0.20 scorecard & feature importance
├── results/                              # Generated model artifacts, tables, and plots
│   ├── hr_attrition.db                   # Relational SQLite database of workforce records
│   ├── tuned_xgboost_model.rds           # Champion Tuned XGBoost model artifact
│   ├── preprocessing_model.rds           # Caret normalization object (center/scale)
│   ├── dummy_encoding_model.rds          # Caret one-hot encoding object
│   ├── *.csv                             # Validated empirical metrics and summary tables
│   └── *.png                             # High-resolution ROC curves, PR curves, feature gain plots
├── docs/                                 # Detailed technical project documentation
│   ├── DA3_Final_Project_Report.docx     # 35-page consolidated final technical report
│   └── DA3_Dashboard_Documentation.md    # Technical dashboard architecture guide
├── presentation/                         # Executive presentation decks
│   └── DA3_Final_Presentation.pptx       # 16-slide final technical presentation
├── data/                                 # Data folder (raw datasets excluded via .gitignore)
│   └── README.md                         # Data documentation and source notes
├── .gitignore                            # Version control exclusion rules
└── README.md                             # Project repository documentation
```

---

## 🛠️ Technologies Used

- **Statistical Analysis & Modeling**: R (v4.0+), `caret`, `xgboost`, `randomForest`, `gbm`, `glmnet`, `e1071`, `pROC`
- **Dashboard & Visualization**: Python (v3.10+), `streamlit`, `plotly`, `pandas`, `numpy`, `scikit-learn`
- **Database & Data Storage**: SQLite, `sqlite3`, `DBI`, `RSQLite`
- **Version Control & Documentation**: Git, GitHub, `python-docx`, `python-pptx`

---

## 🚀 Running the Dashboard

### 1. Prerequisites

Ensure that Python 3.10+ and R are installed on your system.

### 2. Install Python Dependencies

From the project root directory, install the required packages:

```bash
pip install -r streamlit_dashboard/requirements.txt
```

### 3. Launch the Application

Run the Streamlit application from the project root:

```bash
streamlit run streamlit_app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 🔬 Reproducibility & Data Note

The raw enterprise datasets (`data/*.xlsx`) and large aggregate model objects (`results/ml_models.rds`) are intentionally excluded from version control via `.gitignore` to maintain a lightweight repository and respect data privacy practices.

All intermediate data representations, trained champion model files (`results/tuned_xgboost_model.rds`, `preprocessing_model.rds`, `dummy_encoding_model.rds`), SQLite database (`results/hr_attrition.db`), precomputed test predictions, and CSV summary tables are fully tracked. This allows the complete analytical pipeline, threshold evaluation, and Streamlit dashboard to execute without requiring raw data re-import or model retraining.

---

## ⚠️ Limitations

- **Demonstration Dataset**: The dataset reflects an industrial manufacturing workforce context; findings should not be assumed to generalize directly to service, healthcare, or technology industries without local recalibration.
- **Class Imbalance & Trade-offs**: With an 83.34% retention majority class, optimizing the threshold to 0.20 intentionally accepts lower precision (27.43%) to achieve actionable sensitivity (51.62%).
- **Predictive vs. Causal**: Feature importance metrics identify strong statistical associations and predictive splits within decision trees; they do not establish direct causal mechanisms.
- **Decision-Support Scope**: The model is designed exclusively as an early-warning advisory tool for HR professionals and supervisors. Algorithmic predictions should never serve as the sole justification for adverse employment or compensation actions.

---

## 🔮 Future Scope

- **Real-World Organizational Validation**: Validate model transferability and calibrate probabilities on longitudinal multi-organization workforce datasets.
- **Explainable AI (XAI)**: Incorporate real-time SHAP (Shapley Additive exPlanations) force plots directly into the employee scoring tab to provide localized factor attributions.
- **Probability Calibration**: Apply Platt Scaling or Isotonic Regression to align raw tree scores with true empirical probabilities.
- **Fairness & Demographic Auditing**: Implement algorithmic fairness toolkits (e.g., Fairlearn, AIF360) to evaluate demographic parity and equal opportunity metrics across workforce sub-populations.
- **Enterprise Pipeline Integration**: Connect the scoring engine to enterprise HR information systems (e.g., SAP SuccessFactors, Workday) via secure REST APIs.

---

## 👥 Contributors

- **Vansh Goyal**
- **Ritabrata Dey**
- **Chetan Pareek**
