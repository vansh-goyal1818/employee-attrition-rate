# Digital Assignment 3 (DA3): Streamlit Dashboard Technical Documentation
## Employee Attrition Prediction & Workforce Analytics Decision Support System

**Course:** Programming for Data Science | School of Computer Science and Engineering (SCOPE), VIT Chennai  
**Assessment:** Digital Assignment 3 (DA3) — Final Evaluation & Project Demonstration  
**Weightage:** 10 Marks | Submission Date: 16 October 2026  
**Evaluation Focus:** Criterion 2 — Interactive Visualization / Dashboard (2 Marks)  

---

### Project Authors & Team Identification
| Team Member Name | Registration Number | Degree & Specialization |
|:---|:---|:---|
| **Vansh Goyal** | `24BDS1156` | B.Tech Computer Science & Engineering (Data Science) |
| **Ritabrata Dey** | `24BDS1129` | B.Tech Computer Science & Engineering (Data Science) |
| **Chetan Pareek** | `24BDS1125` | B.Tech Computer Science & Engineering (Data Science) |

**Academic Institution:** Vellore Institute of Technology (VIT), Chennai Campus  

---

## 1. Executive Purpose & Context

Voluntary employee turnover in continuous manufacturing operations impairs line balancing, spikes defect scrap rates, and forces expensive emergency contractor hiring. In the Indian automotive manufacturing sector, replacing a specialized machine operator or quality technician incurs an estimated industry cost of **$50,000** (₹40,00,000).

This **Python Streamlit Web Application** (`streamlit_app.py`) serves as the official DA3 interactive decision-support system. Operating on an enterprise dataset of **50,000 employees** across 6 automotive manufacturing facilities, the dashboard enables human resource executives and plant managers to:
1. Benchmark model performance across 10 supervised algorithms and 3 tuned tree ensembles.
2. Dynamically calibrate the operational classification threshold ($T \in [0.10, 0.90]$).
3. Score individual employee attrition risk in real time via an R prediction bridge using the validated Tuned XGBoost champion model.
4. Filter workforce demographics dynamically across facilities, departments, roles, and overtime brackets.
5. Translate empirical risk drivers into actionable HR intervention policies.

---

## 2. Alignment with DA3 Evaluation Rubric

The Streamlit dashboard directly satisfies and supports the official DA3 evaluation criteria:

| DA3 Criterion | Weight | How Addressed in Streamlit Dashboard |
|:---|:---:|:---|
| **1. Statistical Analysis & Performance Comparison** | 2 Marks | **Tabs 2, 3, 7:** Standardized comparative benchmarks across 10 baseline algorithms and 3 tuned ensembles; dynamic threshold metrics; ROC-AUC, F1, Recall, Precision, and Accuracy analysis. |
| **2. Interactive Visualization / Dashboard** | 2 Marks | **Complete Application:** 8-tab Streamlit dashboard featuring reactive Plotly visualizations, dynamic demographic filters, threshold sliders, and single-employee scoring. |
| **3. Result Interpretation & Discussion** | 2 Marks | **Tabs 5, 6, 7:** Deep-dive analysis of Information Gain (EngagementScore, SatisfactionIndex, Overtime), confusion matrix trade-offs, and prescriptive retention actions. |
| **4. Conclusion, Limitations & Future Scope** | 1 Mark | **Tab 8 & Docs:** 4 strategic HR action pillars, ethical algorithmic governance, model limitations, and HRIS integration roadmap. |
| **5. Final Report Quality & Code Organization** | 2 Marks | **Modular Architecture:** Clean separation into `streamlit_app.py`, `streamlit_dashboard/model_utils.py`, `streamlit_dashboard/utils.py`, and `streamlit_dashboard/predict_employee.R`. |
| **6. Final Presentation & Viva Voce** | 1 Mark | **Tab 5 & Documentation:** 3 validated demonstration scenarios ready for live oral defense and faculty demonstration. |

---

## 3. Streamlit Dashboard Architecture

The architecture decouples front-end visualization from backend model execution:

```
+-----------------------------------------------------------------------------------------+
|                                    STREAMLIT FRONTEND                                   |
|  - Multi-tab navigation sidebar (8 modules)                                             |
|  - KPI metric cards & interactive Plotly charts                                         |
|  - Dynamic threshold slider (T in [0.10, 0.90])                                         |
|  - Workforce segmentation filters (Department, Plant, Role, Overtime)                   |
+-----------------------------------------------------------------------------------------+
                                         |         ^
                       User Inputs       |         |  Rendered DataFrames & Figures
                                         v         |
+-----------------------------------------------------------------------------------------+
|                               STREAMLIT BACKEND UTILITIES                               |
|  - `streamlit_dashboard/utils.py`: SQLite workforce data query (`hr_attrition.db`)      |
|  - `streamlit_dashboard/model_utils.py`: Metric calculations & threshold contingency     |
+-----------------------------------------------------------------------------------------+
                                         |         ^
                       Single-Row JSON   |         |  Predicted Probability JSON
                                         v         |
+-----------------------------------------------------------------------------------------+
|                               R PREDICTION BRIDGE ENGINE                                |
|  - `streamlit_dashboard/predict_employee.R` invoked via `Rscript.exe`                  |
|  - Ingests 16 employee parameters; computes SatisfactionIndex                          |
|  - Applies exact R preprocessing (`preprocessing_model.rds`) and dummy encoding         |
|  - Scores via `tuned_xgboost_model.rds` -> Outputs probability                          |
+-----------------------------------------------------------------------------------------+
```

---

## 4. R-to-Python Interoperability Bridge

The champion model was developed in R using `caret` and `xgboost`. To guarantee **100% mathematical fidelity** to the validated DA2 results without retraining in Python:

1. **Dashboard Analytics (Pure Python):** The 9,999 test set predictions generated during stage 12 were exported to `streamlit_dashboard/test_predictions.csv`. In-memory NumPy vectorization computes confusion matrices and trade-off curves in sub-millisecond time.
2. **Individual Inference (R Bridge):** When evaluating custom employee profiles in Tab 5, Python packages the profile into a JSON payload and executes `streamlit_dashboard/predict_employee.R` via `subprocess.run`:
   - R loads `train_data.rds` to align categorical factor levels.
   - Computes `SatisfactionIndex = (EnvironmentSatisfaction + JobSatisfaction + JobInvolvement) / 3`.
   - Passes the 1-row data frame through `dummy_encoding_model.rds` and `preprocessing_model.rds`.
   - Scores the resulting matrix through `tuned_xgboost_model.rds`.
   - Returns a structured JSON response `{ "probability": ..., "satisfaction_index": ..., "status": "success" }` to Python.

---

## 5. Data Sources & Integration

The dashboard is purely a **consumer** of validated project artifacts:

| Artifact | Source File | Records / Size | Purpose |
|:---|:---|:---:|:---|
| **Workforce Database** | `results/hr_attrition.db` | 50,000 rows | Multi-variable demographic exploration in Tab 4 |
| **Test Predictions** | `streamlit_dashboard/test_predictions.csv` | 9,999 rows | Instant threshold evaluation in Tabs 3 & 7 |
| **Baseline Matrix** | `results/model_evaluation_results.csv` | 10 models | Algorithm comparisons in Tab 2 |
| **Tuned Matrix** | `results/tuned_model_comparison.csv` | 3 ensembles | Tuned ensemble comparisons in Tab 2 |
| **Feature Importance** | `results/final_feature_importance.csv` | 31 features | Gain, Cover, Frequency rankings in Tab 6 |
| **Trained Model** | `results/tuned_xgboost_model.rds` | 78.6 KB | Real-time inferencing via prediction bridge in Tab 5 |
| **Preprocessors** | `results/preprocessing_model.rds`, `results/dummy_encoding_model.rds` | 2.2 KB | Transformation pipeline in Tab 5 |

---

## 6. Feature Engineering & Subspace Definition

The live prediction engine processes 15 predictive features established in `results/selected_features.csv`:

```
Numerical Attributes (11):
- EngagementScore          (Scale: 30 to 95; Workforce Mean: 67.5)
- MonthlyIncome            (Scale: INR 18,000 to 45,000; Median: 31,431)
- MonthlyRate              (Scale: INR 15,000 to 50,000; Median: 32,430)
- DailyRate                (Scale: INR 700 to 1,800; Median: 1,253)
- HourlyRate               (Scale: INR 70 to 180; Median: 125)
- YearsAtCompany           (Scale: 0 to 28 yrs; Median: 3.3 yrs)
- DistanceFromHome         (Scale: 0 to 92 km; Median: 32 km)
- MonthlyOvertimeHours     (Scale: 0 to 78 hrs; Median: 26 hrs)
- Education                (Scale: 1 to 5; Median: 2)
- JobInvolvement           (Scale: 1 to 4; Median: 3)
- SatisfactionIndex        (Scale: 1.0 to 4.0; Mean of EnvSat, JobSat, JobInv)

Categorical Attributes (4):
- Department               (7 levels: Assembly A, Assembly B, Logistics, Maintenance, Packaging, Quality Control, Tool Room)
- Plant                    (6 levels: 2W Lighting Pune, 4W Lighting Manesar, 4W Lighting Pune, AW4W Bawal, Densoten Bawal, Switch Manesar)
- EducationField           (6 levels: Electrical, Electronics, Industrial, Life Sciences, Mechanical, Other)
- JobRole                  (5 levels: Engineer, Inspector, Operator, Supervisor, Technician)
```

---

## 7. Mathematical Optimization of Decision Threshold ($T = 0.20$)

Because voluntary attrition represents a minority class (16.66%), standard default thresholding ($T = 0.50$) produces an operational failure known as the **Recall Trap**:

| Operating Threshold ($T$) | TP | FN | FP | TN | Sensitivity (Recall) | Precision | Specificity | F1-Score | Overall Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.10** | 1,439 | 227 | 5,551 | 2,782 | 86.37% | 20.59% | 33.39% | 0.3325 | 42.21% |
| **0.20 (DA3 Calibrated)** | **860** | **806** | **2,275** | **6,058** | **51.62%** | **27.43%** | **72.70%** | **0.3583** | **69.19%** |
| **0.25** | 597 | 1,069 | 1,353 | 6,980 | 35.83% | 30.62% | 83.76% | 0.3302 | 75.78% |
| **0.50 (Uncalibrated)** | **23** | **1,643** | **33** | **8,300** | **1.38%** | **41.07%** | **99.60%** | **0.0267** | **83.24%** |
| **0.70 (Conservative)** | **0** | **1,666** | **0** | **8,333** | **0.00%** | **0.00%** | **100.00%** | **0.0000** | **83.34%** |

### Mathematical & Business Rationale for $T = 0.20$:
1. **Turnover Capture:** Increases True Positives from 23 to **860** (+837 leavers caught in the holdout test set alone, representing over 4,180 additional employees caught enterprise-wide).
2. **Acceptable False Alarm Rate:** Retains a Specificity of **72.70%**, ensuring that 6,058 out of 8,333 retained employees are never unnecessarily flagged.
3. **F1 Peak:** Achieves an F1-Score of **0.3583** (a 13.4-fold improvement over the uncalibrated 0.0267).
4. **Asymmetric Cost Structure:** In manufacturing HR, a False Negative ($50,000 replacement penalty) is **16.6x more costly** than a False Positive ($3,000 proactive retention intervention cost).

---

## 8. Tab-by-Tab Functional Specifications

- **Tab 1: Executive Overview:** Displays 5 core KPI cards, Plotly donut chart of class distribution, methodology summary, and class imbalance explanation.
- **Tab 2: Model Comparison:** Dynamic ranking of 10 baseline algorithms vs 3 tuned tree ensembles; interactive horizontal bar chart and test set table; detailed explanation of the Accuracy Paradox.
- **Tab 3: Threshold Analysis:** Dynamic slider ($0.10 \le T \le 0.90$, step 0.01) with presets (0.20, 0.25, 0.50, 0.70); live 2x2 contingency matrix; multi-metric performance trajectories chart.
- **Tab 4: Workforce Analytics:** Live demographic slicing over 50,000 records from `hr_attrition.db` by Department, Plant, Job Role, and Overtime; dynamic KPI cards and 4 Plotly charts.
- **Tab 5: Employee Risk Calculator:** 16-parameter input matrix; dynamic `SatisfactionIndex` calculation; live scoring via R prediction bridge; 3 validated demonstration scenario presets; automated prescriptive HR recommendations; mandatory academic disclaimer.
- **Tab 6: Feature Importance:** Horizontal bar charts for Information Gain, Cover, and Frequency across Top 10, 15, 25, or 31 features; domain discussion of primary turnover drivers.
- **Tab 7: Confusion Matrix & Diagnostics:** Interactive 2x2 Plotly heatmap; comprehensive diagnostic table; detailed operational error definitions.
- **Tab 8: Business Insights:** 4 strategic HR action pillars (overtime capping, transit routing, supervisory coaching, decision-support integration) and ethical directives.

---

## 9. Single-Employee Demonstration Scenarios

Tab 5 includes three clearly labeled **Illustrative Demonstration Scenarios** evaluated through the live prediction bridge:

| Scenario Label | Archetype Profile | Overtime | Commute | Engagement | Satisfaction | Predicted Probability | Model Classification ($T=0.20$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scenario A** | High Risk Flight Hazard (Assembly B Operator) | 60 hrs | 55 km | 35 / 95 | 1.00 / 4.00 | **84.71%** | **Higher Risk (Action Required)** |
| **Scenario B** | Low Risk Retained (Quality Control Engineer) | 5 hrs | 5 km | 88 / 95 | 4.00 / 4.00 | **1.49%** | **Lower Risk (Stable Talent)** |
| **Scenario C** | Borderline Employee (Logistics Technician) | 25 hrs | 25 km | 68 / 95 | 2.33 / 4.00 | **14.93%** | **Lower Risk (Routine Monitoring)** |

---

## 10. Automated Verification & Testing

The Streamlit dashboard was verified via an automated test suite:
1. **Startup Verification:** Successfully starts and serves on localhost.
2. **Threshold Contingency Verification:**
   - At $T = 0.20$: $\text{TP}=860, \text{FN}=806, \text{FP}=2275, \text{TN}=6058$ (Exact match).
   - At $T = 0.50$: $\text{TP}=23, \text{FN}=1643, \text{FP}=33, \text{TN}=8300$ (Exact match).
   - At $T = 0.70$: $\text{TP}=0, \text{FN}=1666, \text{FP}=0, \text{TN}=8333$ (Exact match).
3. **Prediction Bridge Verification:** All 3 scenarios produce exact probabilities matching the original R pipeline.
4. **Data Integrity:** No files in `R/01` to `R/12` or `results/` were modified.

---

## 11. Limitations & Future Scope

### Limitations
1. **Advisory Decision Support:** Model outputs are advisory and must not automate punitive actions.
2. **Survey Response Latency:** Engagement Score and Satisfaction Index rely on quarterly pulse surveys, which may have latency.

### Future Scope
1. **Direct HRIS API Integration:** Expose the scoring engine as a RESTful API for direct ingestion into SAP SuccessFactors or Workday.
2. **Automated Batch Alerts:** Implement monthly scheduled batch scoring with automated email digests alerting plant managers to emerging flight hazards.

---

## 12. How to Launch the Application

From the project root directory (`D:\Employee Attrition Rate`):

```bash
streamlit run streamlit_app.py
```
