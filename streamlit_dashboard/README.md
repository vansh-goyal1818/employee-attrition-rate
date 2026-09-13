# Employee Attrition Prediction & Workforce Analytics: Streamlit Dashboard

**Digital Assignment 3 (DA3) — Criterion 2 (Interactive Visualization / Dashboard — 2 Marks)**  
*Course:* Programming for Data Science | School of Computer Science and Engineering (SCOPE), VIT Chennai  
*Academic Term:* Fall 2026 | Submission Date: 16 October 2026  

---

### Project Team
- **Vansh Goyal** (`24BDS1156`)
- **Ritabrata Dey** (`24BDS1129`)
- **Chetan Pareek** (`24BDS1125`)

---

## 1. Dashboard Purpose & Architecture

This interactive Python Streamlit application serves as an executive early-warning decision support system for voluntary employee attrition in Indian automotive manufacturing plants. Operating on **50,000 workforce records** across 6 production facilities, the dashboard enables human resource directors and plant operations heads to:

1. **Benchmark Models:** Evaluate 10 baseline classification algorithms and 3 tuned tree ensembles.
2. **Calibrate Decision Boundaries:** Interactively adjust classification probability thresholds ($T \in [0.10, 0.90]$) to balance sensitivity against false alarms.
3. **Score Individual Employees in Real Time:** Use the trained R Tuned XGBoost model via an R-to-Python prediction bridge.
4. **Segment the Workforce:** Filter 50,000 records dynamically across Department, Plant, Job Role, and Overtime ranges.
5. **Synthesize Strategic Interventions:** Provide policy-level recommendations targeting overtime, transit, and front-line mentorship.

---

## 2. Software Stack & Dependencies

The application leverages Python 3.12 with Streamlit, Plotly, Pandas, NumPy, and Scikit-Learn, coupled with R 4.6.1 for model inference:

| Component | Technology | Version | Purpose |
|:---|:---|:---:|:---|
| **Web Framework** | `streamlit` | `>=1.35.0` | Responsive multi-tab executive dashboard |
| **Data Manipulation** | `pandas` | `>=2.0.0` | In-memory data slicing and aggregation |
| **Numerical Processing** | `numpy` | `>=1.24.0` | Fast vector operations and threshold evaluations |
| **Interactive Plotting** | `plotly` | `>=5.15.0` | Dynamic charts with rich hover cards |
| **Database Querying** | `sqlite3` | Built-in | Direct querying of `results/hr_attrition.db` |
| **Prediction Bridge** | `Rscript` | `R 4.6.1` | Executing the trained R/caret XGBoost pipeline |

---

## 3. Installation & Launch Instructions

### Prerequisites
Ensure required Python packages are installed:
```bash
pip install -r streamlit_dashboard/requirements.txt
```

### Launch Command
From the project root directory (`D:\Employee Attrition Rate`), launch the dashboard:

```bash
streamlit run streamlit_app.py
```

The application will launch locally at `http://localhost:8501`.

---

## 4. R-to-Python Model Interoperability Bridge

The champion model (`results/tuned_xgboost_model.rds`) was developed in R using `caret` and `xgboost`. To maintain 100% mathematical fidelity without retraining or approximating in Python:

1. **Test Set Evaluation:** Precomputed test predictions from `results/test_data.rds` are stored in `streamlit_dashboard/test_predictions.csv`, allowing Streamlit to recompute confusion matrices and curves instantly.
2. **Individual Prediction Bridge (`streamlit_dashboard/predict_employee.R`):** When scoring a new employee profile in Tab 5, Python serializes the inputs into a JSON payload and calls `predict_employee.R` via `subprocess.run`. The R script loads `tuned_xgboost_model.rds`, `preprocessing_model.rds`, and `dummy_encoding_model.rds`, runs the exact preprocessing sequence, and returns the predicted probability as JSON to Streamlit.

### Verified Demonstration Scenarios:
- **Scenario A (High Risk Flight Hazard):** Assembly B Operator, 60h overtime, 55km commute, 35 engagement $\rightarrow$ **84.71% Probability** (Flagged for Immediate Review).
- **Scenario B (Low Risk Retained):** Quality Control Engineer, 5h overtime, 5km commute, 88 engagement $\rightarrow$ **1.49% Probability** (Stable).
- **Scenario C (Borderline Employee):** Logistics Technician, 25h overtime, 25km commute, 68 engagement $\rightarrow$ **14.93% Probability** (Routine Monitoring).

---

## 5. Dashboard Navigation Tabs

The dashboard is organized into **8 specialized modules**:

1. **Executive Overview:** High-level KPIs, 16.66% baseline attrition, class imbalance context, and threshold 0.20 justification.
2. **Model Comparison:** Interactive ranking and benchmark table across 10 baseline algorithms and 3 tuned tree ensembles.
3. **Threshold Analysis:** Interactive threshold slider ($0.10 \le T \le 0.90$) with live confusion matrix and performance curves.
4. **Workforce Analytics:** 50,000-record demographic exploration filtered by Department, Plant, Job Role, and Overtime.
5. **Employee Risk Calculator:** Real-time single-employee scoring engine with preset scenarios, custom sliders, and HR recommendations.
6. **Feature Importance:** Interactive Plotly horizontal bar charts for Information Gain, Cover, and Frequency across 31 features.
7. **Confusion Matrix & Diagnostics:** 2x2 interactive heatmap, full diagnostic performance table, and trade-off analysis.
8. **Business Insights:** Empirical findings, operational recommendations (overtime capping, transit routing, supervisory coaching), and ethical governance directives.

---

## 6. Ground-Truth Performance Verification

At key evaluation thresholds, the Streamlit dashboard reproduces the exact validated results:

- **At Threshold 0.20 (DA3 Optimal Policy):**
  - $\text{TP} = 860, \quad \text{FN} = 806, \quad \text{FP} = 2,275, \quad \text{TN} = 6,058$
  - $\text{Recall} = 51.62\%, \quad \text{Precision} = 27.43\%, \quad F_1 = 0.3583, \quad \text{Accuracy} = 69.19\%$
- **At Threshold 0.50 (Default Caret Policy):**
  - $\text{TP} = 23, \quad \text{FN} = 1,643, \quad \text{FP} = 33, \quad \text{TN} = 8,300$
  - $\text{Recall} = 1.38\%, \quad \text{Precision} = 41.07\%, \quad F_1 = 0.0267, \quad \text{Accuracy} = 83.24\%$
- **At Threshold 0.70 (Conservative Policy):**
  - $\text{TP} = 0, \quad \text{FN} = 1,666, \quad \text{FP} = 0, \quad \text{TN} = 8,333$
  - $\text{Recall} = 0.00\%, \quad \text{Precision} = 0.00\%, \quad F_1 = 0.0000, \quad \text{Accuracy} = 83.34\%$

---

## 7. Known Limitations

1. **Decision Support Nature:** Predictions are advisory and must not automate punitive actions.
2. **Industrial Focus:** Feature distributions reflect manufacturing assembly plants and require recalibration for corporate or software engineering sectors.
