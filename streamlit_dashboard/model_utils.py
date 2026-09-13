"""
Model utilities and inference bridge for the Employee Attrition Streamlit Dashboard.
Handles loading model evaluation tables, real-time threshold calculations,
and the R prediction bridge for individual employee risk scoring.
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
import streamlit as st

# Locate R executable
R_EXECUTABLE = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
if not os.path.exists(R_EXECUTABLE):
    R_EXECUTABLE = "Rscript"

BRIDGE_SCRIPT = os.path.abspath("streamlit_dashboard/predict_employee.R")
TEST_PREDICTIONS_CSV = "streamlit_dashboard/test_predictions.csv"

@st.cache_data
def load_test_predictions():
    """
    Loads precomputed holdout test set (9,999 records) actual labels and
    Tuned XGBoost probabilities.
    """
    if os.path.exists(TEST_PREDICTIONS_CSV):
        return pd.read_csv(TEST_PREDICTIONS_CSV)
    st.error(f"Missing {TEST_PREDICTIONS_CSV}. Run export script first.")
    return pd.DataFrame()

@st.cache_data
def load_baseline_models():
    """Loads baseline 10-model evaluation matrix."""
    path = "results/model_evaluation_results.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_tuned_models():
    """Loads tuned tree ensemble comparison matrix."""
    path = "results/tuned_model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_feature_importance():
    """Loads XGBoost feature importance rankings (Gain, Cover, Frequency)."""
    path = "results/final_feature_importance.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_threshold_analysis():
    """Loads precomputed threshold evaluation points."""
    path = "results/threshold_analysis.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def compute_metrics_at_threshold(threshold=0.20):
    """
    Dynamically computes the exact confusion matrix and classification metrics
    for the Tuned XGBoost model on the 9,999 test records at the given threshold.
    """
    df = load_test_predictions()
    if df.empty:
        return {}
    
    actual = df["Actual"].astype(str).str.strip().values
    probs = df["Probability"].values
    
    preds = np.where(probs >= threshold, "Yes", "No")
    
    tp = int(np.sum((actual == "Yes") & (preds == "Yes")))
    fn = int(np.sum((actual == "Yes") & (preds == "No")))
    fp = int(np.sum((actual == "No") & (preds == "Yes")))
    tn = int(np.sum((actual == "No") & (preds == "No")))
    total = len(actual)
    
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "threshold": threshold,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "Total": total,
        "Recall": rec,
        "Precision": prec,
        "F1_Score": f1,
        "Accuracy": acc,
        "Specificity": spec
    }

def predict_single_employee(employee_dict):
    """
    Executes the trained R XGBoost model pipeline via the R prediction bridge.
    Passes JSON profile to predict_employee.R and parses resulting JSON.
    """
    try:
        json_payload = json.dumps(employee_dict)
        res = subprocess.run(
            [R_EXECUTABLE, BRIDGE_SCRIPT, json_payload],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=30
        )
        if res.returncode != 0:
            return {"error": f"R script failed with code {res.returncode}: {res.stderr.strip()}"}
        
        # Extract last JSON line from stdout
        lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
        if not lines:
            return {"error": "R script produced no output."}
            
        last_line = lines[-1]
        data = json.loads(last_line)
        return data
    except Exception as e:
        return {"error": f"Exception while executing prediction bridge: {str(e)}"}
