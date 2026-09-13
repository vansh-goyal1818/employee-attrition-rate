"""
Utility and helper functions for the Employee Attrition Streamlit Dashboard.
Handles database connections, data caching, formatting, and recommendation generation.
"""

import sqlite3
import pandas as pd
import streamlit as st

DATABASE_PATH = "results/hr_attrition.db"

@st.cache_data
def load_workforce_data():
    """
    Loads the full 50,000-employee workforce dataset directly from SQLite.
    Returns a pandas DataFrame.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql_query("SELECT * FROM employee_data", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading workforce database from {DATABASE_PATH}: {e}")
        return pd.DataFrame()

def generate_hr_recommendations(profile, probability, threshold=0.20):
    """
    Synthesizes tailored, policy-level HR retention recommendations based on
    the individual employee profile attributes and predicted turnover risk.
    """
    recs = []
    
    ot = profile.get("MonthlyOvertimeHours", 0)
    dist = profile.get("DistanceFromHome", 0)
    eng = profile.get("EngagementScore", 67)
    tenure = profile.get("YearsAtCompany", 3.5)
    env_sat = profile.get("EnvironmentSatisfaction", 2)
    job_sat = profile.get("JobSatisfaction", 2)
    job_inv = profile.get("JobInvolvement", 3)
    sat_idx = (env_sat + job_sat + job_inv) / 3.0

    if ot > 28:
        recs.append({
            "title": "Overtime Capping & Shift Load Balancing",
            "detail": f"Monthly overtime ({ot} hrs) exceeds the sustainable manufacturing ceiling (28 hrs). Enact mandatory shift caps and reassign auxiliary tooling to relief operators.",
            "type": "warning"
        })
        
    if dist > 30:
        recs.append({
            "title": "Industrial Transit & Commute Subsidies",
            "detail": f"Long one-way commute distance ({dist} km). Enroll the employee in plant-chartered shuttle routes or evaluate flexible shift clustering to reduce daily travel fatigue.",
            "type": "info"
        })
        
    if eng < 60:
        recs.append({
            "title": "Supervisory Mentorship & Pulse Review",
            "detail": f"Engagement score ({eng:.0f} / 95) falls within the vulnerable quartile. Schedule structured bi-weekly 1-on-1 development sessions with the shift manager.",
            "type": "warning"
        })
        
    if sat_idx < 2.0:
        recs.append({
            "title": "Workplace Environment & Ergonomic Audit",
            "detail": f"Composite satisfaction index ({sat_idx:.2f} / 4.00) indicates physical or environmental distress. Initiate a confidential workstation and shift climate assessment.",
            "type": "warning"
        })
        
    if tenure < 2.0:
        recs.append({
            "title": "Early-Career Onboarding Peer Buddy",
            "detail": f"Tenure is under 2.0 years ({tenure:.1f} yrs). Pair the employee with a senior technician mentor to reinforce institutional belonging.",
            "type": "info"
        })
        
    if probability >= threshold:
        recs.append({
            "title": "Proactive HR Retention Protocol",
            "detail": f"Model-estimated turnover probability ({probability*100:.1f}%) meets or exceeds the calibrated action threshold ({threshold*100:.1f}%). Initiate retention conversation within 14 business days.",
            "type": "danger"
        })
    else:
        recs.append({
            "title": "Routine Talent Monitoring",
            "detail": f"Turnover risk ({probability*100:.1f}%) remains comfortably below the action threshold ({threshold*100:.1f}%). Maintain standard annual review cycle.",
            "type": "success"
        })
        
    return recs
