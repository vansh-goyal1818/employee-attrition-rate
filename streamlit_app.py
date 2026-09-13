# ==============================================================================
# DIGITAL ASSIGNMENT 3 (DA3) - STREAMLIT DECISION SUPPORT DASHBOARD
# Project: Employee Attrition Prediction & Workforce Analytics Using Machine Learning
# Authors: Vansh Goyal (24BDS1156), Ritabrata Dey (24BDS1129), Chetan Pareek (24BDS1125)
# Course:  Programming for Data Science | SCOPE, VIT Chennai
# Date:    October 2026
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Insert current directory into path for modular imports
sys.path.insert(0, os.path.abspath("."))
from streamlit_dashboard.model_utils import (
    load_test_predictions,
    load_baseline_models,
    load_tuned_models,
    load_feature_importance,
    load_threshold_analysis,
    compute_metrics_at_threshold,
    predict_single_employee
)
from streamlit_dashboard.utils import (
    load_workforce_data,
    generate_hr_recommendations
)

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Employee Attrition Analytics | DA3",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional CSS - Theme-Independent with Explicit High-Contrast Colors
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1B365D !important; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #4A777A !important; margin-bottom: 20px; font-weight: 500; }
    
    /* KPI Metric Cards */
    .kpi-card { background-color: #F8FAFC !important; border-radius: 8px; padding: 18px; border: 1px solid #CBD5E1 !important; text-align: center; }
    .kpi-val { font-size: 1.8rem; font-weight: 800; color: #1B365D !important; }
    .kpi-lbl { font-size: 0.88rem; color: #475569 !important; font-weight: 700; text-transform: uppercase; margin-bottom: 4px; }
    .kpi-sub { font-size: 0.82rem; color: #64748B !important; font-weight: 500; }
    
    /* Callout & Banner Blocks (Ensuring dark readable text regardless of theme) */
    .callout-box { background-color: #F0FDF4 !important; border-left: 5px solid #16A34A !important; border: 1px solid #DCFCE7 !important; padding: 16px; border-radius: 6px; margin-bottom: 14px; }
    .callout-box, .callout-box * { color: #1E293B !important; }
    .callout-box b, .callout-box strong { color: #166534 !important; font-weight: 700; }
    
    .callout-warn { background-color: #FEFCE8 !important; border-left: 5px solid #EAB308 !important; border: 1px solid #FEF08A !important; padding: 16px; border-radius: 6px; margin-bottom: 14px; }
    .callout-warn, .callout-warn * { color: #1E293B !important; }
    .callout-warn b, .callout-warn strong { color: #854D0E !important; font-weight: 700; }
    
    .callout-alert { background-color: #FEF2F2 !important; border-left: 5px solid #DC2626 !important; border: 1px solid #FECACA !important; padding: 16px; border-radius: 6px; margin-bottom: 14px; }
    .callout-alert, .callout-alert * { color: #1E293B !important; }
    .callout-alert b, .callout-alert strong { color: #991B1B !important; font-weight: 700; }
    
    .callout-info { background-color: #F0F9FF !important; border-left: 5px solid #0284C7 !important; border: 1px solid #BAE6FD !important; padding: 16px; border-radius: 6px; margin-bottom: 14px; }
    .callout-info, .callout-info * { color: #1E293B !important; }
    .callout-info b, .callout-info strong { color: #0369A1 !important; font-weight: 700; }
    
    .team-badge { background-color: #E2E8F0; color: #1E293B !important; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/users-gear.svg", width=50)
    st.markdown("### **Workforce Analytics**")
    st.markdown("<span class='team-badge'>DA3 Evaluation</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    selected_tab = st.radio(
        "Navigation Menu:",
        [
            "1. Executive Overview",
            "2. Model Comparison",
            "3. Threshold Analysis",
            "4. Workforce Analytics",
            "5. Employee Risk Calculator",
            "6. Feature Importance",
            "7. Confusion Matrix & Diagnostics",
            "8. Business Insights"
        ]
    )
    
    st.markdown("---")
    st.markdown("#### **Project Attribution**")
    st.markdown("""
    **VIT Chennai (SCOPE)**  
    *Programming for Data Science*  
    - **Vansh Goyal** (`24BDS1156`)  
    - **Ritabrata Dey** (`24BDS1129`)  
    - **Chetan Pareek** (`24BDS1125`)  
    """)
    st.caption("Fall 2026 | Submission: 16 October 2026")

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================
if selected_tab == "1. Executive Overview":
    st.markdown("<div class='main-header'>Executive Overview: Workforce Attrition Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Digital Assignment 3 (DA3) — Machine Learning Early-Warning Decision Support Architecture</div>", unsafe_allow_html=True)
    
    # 5 KPI Cards with explicit dark text
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("<div class='kpi-card'><div class='kpi-lbl'>Total Workforce</div><div class='kpi-val'>50,000</div><div class='kpi-sub'>6 Indian Facilities</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='kpi-card'><div class='kpi-lbl'>Attrition Rate</div><div class='kpi-val' style='color:#DC2626 !important;'>16.66%</div><div class='kpi-sub'>8,332 Leavers</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='kpi-card'><div class='kpi-lbl'>Champion Model</div><div class='kpi-val' style='color:#16A34A !important;'>Tuned XGBoost</div><div class='kpi-sub'>ROC-AUC: 0.6716</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='kpi-card'><div class='kpi-lbl'>Calibrated Threshold</div><div class='kpi-val' style='color:#0284C7 !important;'>0.20</div><div class='kpi-sub'>Optimized Recall</div></div>", unsafe_allow_html=True)
    with col5:
        st.markdown("<div class='kpi-card'><div class='kpi-lbl'>Target Recall (Test)</div><div class='kpi-val' style='color:#D97706 !important;'>51.62%</div><div class='kpi-sub'>Precision: 27.43%</div></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_left, col_right = st.columns([7, 5])
    
    with col_left:
        st.subheader("Workforce Attrition Class Distribution")
        pie_df = pd.DataFrame({
            "Status": ["Retained (No)", "Attrited (Yes)"],
            "Count": [41668, 8332],
            "Percentage": ["83.34%", "16.66%"]
        })
        fig_pie = px.pie(
            pie_df, values="Count", names="Status",
            color="Status",
            color_discrete_map={"Retained (No)": "#1B365D", "Attrited (Yes)": "#DC2626"},
            hole=0.55
        )
        fig_pie.update_traces(textinfo="label+percent", hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}")
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=340)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        st.subheader("Project Methodology & Context")
        st.markdown("""
        - **Industrial Context:** Voluntary turnover across manufacturing corridors (Pune, Manesar, Bawal) disrupts just-in-time auto production lines.
        - **Replacement Liability:** Industry estimates indicate an average replacement cost of **$50,000** per skilled worker (loss of throughput, recruitment, onboarding scrap).
        - **Methodological Pipeline:** 10 supervised classification algorithms benchmarked under an 80/20 stratified partition, followed by hyperparameter tuning and decision threshold calibration.
        """)
        
    st.subheader("The Class Imbalance Challenge & Decision Threshold Selection")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background-color: #FEFCE8; border-left: 5px solid #EAB308; border: 1px solid #FEF08A; padding: 16px; border-radius: 6px; margin-bottom: 14px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #854D0E; margin-bottom: 5px;">1. The Imbalance Trap:</div>
            <div style="color: #1E293B; font-size: 0.95rem; line-height: 1.5;">
                With an attrition rate of 16.66%, the majority class (83.34%) heavily dominates. A naive classifier that blindly predicts 'No' for every single employee achieves an impressive <b style="color:#0F172A;">83.34% accuracy</b>, yet delivers exactly <b style="color:#DC2626;">0% Recall</b>, detecting zero leavers.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background-color: #F0FDF4; border-left: 5px solid #16A34A; border: 1px solid #DCFCE7; padding: 16px; border-radius: 6px; margin-bottom: 14px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #166534; margin-bottom: 5px;">2. Why Threshold 0.20 Was Selected:</div>
            <div style="color: #1E293B; font-size: 0.95rem; line-height: 1.5;">
                At the standard default threshold of 0.50, Tuned XGBoost caught only <b style="color:#DC2626;">23 out of 1,666 leavers (1.38% Recall)</b>. Calibrating the decision boundary to <b style="color:#166534;">0.20</b> elevated detection to <b style="color:#166534;">860 leavers (51.62% Recall)</b>, while maintaining an acceptable 72.70% Specificity.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: MODEL COMPARISON
# ==============================================================================
elif selected_tab == "2. Model Comparison":
    st.markdown("<div class='main-header'>Supervised Model Evaluation & Benchmarks</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>DA3 Criterion 1 — Performance Comparison across 10 Supervised Algorithms & Tuned Ensembles</div>", unsafe_allow_html=True)
    
    view_scope = st.radio(
        "Select Benchmark Scope:",
        ["Baseline Models (10 Algorithms)", "Tuned Tree Ensembles (Stage 8)"],
        horizontal=True
    )
    
    if view_scope == "Baseline Models (10 Algorithms)":
        df_models = load_baseline_models()
        title_suffix = "10 Baseline Algorithms (Default 0.50 Threshold)"
    else:
        df_models = load_tuned_models()
        title_suffix = "Tuned Tree Ensembles"
        
    metric_choice = st.selectbox(
        "Rank Models By Metric:",
        ["ROC_AUC", "Recall", "F1_Score", "Accuracy", "Precision"],
        index=0
    )
    
    # Clean display name
    df_plot = df_models.copy()
    df_plot["DisplayName"] = df_plot["Model"].str.replace("_", " ")
    df_plot = df_plot.dropna(subset=[metric_choice])
    df_plot = df_plot.sort_values(by=metric_choice, ascending=True)
    
    # Plotly horizontal bar chart
    fig_comp = px.bar(
        df_plot,
        x=metric_choice,
        y="DisplayName",
        orientation="h",
        text=df_plot[metric_choice].apply(lambda x: f"{x*100:.2f}%" if metric_choice in ["Accuracy", "Precision", "Recall"] else f"{x:.4f}"),
        color=metric_choice,
        color_continuous_scale="Blues",
        title=f"Comparative {metric_choice} across {title_suffix}"
    )
    fig_comp.update_layout(yaxis_title="", height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.subheader("Standardized Performance Evaluation Matrix (Holdout Test Set)")
    df_display = df_models.copy()
    for col in ["Accuracy", "Precision", "Recall"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
    for col in ["F1_Score", "ROC_AUC"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
            
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div style="background-color: #F0F9FF; border-left: 5px solid #0284C7; border: 1px solid #BAE6FD; padding: 16px; border-radius: 6px; margin-top: 15px;">
        <div style="font-weight: 700; font-size: 1.05rem; color: #0369A1; margin-bottom: 5px;">Analytical Discussion (The Accuracy Paradox):</div>
        <div style="color: #1E293B; font-size: 0.95rem; line-height: 1.5;">
            Baseline algorithms like LDA (83.47% accuracy) and Logistic Regression (83.45% accuracy) appear strong on paper but achieve less than 3% sensitivity. This occurs because the loss function optimizes global accuracy over rare-class detection. XGBoost was chosen as the champion architecture because it demonstrated the highest discriminative stability (ROC-AUC: 0.6716) and proved uniquely responsive to probability threshold calibration.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 3: THRESHOLD ANALYSIS
# ==============================================================================
elif selected_tab == "3. Threshold Analysis":
    st.markdown("<div class='main-header'>Dynamic Threshold Analysis & Calibration</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>DA3 Criteria 1 & 3 — Calibrating Decision Boundaries on the 9,999 Holdout Test Records</div>", unsafe_allow_html=True)
    
    # Preset Buttons
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)
    if "thresh_val" not in st.session_state:
        st.session_state.thresh_val = 0.20
        
    with c_btn1:
        if st.button("Optimal DA3 Policy (0.20)", use_container_width=True):
            st.session_state.thresh_val = 0.20
    with c_btn2:
        if st.button("Balanced F1 Policy (0.25)", use_container_width=True):
            st.session_state.thresh_val = 0.25
    with c_btn3:
        if st.button("Default Uncalibrated (0.50)", use_container_width=True):
            st.session_state.thresh_val = 0.50
    with c_btn4:
        if st.button("Conservative Policy (0.70)", use_container_width=True):
            st.session_state.thresh_val = 0.70
            
    th_slider = st.slider(
        "Classification Decision Threshold (T):",
        min_value=0.10,
        max_value=0.90,
        value=float(st.session_state.thresh_val),
        step=0.01,
        help="Employees with model-predicted probability >= T are flagged for proactive retention."
    )
    st.session_state.thresh_val = th_slider
    
    # Compute live metrics
    m = compute_metrics_at_threshold(th_slider)
    
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Sensitivity / Recall", f"{m['Recall']*100:.2f}%", help="True Leavers Detected / Total Actual Leavers")
    with k2:
        st.metric("Precision / PPV", f"{m['Precision']*100:.2f}%", help="True Leavers / Total Flagged Cases")
    with k3:
        st.metric("F1-Score", f"{m['F1_Score']:.4f}", help="Harmonic mean of precision and recall")
    with k4:
        st.metric("Overall Accuracy", f"{m['Accuracy']*100:.2f}%", help="Total Correct Classifications / 9,999")
    with k5:
        st.metric("Specificity", f"{m['Specificity']*100:.2f}%", help="True Non-Leavers Correctly Filtered Out")
        
    st.markdown("---")
    
    # Live Confusion Matrix Display
    col_mat, col_plot = st.columns([5, 7])
    
    with col_mat:
        st.subheader("Live Confusion Matrix (9,999 Test Set)")
        cm_html = f"""
        <table style="width:100%; border-collapse: collapse; text-align: center; font-size: 1rem; border: 1px solid #CBD5E1;">
            <tr style="border-bottom: 2px solid #94A3B8;">
                <th style="padding: 10px; background-color: #F8FAFC; color: #0F172A;"></th>
                <th style="padding: 10px; background-color: #F1F5F9; color: #0F172A; font-weight: 700;">Predicted NO</th>
                <th style="padding: 10px; background-color: #F1F5F9; color: #0F172A; font-weight: 700;">Predicted YES</th>
            </tr>
            <tr style="border-bottom: 1px solid #CBD5E1;">
                <th style="padding: 14px; background-color: #F8FAFC; color: #0F172A; text-align: left; font-weight: 700;">Actual NO (8,333)</th>
                <td style="background-color: #F1F5F9; padding: 14px; font-weight: 800; color: #0F172A;">
                    TN = {m['TN']:,}<br><span style='font-weight: 500; font-size: 0.85rem; color: #475569;'>Correct Retained</span>
                </td>
                <td style="background-color: #FEF08A; padding: 14px; font-weight: 800; color: #854D0E;">
                    FP = {m['FP']:,}<br><span style='font-weight: 500; font-size: 0.85rem; color: #854D0E;'>False Alarms</span>
                </td>
            </tr>
            <tr>
                <th style="padding: 14px; background-color: #F8FAFC; color: #0F172A; text-align: left; font-weight: 700;">Actual YES (1,666)</th>
                <td style="background-color: #FECACA; padding: 14px; font-weight: 800; color: #991B1B;">
                    FN = {m['FN']:,}<br><span style='font-weight: 500; font-size: 0.85rem; color: #991B1B;'>Missed Leavers</span>
                </td>
                <td style="background-color: #BBF7D0; padding: 14px; font-weight: 800; color: #166534;">
                    TP = {m['TP']:,}<br><span style='font-weight: 500; font-size: 0.85rem; color: #166534;'>Caught Leavers</span>
                </td>
            </tr>
        </table>
        """
        st.markdown(cm_html, unsafe_allow_html=True)
        st.caption(f"Evaluated at Decision Threshold T = {th_slider:.2f}")
        
    with col_plot:
        st.subheader("Performance Trajectories Across Thresholds")
        th_range = np.arange(0.10, 0.91, 0.02)
        curve_data = []
        for t_val in th_range:
            res_t = compute_metrics_at_threshold(t_val)
            curve_data.append(res_t)
        df_curves = pd.DataFrame(curve_data)
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=df_curves["threshold"], y=df_curves["Recall"], mode="lines", name="Recall (Sensitivity)", line=dict(color="#16A34A", width=3)))
        fig_curve.add_trace(go.Scatter(x=df_curves["threshold"], y=df_curves["Precision"], mode="lines", name="Precision", line=dict(color="#0284C7", width=2.5)))
        fig_curve.add_trace(go.Scatter(x=df_curves["threshold"], y=df_curves["F1_Score"], mode="lines", name="F1-Score", line=dict(color="#1B365D", width=2.5)))
        fig_curve.add_trace(go.Scatter(x=df_curves["threshold"], y=df_curves["Specificity"], mode="lines", name="Specificity", line=dict(color="#64748B", width=1.5, dash="dot")))
        
        fig_curve.add_vline(x=th_slider, line_width=2, line_dash="dash", line_color="#DC2626", annotation_text=f"T={th_slider:.2f}", annotation_position="top right")
        
        fig_curve.update_layout(
            xaxis_title="Classification Threshold",
            yaxis_title="Score (0 to 1)",
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

# ==============================================================================
# TAB 4: WORKFORCE ANALYTICS
# ==============================================================================
elif selected_tab == "4. Workforce Analytics":
    st.markdown("<div class='main-header'>Workforce Analytics & Demographic Segmentation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>DA3 Criterion 2 — Multi-Variable Population Exploration across 50,000 Production Records</div>", unsafe_allow_html=True)
    
    df_wf = load_workforce_data()
    if df_wf.empty:
        st.warning("Loading dataset from SQLite...")
        st.stop()
        
    # Filters
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        depts = ["All"] + sorted(df_wf["Department"].dropna().unique().tolist())
        sel_dept = st.selectbox("Department:", depts)
    with f2:
        plants = ["All"] + sorted(df_wf["Plant"].dropna().unique().tolist())
        sel_plant = st.selectbox("Manufacturing Plant:", plants)
    with f3:
        roles = ["All"] + sorted(df_wf["JobRole"].dropna().unique().tolist())
        sel_role = st.selectbox("Job Role:", roles)
    with f4:
        max_ot = int(df_wf["MonthlyOvertimeHours"].max())
        ot_range = st.slider("Monthly Overtime Hours:", 0, max_ot, (0, max_ot), step=5)
        
    # Apply filtering
    df_filtered = df_wf.copy()
    if sel_dept != "All":
        df_filtered = df_filtered[df_filtered["Department"] == sel_dept]
    if sel_plant != "All":
        df_filtered = df_filtered[df_filtered["Plant"] == sel_plant]
    if sel_role != "All":
        df_filtered = df_filtered[df_filtered["JobRole"] == sel_role]
    df_filtered = df_filtered[(df_filtered["MonthlyOvertimeHours"] >= ot_range[0]) & 
                             (df_filtered["MonthlyOvertimeHours"] <= ot_range[1])]
    
    # Filtered Metrics
    f_cnt = len(df_filtered)
    f_att = int((df_filtered["AttritionFlag"] == "Yes").sum())
    f_rate = (f_att / f_cnt * 100) if f_cnt > 0 else 0.0
    f_ot = df_filtered["MonthlyOvertimeHours"].mean() if f_cnt > 0 else 0.0
    f_eng = df_filtered["EngagementScore"].mean() if f_cnt > 0 else 0.0
    f_dist = df_filtered["DistanceFromHome"].mean() if f_cnt > 0 else 0.0
    
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Headcount", f"{f_cnt:,}")
    k2.metric("Attrition Count", f"{f_att:,}")
    k3.metric("Attrition Rate", f"{f_rate:.2f}%")
    k4.metric("Avg Overtime", f"{f_ot:.1f} hrs")
    k5.metric("Avg Engagement", f"{f_eng:.1f} / 95")
    k6.metric("Avg Distance", f"{f_dist:.1f} km")
    
    st.markdown("---")
    
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.subheader("Attrition Rate by Department")
        dept_grp = df_filtered.groupby("Department")["AttritionFlag"].apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="Rate")
        fig_d = px.bar(dept_grp, x="Department", y="Rate", text=dept_grp["Rate"].apply(lambda x: f"{x:.1f}%"), color="Rate", color_continuous_scale="Reds")
        fig_d.update_layout(coloraxis_showscale=False, yaxis_title="Attrition Rate (%)", height=320)
        st.plotly_chart(fig_d, use_container_width=True)
        
    with c_p2:
        st.subheader("Attrition Rate by Manufacturing Plant")
        plant_grp = df_filtered.groupby("Plant")["AttritionFlag"].apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="Rate")
        fig_p = px.bar(plant_grp, x="Plant", y="Rate", text=plant_grp["Rate"].apply(lambda x: f"{x:.1f}%"), color="Rate", color_continuous_scale="Blues")
        fig_p.update_layout(coloraxis_showscale=False, yaxis_title="Attrition Rate (%)", height=320)
        st.plotly_chart(fig_p, use_container_width=True)
        
    c_p3, c_p4 = st.columns(2)
    with c_p3:
        st.subheader("Engagement Score Distribution by Turnover Status")
        fig_box = px.box(df_filtered, x="AttritionFlag", y="EngagementScore", color="AttritionFlag",
                         color_discrete_map={"No": "#16A34A", "Yes": "#DC2626"},
                         category_orders={"AttritionFlag": ["No", "Yes"]})
        fig_box.update_layout(height=300, showlegend=False, xaxis_title="Attrition Status", yaxis_title="Engagement Score (30-95)")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with c_p4:
        st.subheader("Monthly Overtime Distribution by Turnover Status")
        fig_hist = px.histogram(df_filtered, x="MonthlyOvertimeHours", color="AttritionFlag", barmode="overlay",
                                opacity=0.7, color_discrete_map={"No": "#16A34A", "Yes": "#DC2626"})
        fig_hist.update_layout(height=300, xaxis_title="Monthly Overtime Hours", yaxis_title="Employee Count")
        st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# TAB 5: EMPLOYEE RISK CALCULATOR
# ==============================================================================
elif selected_tab == "5. Employee Risk Calculator":
    st.markdown("<div class='main-header'>Real-Time Employee Risk Scoring Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>DA3 Criteria 2 & 3 — Live Inference Bridge via Trained R Tuned XGBoost Model Pipeline</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #F0F9FF; border-left: 5px solid #0284C7; border: 1px solid #BAE6FD; padding: 16px; border-radius: 6px; margin-bottom: 15px;">
        <div style="font-weight: 700; font-size: 1.05rem; color: #0369A1; margin-bottom: 4px;">Mandatory Academic & Ethics Disclaimer:</div>
        <div style="color: #1E293B; font-size: 0.92rem; line-height: 1.5;">
            <i>This model is a decision-support tool and should not be used as the sole basis for employment decisions.</i><br>
            The three demonstration buttons below represent <b>illustrative demonstration scenarios</b> designed to evaluate model behavior across risk archetypes. They do not represent real employees, and the predicted probability reflects an algorithmic score rather than an absolute real-world probability.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Preset scenarios
    s_col1, s_col2, s_col3 = st.columns(3)
    if "calc_profile" not in st.session_state:
        st.session_state.calc_profile = {
            "Department": "Logistics",
            "Plant": "4W Lighting Manesar",
            "EducationField": "Life Sciences",
            "JobRole": "Technician",
            "MonthlyIncome": 31000,
            "MonthlyRate": 32000,
            "DailyRate": 1250,
            "HourlyRate": 125,
            "DistanceFromHome": 25,
            "MonthlyOvertimeHours": 25,
            "YearsAtCompany": 4.0,
            "EngagementScore": 68,
            "Education": 2,
            "JobInvolvement": 3,
            "EnvironmentSatisfaction": 2,
            "JobSatisfaction": 2
        }
        
    with s_col1:
        if st.button("Scenario A: High Risk Flight Hazard", use_container_width=True):
            st.session_state.calc_profile = {
                "Department": "Assembly B", "Plant": "Densoten Bawal (Car Infotainment)",
                "EducationField": "Mechanical", "JobRole": "Operator",
                "MonthlyIncome": 19000, "MonthlyRate": 18000, "DailyRate": 750, "HourlyRate": 75,
                "DistanceFromHome": 55, "MonthlyOvertimeHours": 60, "YearsAtCompany": 1.0,
                "EngagementScore": 35, "Education": 2, "JobInvolvement": 1,
                "EnvironmentSatisfaction": 1, "JobSatisfaction": 1
            }
    with s_col2:
        if st.button("Scenario B: Low Risk Retained", use_container_width=True):
            st.session_state.calc_profile = {
                "Department": "Quality Control", "Plant": "4W Lighting Pune",
                "EducationField": "Electronics", "JobRole": "Engineer",
                "MonthlyIncome": 42000, "MonthlyRate": 45000, "DailyRate": 1600, "HourlyRate": 160,
                "DistanceFromHome": 5, "MonthlyOvertimeHours": 5, "YearsAtCompany": 10.0,
                "EngagementScore": 88, "Education": 4, "JobInvolvement": 4,
                "EnvironmentSatisfaction": 4, "JobSatisfaction": 4
            }
    with s_col3:
        if st.button("Scenario C: Borderline Employee", use_container_width=True):
            st.session_state.calc_profile = {
                "Department": "Logistics", "Plant": "4W Lighting Manesar",
                "EducationField": "Life Sciences", "JobRole": "Technician",
                "MonthlyIncome": 31000, "MonthlyRate": 32000, "DailyRate": 1250, "HourlyRate": 125,
                "DistanceFromHome": 25, "MonthlyOvertimeHours": 25, "YearsAtCompany": 4.0,
                "EngagementScore": 68, "Education": 2, "JobInvolvement": 3,
                "EnvironmentSatisfaction": 2, "JobSatisfaction": 2
            }
            
    prof = st.session_state.calc_profile
    
    col_inputs, col_results = st.columns([7, 5])
    
    with col_inputs:
        st.subheader("1. Organizational & Demographic Parameters")
        c1, c2 = st.columns(2)
        with c1:
            depts_all = ["Assembly A", "Assembly B", "Logistics", "Maintenance", "Packaging", "Quality Control", "Tool Room"]
            dept = st.selectbox("Department:", depts_all, index=depts_all.index(prof["Department"]) if prof["Department"] in depts_all else 0)
            plants_all = ["2W Lighting Pune", "4W Lighting Manesar", "4W Lighting Pune", "AW4W Bawal (Alloy Wheels)", "Densoten Bawal (Car Infotainment)", "Switch Manesar"]
            plant = st.selectbox("Plant:", plants_all, index=plants_all.index(prof["Plant"]) if prof["Plant"] in plants_all else 0)
            roles_all = ["Engineer", "Inspector", "Operator", "Supervisor", "Technician"]
            role = st.selectbox("Job Role:", roles_all, index=roles_all.index(prof["JobRole"]) if prof["JobRole"] in roles_all else 0)
            tenure = st.slider("Years at Company:", 0.0, 25.0, float(prof["YearsAtCompany"]), step=0.5)
        with c2:
            income = st.slider("Monthly Income (INR):", 18000, 45000, int(prof["MonthlyIncome"]), step=500)
            mrate = st.slider("Monthly Billing Rate:", 15000, 50000, int(prof["MonthlyRate"]), step=500)
            drate = st.slider("Daily Rate:", 700, 1800, int(prof["DailyRate"]), step=25)
            hrate = st.slider("Hourly Rate:", 70, 180, int(prof["HourlyRate"]), step=5)
            
        st.subheader("2. Workplace Experience & Commute")
        c3, c4 = st.columns(2)
        with c3:
            ot = st.slider("Monthly Overtime Hours:", 0, 75, int(prof["MonthlyOvertimeHours"]), step=1)
            dist = st.slider("Distance From Home (km):", 0, 90, int(prof["DistanceFromHome"]), step=1)
        with c4:
            fields_all = ["Electrical", "Electronics", "Industrial", "Life Sciences", "Mechanical", "Other"]
            field = st.selectbox("Education Field:", fields_all, index=fields_all.index(prof["EducationField"]) if prof["EducationField"] in fields_all else 0)
            edu = st.slider("Education Level (1 to 5):", 1, 5, int(prof["Education"]))
            
        st.subheader("3. Psychometric & Wellbeing Ratings")
        c5, c6 = st.columns(2)
        with c5:
            eng = st.slider("Engagement Score (30 - 95):", 30, 95, int(prof["EngagementScore"]))
            inv = st.slider("Job Involvement (1 - 4):", 1, 4, int(prof["JobInvolvement"]))
        with c6:
            env_sat = st.slider("Environment Satisfaction (1 - 4):", 1, 4, int(prof["EnvironmentSatisfaction"]))
            job_sat = st.slider("Job Satisfaction (1 - 4):", 1, 4, int(prof["JobSatisfaction"]))
            
        # Dynamically calculated satisfaction index
        sat_idx = (env_sat + job_sat + inv) / 3.0
        st.info(f"**Computed SatisfactionIndex:** `{sat_idx:.2f} / 4.00` (Mean of Environment, Job Satisfaction & Involvement)")
        
        btn_score = st.button("Run Real-Time Model Inference", type="primary", use_container_width=True)
        
    with col_results:
        st.subheader("Model Risk Assessment")
        
        calc_threshold = st.slider(
            "Evaluation Operating Threshold:",
            0.10, 0.50, 0.20, step=0.05,
            key="calc_th_slider",
            help="Decision boundary for intervention classification"
        )
        
        # Package input dictionary
        emp_input = {
            "Department": dept,
            "Plant": plant,
            "EducationField": field,
            "JobRole": role,
            "MonthlyIncome": income,
            "MonthlyRate": mrate,
            "DailyRate": drate,
            "HourlyRate": hrate,
            "DistanceFromHome": dist,
            "MonthlyOvertimeHours": ot,
            "YearsAtCompany": tenure,
            "EngagementScore": eng,
            "Education": edu,
            "JobInvolvement": inv,
            "EnvironmentSatisfaction": env_sat,
            "JobSatisfaction": job_sat
        }
        
        # Run prediction bridge
        with st.spinner("Scoring employee vector through R prediction bridge..."):
            pred_res = predict_single_employee(emp_input)
            
        if "error" in pred_res:
            st.error(pred_res["error"])
        else:
            prob = pred_res["probability"]
            st.markdown(f"""
            <div style="text-align:center; padding: 25px; background-color: #F8FAFC; border-radius: 8px; border: 1px solid #CBD5E1;">
                <div style="font-size: 0.9rem; font-weight: 700; color: #475569;">ESTIMATED ATTRITION PROBABILITY</div>
                <div style="font-size: 3.5rem; font-weight: 800; color: #1B365D; margin: 10px 0;">{prob*100:.1f}%</div>
                <div style="font-size: 0.88rem; color: #475569;">Operating Decision Threshold: <b style="color: #0F172A;">{calc_threshold*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Classification
            if prob >= calc_threshold:
                st.markdown(f"""
                <div style="background-color: #FEF2F2; border-left: 5px solid #DC2626; border: 1px solid #FECACA; padding: 14px; border-radius: 6px; margin-top: 15px;">
                    <div style="font-weight: 700; font-size: 1rem; color: #991B1B; margin-bottom: 3px;">Decision: HIGHER RISK (INTERVENTION RECOMMENDED)</div>
                    <div style="color: #1E293B; font-size: 0.92rem;">Probability ({prob*100:.1f}%) meets or exceeds the calibrated action threshold ({calc_threshold*100:.1f}%). Flagged for retention review within 14 business days.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #F0FDF4; border-left: 5px solid #16A34A; border: 1px solid #DCFCE7; padding: 14px; border-radius: 6px; margin-top: 15px;">
                    <div style="font-weight: 700; font-size: 1rem; color: #166534; margin-bottom: 3px;">Decision: LOWER RISK (RETAINED)</div>
                    <div style="color: #1E293B; font-size: 0.92rem;">Probability ({prob*100:.1f}%) is below the action threshold ({calc_threshold*100:.1f}%). Continue routine talent development cadence.</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("#### **Prescriptive Action Recommendations**")
            recommendations = generate_hr_recommendations(emp_input, prob, calc_threshold)
            for r in recommendations:
                box_bg = "#F0FDF4" if r["type"] == "success" else ("#FEF2F2" if r["type"] == "danger" else ("#FEFCE8" if r["type"] == "warning" else "#F0F9FF"))
                box_border = "#16A34A" if r["type"] == "success" else ("#DC2626" if r["type"] == "danger" else ("#EAB308" if r["type"] == "warning" else "#0284C7"))
                title_color = "#166534" if r["type"] == "success" else ("#991B1B" if r["type"] == "danger" else ("#854D0E" if r["type"] == "warning" else "#0369A1"))
                st.markdown(f"""
                <div style="background-color: {box_bg}; border-left: 4px solid {box_border}; padding: 10px 14px; border-radius: 4px; margin-bottom: 10px; border: 1px solid #E2E8F0;">
                    <div style="font-weight: 700; font-size: 0.92rem; color: {title_color}; margin-bottom: 2px;">• {r['title']}</div>
                    <div style="color: #1E293B; font-size: 0.88rem; line-height: 1.4;">{r['detail']}</div>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# TAB 6: FEATURE IMPORTANCE
# ==============================================================================
elif selected_tab == "6. Feature Importance":
    st.markdown("<div class='main-header'>XGBoost Ensemble Feature Importance</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>DA3 Criterion 3 — Tree Split Contributions across Information Gain, Cover, and Frequency</div>", unsafe_allow_html=True)
    
    df_feat = load_feature_importance()
    if df_feat.empty:
        st.warning("Feature importance CSV not found.")
        st.stop()
        
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        metric_dim = st.selectbox("Importance Metric:", ["Gain (Predictive Power)", "Cover (Observation Weight)", "Frequency (Split Count)"])
    with f_col2:
        top_n = st.selectbox("Features to Display:", [10, 15, 25, 31], index=1)
        
    metric_key = "Gain" if "Gain" in metric_dim else ("Cover" if "Cover" in metric_dim else "Frequency")
    
    df_sorted = df_feat.sort_values(by=metric_key, ascending=True).tail(top_n)
    df_sorted["DisplayPct"] = df_sorted[metric_key].apply(lambda x: f"{x*100:.2f}%")
    
    fig_feat = px.bar(
        df_sorted,
        x=metric_key,
        y="Feature",
        orientation="h",
        text="DisplayPct",
        color=metric_key,
        color_continuous_scale="Blues",
        title=f"Top {top_n} Features Ranked by {metric_key}"
    )
    fig_feat.update_layout(height=450, coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig_feat, use_container_width=True)
    
    st.subheader("Domain Analysis: The Primary Drivers of Manufacturing Attrition")
    st.markdown("""
    - **EngagementScore (35.71% Information Gain):** Overwhelmingly the single strongest predictor. Disengagement is the root cognitive state that precedes physical turnover.
    - **SatisfactionIndex (9.26% Gain):** Synthesizes physical environment, job satisfaction, and involvement. Reflects day-to-day manufacturing climate.
    - **MonthlyOvertimeHours (8.93% Gain):** Direct proxy for physical fatigue, shift exhaustion, and work-life balance deterioration.
    - **DistanceFromHome (8.21% Gain):** Long commute corridors across industrial belts compound daily fatigue, precipitating voluntary resignation.
    """)

# ==============================================================================
# TAB 7: CONFUSION MATRIX & DIAGNOSTICS
# ==============================================================================
elif selected_tab == "7. Confusion Matrix & Diagnostics":
    st.markdown("<div class='main-header'>Diagnostic Matrix & Classification Errors</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>DA3 Criteria 1 & 5 — Evaluation of True Positives, False Positives, and Trade-Off Dynamics</div>", unsafe_allow_html=True)
    
    cm_thresh = st.slider("Diagnostic Threshold (T):", 0.10, 0.90, 0.20, step=0.01)
    diag = compute_metrics_at_threshold(cm_thresh)
    
    c1, c2 = st.columns([5, 7])
    
    with c1:
        st.subheader("Interactive 2x2 Heatmap")
        z_vals = [[diag["TN"], diag["FP"]], [diag["FN"], diag["TP"]]]
        fig_heat = px.imshow(
            z_vals,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["Predicted No", "Predicted Yes"],
            y=["Actual No", "Actual Yes"],
            color_continuous_scale="Blues",
            text_auto=True
        )
        fig_heat.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with c2:
        st.subheader("Classification Error Definitions & Trade-Offs")
        st.markdown(f"""
        - **False Positive (Type I Error, FP = {diag['FP']:,}):**  
          An employee predicted as an attrition risk who would have stayed.  
          *Business Impact:* Moderate financial cost of retention interventions ($3,000 package) expended on stable talent.
        - **False Negative (Type II Error, FN = {diag['FN']:,}):**  
          An employee who intended to leave but was not flagged by the model.  
          *Business Impact:* Severe financial cost ($50,000 replacement penalty) plus production disruption.
        - **Strategic Decision:** In workforce analytics, a False Negative is approximately **16.6x more costly** than a False Positive ($50k vs $3k). Lowering the threshold to 0.20 deliberately trades more False Positives for drastically fewer False Negatives.
        """)
        
    st.subheader("Diagnostic Metrics Table")
    diag_df = pd.DataFrame({
        "Diagnostic Metric": [
            "Sensitivity / Recall (TP / [TP + FN])",
            "Specificity (TN / [TN + FP])",
            "Precision / PPV (TP / [TP + FP])",
            "Negative Predictive Value (TN / [TN + FN])",
            "Overall Accuracy ([TP + TN] / N)",
            "Balanced Accuracy ([Sens + Spec] / 2)",
            "F1-Score (Harmonic Mean)"
        ],
        "Mathematical Formula": [
            f"{diag['TP']} / 1,666",
            f"{diag['TN']} / 8,333",
            f"{diag['TP']} / {diag['TP']+diag['FP']}",
            f"{diag['TN']} / {diag['TN']+diag['FN']}",
            f"{diag['TP']+diag['TN']} / 9,999",
            f"({diag['Recall']:.4f} + {diag['Specificity']:.4f}) / 2",
            "2 * (Prec * Rec) / (Prec + Rec)"
        ],
        "Value": [
            f"{diag['Recall']*100:.2f}%",
            f"{diag['Specificity']*100:.2f}%",
            f"{diag['Precision']*100:.2f}%",
            f"{(diag['TN']/(diag['TN']+diag['FN']))*100:.2f}%",
            f"{diag['Accuracy']*100:.2f}%",
            f"{((diag['Recall']+diag['Specificity'])/2)*100:.2f}%",
            f"{diag['F1_Score']:.4f}"
        ]
    })
    st.dataframe(diag_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 8: BUSINESS INSIGHTS
# ==============================================================================
elif selected_tab == "8. Business Insights":
    st.markdown("<div class='main-header'>Strategic Business Insights & HR Prescriptions</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>DA3 Criteria 3 & 4 — Empirical Findings and Operational Interventions for Executive HR</div>", unsafe_allow_html=True)
    
    st.subheader("1. Empirical Findings from Model Analysis")
    st.markdown("""
    - **Cognitive vs Physical Catalysts:** While `EngagementScore` (35.71% Gain) and `SatisfactionIndex` (9.26% Gain) are the strongest individual predictors, the physical strain triad of `MonthlyOvertimeHours` (8.93%) and `DistanceFromHome` (8.21%) accounts for over 17% of total tree splits.
    - **Threshold Sensitivity:** Lowering the operational decision threshold from 0.50 to 0.20 fundamentally alters operational utility—elevating voluntary leaver detection from **1.38% to 51.62%** (+837 employees caught in the holdout test set alone).
    - **False Positive Governance:** Increased false positives (2,275 at threshold 0.20) are acceptable because the $3,000 proactive retention intervention cost is minor relative to the $50,000 turnover penalty.
    """)
    
    st.subheader("2. Operational Recommendations for Plant Leadership")
    c_rec1, c_rec2 = st.columns(2)
    with c_rec1:
        st.markdown("""
        <div style="background-color: #F0FDF4; border-left: 5px solid #16A34A; border: 1px solid #DCFCE7; padding: 16px; border-radius: 6px; margin-bottom: 14px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #166534; margin-bottom: 4px;">Pillar 1: Overtime Capping Governance</div>
            <div style="color: #1E293B; font-size: 0.95rem; line-height: 1.5;">Empirical data demonstrates an inflection point at 28 overtime hours/month. Plant managers in Assembly B and Tool Room should mandate hard monthly ceilings and rotate auxiliary workloads to part-time or relief pools.</div>
        </div>
        <div style="background-color: #F0FDF4; border-left: 5px solid #16A34A; border: 1px solid #DCFCE7; padding: 16px; border-radius: 6px; margin-bottom: 14px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #166534; margin-bottom: 4px;">Pillar 2: Industrial Transit Routing</div>
            <div style="color: #1E293B; font-size: 0.95rem; line-height: 1.5;">Commuters traveling >30 km exhibit 2.4x higher turnover propensity. Subsidized company transit routes connecting suburban hubs to Bawal and Pune facilities will mitigate fatigue-driven churn.</div>
        </div>
        """, unsafe_allow_html=True)
    with c_rec2:
        st.markdown("""
        <div style="background-color: #F0FDF4; border-left: 5px solid #16A34A; border: 1px solid #DCFCE7; padding: 16px; border-radius: 6px; margin-bottom: 14px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #166534; margin-bottom: 4px;">Pillar 3: Front-Line Supervisory Coaching</div>
            <div style="color: #1E293B; font-size: 0.95rem; line-height: 1.5;">Disengagement is often a symptom of shop-floor supervisory friction. Introducing monthly 1-on-1 development reviews will elevate engagement scores above the critical 60.0 threshold.</div>
        </div>
        <div style="background-color: #F0FDF4; border-left: 5px solid #16A34A; border: 1px solid #DCFCE7; padding: 16px; border-radius: 6px; margin-bottom: 14px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #166534; margin-bottom: 4px;">Pillar 4: Decision-Support Integration</div>
            <div style="color: #1E293B; font-size: 0.95rem; line-height: 1.5;">Integrate this predictive dashboard into bi-monthly HR operational reviews to intervene proactively before employees formally submit resignation notices.</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style="background-color: #FEFCE8; border-left: 5px solid #EAB308; border: 1px solid #FEF08A; padding: 16px; border-radius: 6px; margin-top: 15px;">
        <div style="font-weight: 700; font-size: 1.05rem; color: #854D0E; margin-bottom: 4px;">Ethical & Governance Directive:</div>
        <div style="color: #1E293B; font-size: 0.95rem; line-height: 1.5;">Machine learning predictions must remain advisory. Algorithmic scores should trigger supportive conversations, workload rebalancing, and career development rather than punitive or automated employment actions.</div>
    </div>
    """, unsafe_allow_html=True)
