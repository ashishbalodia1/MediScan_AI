"""
MediScan AI - Clinical Disease Risk Assessment System
Predictive risk modeling for Diabetes, Cardiovascular Disease, and Parkinson's.

Author: Ashish Balodia
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Ensure unpickling compatibility across environments
try:
    import sklearn._loss._loss
    sys.modules["_loss"] = sklearn._loss._loss
except (ImportError, AttributeError):
    pass

# --- Page Configuration ---
st.set_page_config(
    page_title="MediScan AI | Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { box-sizing: border-box; }

    .stApp {
        background: #0f1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #9ca3af !important;
        font-size: 14px;
        padding: 9px 14px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.15s ease-in-out;
        border: 1px solid transparent;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: #1f2937;
        color: #f9fafb !important;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f9fafb;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .main-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 28px;
    }
    h1, h2, h3 { color: #f9fafb; font-weight: 600; }
    label { color: #d1d5db !important; font-size: 0.86rem !important; font-weight: 500 !important; }

    .stat-card {
        background: #171c26;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .stat-card:hover { border-color: #4a5568; }
    .stat-number { font-size: 2rem; font-weight: 700; color: #60a5fa; line-height: 1.1; }
    .stat-label { font-size: 0.76rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 6px; }

    .info-card {
        background: #171c26;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .info-card h4 { color: #e5e7eb; font-size: 0.98rem; font-weight: 600; margin-bottom: 8px; }
    .info-card p { color: #9ca3af; font-size: 0.86rem; line-height: 1.55; }

    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e5e7eb;
        padding: 12px 0 6px 0;
        border-bottom: 1px solid #2d3748;
        margin-bottom: 18px;
    }

    .result-positive {
        background: #2b171a;
        border: 1.5px solid #ef4444;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 18px 0;
        text-align: center;
    }
    .result-negative {
        background: #13241b;
        border: 1.5px solid #22c55e;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 18px 0;
        text-align: center;
    }
    .result-title { font-size: 1.45rem; font-weight: 700; margin-bottom: 6px; }
    .result-positive .result-title { color: #f87171; }
    .result-negative .result-title { color: #4ade80; }
    .result-subtitle { color: #9ca3af; font-size: 0.88rem; }

    .confidence-bar-bg {
        background: #1f2937;
        border-radius: 999px;
        height: 7px;
        margin: 14px 0 6px 0;
        overflow: hidden;
    }
    .confidence-bar-fill { height: 100%; border-radius: 999px; }

    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 600; }
    .badge-blue   { background: #1e3a8a40; color: #60a5fa; }
    .badge-green  { background: #14532d40; color: #4ade80; }
    .badge-purple { background: #581c8740; color: #c084fc; }

    .stButton > button {
        background: #2563eb;
        color: #ffffff !important;
        border: 1px solid #1d4ed8;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 0.95rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: background 0.15s ease;
    }
    .stButton > button:hover { background: #1d4ed8; }

    .stTabs [data-baseweb="tab-list"] { background: #171c26; border-radius: 8px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; color: #9ca3af; font-size: 0.88rem; }
    .stTabs [aria-selected="true"] { background: #2563eb !important; color: #ffffff !important; }

    .disclaimer {
        background: #1a1e29;
        border: 1px solid #3b4252;
        border-left: 3px solid #f59e0b;
        border-radius: 6px;
        padding: 12px 14px;
        color: #9ca3af;
        font-size: 0.82rem;
        margin-top: 20px;
    }

    .stNumberInput input, .stSelectbox select {
        background: #171c26 !important;
        border: 1px solid #2d3748 !important;
        color: #f9fafb !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Paths ---
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "saved_models"
DATASET_DIR = BASE_DIR / "dataset"

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "patient_profile": {
            "name": "",
            "age": 35,
            "gender": "Not specified",
            "phone": "",
            "blood_group": "Unknown",
            "medical_history": "",
            "allergies": "",
            "medications": "",
            "previous_reports": "",
            "emergency_contact": "",
        },
        "risk_history": [],
        "uploaded_reports": [],
        "uploaded_report_keys": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# --- Model & Inference Pipeline ---
@st.cache_resource
def load_model(disease: str):
    model = joblib.load(MODELS_DIR / f"{disease}_model.pkl")
    scaler = joblib.load(MODELS_DIR / f"{disease}_scaler.pkl")
    with open(MODELS_DIR / f"{disease}_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return model, scaler, metrics

def run_prediction(disease: str, features: list):
    model, scaler, metrics = load_model(disease)
    feature_names = metrics.get("features", [])
    if feature_names and len(feature_names) == len(features):
        df_input = pd.DataFrame([features], columns=feature_names)
        arr_s = scaler.transform(df_input)
    else:
        arr = np.array(features).reshape(1, -1)
        arr_s = scaler.transform(arr)
    pred = model.predict(arr_s)[0]
    proba = model.predict_proba(arr_s)[0]
    return int(pred), float(proba[1]), metrics

def add_risk_history(disease: str, pred: int, prob: float, vitals: dict):
    risk_percent = prob * 100 if pred == 1 else (1 - prob) * 100
    st.session_state.risk_history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Assessment": disease,
        "Risk Category": "Elevated Risk" if pred == 1 else "Normal Range",
        "Confidence Score (%)": round(risk_percent, 1),
        **vitals,
    })

def render_referral_panel(disease: str, pred: int):
    specialists = {
        "Diabetes": ("Endocrinologist", "HbA1c, Fasting Blood Sugar, Kidney Function Panel"),
        "Heart Disease": ("Cardiologist", "12-Lead ECG, Lipid Profile, Echocardiogram"),
        "Parkinson's": ("Neurologist", "Clinical Motor Evaluation, UPDRS Exam, Speech Acoustic Analysis"),
    }
    specialist, next_tests = specialists.get(disease, ("Physician", "Routine clinical evaluation"))
    urgency = "Clinical consultation advised" if pred == 1 else "Routine monitoring sufficient"

    st.markdown('<div class="section-header">Clinical Follow-up & Recommendations</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(f"Recommended Specialist: {specialist}")
    with c2:
        st.info(f"Evaluation Priority: {urgency}")
    with c3:
        st.info(f"Suggested Diagnostic Tests: {next_tests}")

    with st.expander("Clinical Summary Notes"):
        city = st.text_input("Patient Location / Facility", value="", key=f"{disease}_city")
        symptoms = st.text_area("Patient Reported Symptoms", key=f"{disease}_symptoms")
        st.text_area(
            "Draft Referral Note",
            value=(
                f"Referral to: {specialist}\n"
                f"Preliminary Assessment: {urgency}\n"
                f"Facility/Location: {city or 'Unspecified'}\n"
                f"Reported Symptoms: {symptoms or 'None documented'}"
            ),
            height=110,
            key=f"{disease}_referral_note",
        )

def render_emergency_guidance(disease: str, pred: int):
    if pred != 1:
        return

    guidance = {
        "Diabetes": [
            "Monitor for acute symptoms: severe dehydration, confusion, hyperventilation, or persistent vomiting.",
            "Do not adjust prescription insulin regimens without clinical supervision.",
            "Seek emergency care if blood glucose is critically elevated with ketones present.",
        ],
        "Heart Disease": [
            "Seek immediate emergency care for acute chest pressure, radiation to arm/jaw, or sudden dyspnea.",
            "Do not drive yourself if experiencing sudden severe chest distress.",
            "Have recent ECGs and medication lists readily available for triage staff.",
        ],
        "Parkinson's": [
            "Consult a neurologist for confirmation using standard UPDRS motor scoring.",
            "Seek emergency care if sudden swallowing difficulty or acute severe motor freezing occurs.",
            "Maintain an objective log of motor fluctuations relative to medication schedules.",
        ],
    }

    st.error("Caution: Model identified clinical indicators consistent with elevated risk.")
    for item in guidance.get(disease, []):
        st.warning(item)
    st.info("In acute clinical emergencies, contact local emergency services immediately.")

def analyze_uploaded_report(uploaded_file):
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    size_kb = len(uploaded_file.getvalue()) / 1024
    summary = [f"File: {name} ({size_kb:.1f} KB)"]

    if suffix == ".csv":
        try:
            df = pd.read_csv(uploaded_file)
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            summary.append(f"Structure: {len(df)} rows, {len(df.columns)} columns")
            if numeric_cols:
                stats = df[numeric_cols].describe().T[["mean", "min", "max"]].round(2)
                return summary, stats
        except Exception as exc:
            summary.append(f"CSV parsing error: {exc}")
    elif suffix in [".txt", ".md"]:
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        summary.append(f"Text report parsed ({len(text)} characters).")
        summary.append(f"Excerpt: {text[:160]}...")
    else:
        summary.append(f"{suffix.upper()} file received and indexed for session reference.")

    return summary, None

def show_result(pred: int, prob: float, positive_label: str, negative_label: str,
                positive_sub: str = "", negative_sub: str = ""):
    """Render the assessment result badge."""
    if pred == 1:
        bar_color = "linear-gradient(90deg,#ef4444,#f97316)"
        conf_val = prob * 100
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-title">{positive_label}</div>
            <div class="result-subtitle">{positive_sub}</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{conf_val:.1f}%;background:{bar_color};"></div>
            </div>
            <div style="color:#f87171;font-weight:700;font-size:1.15rem;margin-top:6px;">{conf_val:.1f}% Confidence Score</div>
        </div>""", unsafe_allow_html=True)
    else:
        bar_color = "linear-gradient(90deg,#22c55e,#16a34a)"
        conf_val = (1 - prob) * 100
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-title">{negative_label}</div>
            <div class="result-subtitle">{negative_sub}</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{conf_val:.1f}%;background:{bar_color};"></div>
            </div>
            <div style="color:#4ade80;font-weight:700;font-size:1.15rem;margin-top:6px;">{conf_val:.1f}% Confidence Score</div>
        </div>""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <div style='padding: 12px 0 16px 0;'>
        <div style='font-size: 1.25rem; font-weight: 700; color: #f9fafb;'>MediScan AI</div>
        <div style='color: #6b7280; font-size: 0.8rem; margin-top: 2px;'>Clinical Risk Assessment System</div>
    </div>
    <hr style='border-color: #1f2937; margin: 0 0 16px 0;'>
    <div style='color: #6b7280; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Patient Profile",
            "Report Upload & Analysis",
            "Diabetes Risk",
            "Heart Disease Risk",
            "Parkinson's Risk",
            "Risk History",
            "Model Analytics",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border-color: #1f2937; margin: 24px 0 14px 0;'>
    <div style='color: #6b7280; font-size: 0.74rem; line-height: 1.5;'>
        Investigational decision-support tool. Not approved as a primary diagnostic device.
    </div>
    """, unsafe_allow_html=True)

# --- View: Home ---
if page == "Home":
    st.markdown('<h1 class="main-title">MediScan AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Decision-support system for metabolic, cardiovascular, and vocal acoustic risk evaluation</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("3", "#60a5fa", "Pathologies Modeled"),
        ("98%+", "#a78bfa", "Validation Accuracy"),
        ("3,300", "#34d399", "Benchmark Cohort Records"),
        ("GBM", "#fb923c", "Gradient Boosting Core"),
    ]
    for col, (num, color, label) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color:{color};">{num}</div>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Clinical Modules</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""
        <div class="info-card">
            <h4>Diabetes Risk Assessment</h4>
            <p>Evaluates glucose dysregulation, insulin resistance, BMI, and metabolic markers to predict diabetic likelihood.</p>
            <br><span class="badge badge-blue">8 Features</span>
            <span class="badge badge-green" style="margin-left:6px;">97.9% Val Acc</span>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="info-card">
            <h4>Cardiovascular Disease</h4>
            <p>Assesses coronary risk utilizing resting hemodynamics, exercise ECG depression, and serum lipid metrics.</p>
            <br><span class="badge badge-blue">13 Features</span>
            <span class="badge badge-green" style="margin-left:6px;">Ensemble GBM</span>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div class="info-card">
            <h4>Parkinson's Vocal Biomarkers</h4>
            <p>Analyzes vocal perturbation including jitter, shimmer, harmonic noise ratio, and nonlinear pitch entropy.</p>
            <br><span class="badge badge-blue">13 Features</span>
            <span class="badge badge-purple" style="margin-left:6px;">Acoustic Features</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">System Workflow</div>', unsafe_allow_html=True)

    steps = st.columns(4)
    workflow_steps = [
        ("1. Select Module", "Choose the target disease assessment from the navigation panel."),
        ("2. Enter Vitals", "Input documented patient vitals, laboratory tests, or acoustic metrics."),
        ("3. Pipeline Execution", "Data is scaled with feature-specific standard normalizers and scored."),
        ("4. Clinical Output", "Review probability score, categorized risk band, and suggested next steps."),
    ]
    for col, (title, desc) in zip(steps, workflow_steps):
        with col:
            st.markdown(f"""
            <div class="info-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>Investigational Notice:</strong> MediScan AI is developed for educational and experimental research purposes.
        All algorithmic outputs must be interpreted by licensed healthcare clinicians alongside standard diagnostic workups.
    </div>""", unsafe_allow_html=True)

# --- View: Patient Profile ---
elif page == "Patient Profile":
    st.markdown('<h1 class="main-title">Patient Profile Management</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Manage session-level demographic and baseline health records</p>', unsafe_allow_html=True)

    profile = st.session_state.patient_profile
    with st.form("patient_profile_form"):
        st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            name = st.text_input("Patient Identifier / Name", value=profile["name"])
        with c2:
            age = st.number_input("Age (Years)", 0, 120, int(profile["age"]))
        with c3:
            gender_options = ["Not specified", "Female", "Male", "Other"]
            g_idx = gender_options.index(profile["gender"]) if profile["gender"] in gender_options else 0
            gender = st.selectbox("Biological Sex / Gender", gender_options, index=g_idx)
        with c4:
            bg_options = ["Unknown", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
            bg_idx = bg_options.index(profile["blood_group"]) if profile["blood_group"] in bg_options else 0
            blood_group = st.selectbox("Blood Group", bg_options, index=bg_idx)

        st.markdown('<div class="section-header">Clinical History</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            phone = st.text_input("Contact Number", value=profile["phone"])
            allergies = st.text_area("Documented Allergies", value=profile["allergies"], height=85)
            medical_history = st.text_area("Known Medical Conditions", value=profile["medical_history"], height=110)
        with m2:
            emergency_contact = st.text_input("Emergency Contact", value=profile["emergency_contact"])
            medications = st.text_area("Current Prescriptions", value=profile["medications"], height=85)
            previous_reports = st.text_area("Prior Diagnostic History", value=profile["previous_reports"], height=110)

        saved = st.form_submit_button("Update Session Profile", use_container_width=True)

    if saved:
        st.session_state.patient_profile = {
            "name": name,
            "age": age,
            "gender": gender,
            "phone": phone,
            "blood_group": blood_group,
            "medical_history": medical_history,
            "allergies": allergies,
            "medications": medications,
            "previous_reports": previous_reports,
            "emergency_contact": emergency_contact,
        }
        st.success("Session profile successfully updated.")

    st.markdown('<div class="section-header">Active Session Record</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([st.session_state.patient_profile]), hide_index=True, use_container_width=True)

# --- View: Report Upload ---
elif page == "Report Upload & Analysis":
    st.markdown('<h1 class="main-title">Diagnostic Data Ingestion</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Upload diagnostic tabular data or text summaries for session reference</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Supported formats: CSV, TXT, MD, PDF",
        type=["csv", "txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            report_key = f"{uploaded_file.name}:{uploaded_file.size}"
            summary, stats = analyze_uploaded_report(uploaded_file)
            if report_key not in st.session_state.uploaded_report_keys:
                st.session_state.uploaded_reports.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "File": uploaded_file.name,
                    "Overview": " | ".join(summary),
                })
                st.session_state.uploaded_report_keys.append(report_key)

            with st.expander(uploaded_file.name, expanded=True):
                for line in summary:
                    st.write(line)
                if stats is not None:
                    st.dataframe(stats, use_container_width=True)

    st.markdown('<div class="section-header">Session Documents</div>', unsafe_allow_html=True)
    if st.session_state.uploaded_reports:
        st.dataframe(pd.DataFrame(st.session_state.uploaded_reports), hide_index=True, use_container_width=True)
    else:
        st.info("No documents uploaded in this browser session.")

# --- View: Diabetes ---
elif page == "Diabetes Risk":
    st.markdown('<h1 class="main-title">Diabetes Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Evaluates metabolic indicators against calibrated gradient boosting decision boundaries</p>', unsafe_allow_html=True)

    with st.expander("Quick Demo Presets", expanded=False):
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.caption("Standard Healthy Baseline: Fasting glucose 95, BMI 23.5, Blood Pressure 70")
        with c_p2:
            st.caption("Elevated Risk Profile: Fasting glucose 165, BMI 34.0, Blood Pressure 85")

    with st.form("diabetes_form"):
        st.markdown('<div class="section-header">Demographics & Family History</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1: pregnancies = st.number_input("Pregnancies", 0, 17, 1, help="Total obstetric count")
        with col2: age = st.number_input("Age (Years)", 18, 81, 35)
        with col3: bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 26.0, 0.1)
        with col4: dpf = st.number_input("Diabetes Pedigree Function", 0.05, 2.50, 0.47, 0.001, help="Genetic predisposition index")

        st.markdown('<div class="section-header">Metabolic & Blood Markers</div>', unsafe_allow_html=True)
        col5, col6, col7, col8 = st.columns(4)
        with col5: glucose = st.number_input("Plasma Glucose (mg/dL)", 50, 200, 120, help="2-hr oral glucose tolerance test value")
        with col6: insulin = st.number_input("Serum Insulin (μU/mL)", 0, 846, 80)
        with col7: blood_pressure = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 122, 72)
        with col8: skin_thickness = st.number_input("Triceps Skinfold (mm)", 0, 99, 20)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Diabetes Risk Assessment", use_container_width=True)

    if submitted:
        features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        pred, prob, _ = run_prediction("diabetes", features)
        add_risk_history("Diabetes", pred, prob, {
            "BMI": bmi,
            "Glucose": glucose,
            "Blood Pressure": blood_pressure,
        })
        col_r, col_s = st.columns([2, 1])
        with col_r:
            show_result(
                pred, prob,
                "Elevated Diabetes Risk Identified", "Low Diabetes Risk Profile",
                "Patient parameters cross risk thresholds for glucose intolerance.",
                "Patient metrics fall within expected normal baseline distributions."
            )
        with col_s:
            st.markdown('<div class="info-card"><h4>Input Feature Verification</h4></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Feature": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "DPF", "Age"],
                "Documented Value": features
            }), hide_index=True, use_container_width=True)

        render_referral_panel("Diabetes", pred)
        render_emergency_guidance("Diabetes", pred)

# --- View: Heart Disease ---
elif page == "Heart Disease Risk":
    st.markdown('<h1 class="main-title">Cardiovascular Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Coronary artery disease evaluation based on clinical hemodynamics and stress response</p>', unsafe_allow_html=True)

    with st.form("heart_form"):
        st.markdown('<div class="section-header">Baseline Demographics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: age = st.number_input("Age (Years)", 25, 80, 52)
        with col2: sex = st.selectbox("Biological Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        with col3: cp = st.selectbox("Chest Pain Presentation",
                                     [(0, "Typical Angina"), (1, "Atypical Angina"), (2, "Non-anginal Pain"), (3, "Asymptomatic")],
                                     format_func=lambda x: x[1])

        st.markdown('<div class="section-header">Hemodynamics & Baseline Labs</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", 90, 200, 130)
            chol = st.number_input("Serum Cholesterol (mg/dL)", 120, 600, 246)
        with col5:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
            restecg = st.selectbox("Resting Electrocardiogram",
                                   [(0, "Normal"), (1, "ST-T Wave Abnormality"), (2, "Left Ventricular Hypertrophy")],
                                   format_func=lambda x: x[1])
        with col6:
            thalach = st.number_input("Maximum Achieved Heart Rate", 70, 202, 150)
            exang = st.selectbox("Exercise-Induced Angina", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])

        st.markdown('<div class="section-header">Exercise Stress & Angiography Findings</div>', unsafe_allow_html=True)
        col7, col8, col9, col10 = st.columns(4)
        with col7: oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)
        with col8: slope = st.selectbox("ST Segment Slope", [(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")], format_func=lambda x: x[1])
        with col9: ca = st.selectbox("Major Vessels Colored (Fluoroscopy)", [0, 1, 2, 3, 4])
        with col10: thal = st.selectbox("Thallium Stress Scintigraphy",
                                        [(0, "Normal"), (1, "Fixed Defect"), (2, "Reversible Defect"), (3, "Unknown")],
                                        format_func=lambda x: x[1])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Cardiovascular Assessment", use_container_width=True)

    if submitted:
        features = [age, sex[1], cp[0], trestbps, chol, fbs[0], restecg[0], thalach, exang[0], oldpeak, slope[0], ca, thal[0]]
        pred, prob, _ = run_prediction("heart", features)
        add_risk_history("Heart Disease", pred, prob, {
            "Cholesterol": chol,
            "Blood Pressure": trestbps,
            "Max Heart Rate": thalach,
        })
        show_result(
            pred, prob,
            "Coronary Disease Indicators Detected", "Normal Cardiovascular Profile",
            "Hemodynamic and stress test measures indicate potential ischemic heart disease.",
            "Clinical indicators remain within nominal reference ranges."
        )
        render_referral_panel("Heart Disease", pred)
        render_emergency_guidance("Heart Disease", pred)

# --- View: Parkinson's ---
elif page == "Parkinson's Risk":
    st.markdown('<h1 class="main-title">Parkinson\'s Disease Acoustic Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Phonation frequency instability and nonlinear dynamics for early neurodegenerative risk detection</p>', unsafe_allow_html=True)

    with st.form("parkinsons_form"):
        st.markdown('<div class="section-header">Vocal Fundamental Frequency (Hz)</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: fo = st.number_input("Average Fundamental Frequency (MDVP:Fo)", 80.0, 270.0, 154.228, 0.001)
        with col2: fhi = st.number_input("Maximum Fundamental Frequency (MDVP:Fhi)", 100.0, 600.0, 197.105, 0.001)
        with col3: flo = st.number_input("Minimum Fundamental Frequency (MDVP:Flo)", 65.0, 240.0, 116.082, 0.001)

        st.markdown('<div class="section-header">Frequency & Amplitude Perturbation Measures</div>', unsafe_allow_html=True)
        col4, col5, col6, col7 = st.columns(4)
        with col4: jitter = st.number_input("Cycle-to-Cycle Jitter (%)", 0.001, 0.030, 0.00662, 0.00001, format="%.5f")
        with col5: shimmer = st.number_input("Amplitude Shimmer", 0.009, 0.300, 0.05416, 0.0001, format="%.5f")
        with col6: nhr = st.number_input("Noise-to-Harmonics Ratio (NHR)", 0.001, 0.500, 0.01337, 0.0001, format="%.5f")
        with col7: hnr = st.number_input("Harmonics-to-Noise Ratio (HNR dB)", 8.0, 35.0, 21.640, 0.001)

        st.markdown('<div class="section-header">Nonlinear Dynamic & Fractal Complexity Measures</div>', unsafe_allow_html=True)
        col8, col9, col10, col11, col12, col13 = st.columns(6)
        with col8: rpde = st.number_input("RPDE", 0.20, 0.95, 0.414783, 0.0001, format="%.6f")
        with col9: dfa = st.number_input("DFA", 0.50, 0.97, 0.815285, 0.0001, format="%.6f")
        with col10: spread1 = st.number_input("Spread1", -9.0, -2.0, -4.813031, 0.0001, format="%.6f")
        with col11: spread2 = st.number_input("Spread2", 0.02, 0.55, 0.266482, 0.0001, format="%.6f")
        with col12: d2 = st.number_input("Correlation Dimension (D2)", 1.5, 4.0, 2.301442, 0.0001, format="%.6f")
        with col13: ppe = st.number_input("Pitch Period Entropy (PPE)", 0.02, 0.70, 0.284654, 0.0001, format="%.6f")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Parkinson's Vocal Assessment", use_container_width=True)

    if submitted:
        features = [fo, fhi, flo, jitter, shimmer, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        pred, prob, _ = run_prediction("parkinsons", features)
        add_risk_history("Parkinson's", pred, prob, {
            "Jitter": jitter,
            "Shimmer": shimmer,
            "HNR": hnr,
        })
        show_result(
            pred, prob,
            "Acoustic Indicators of Dysphonia / Parkinson's", "Acoustic Measures Within Expected Norms",
            "Acoustic jitter, shimmer, and pitch entropy indicate phonation impairment.",
            "Vocal stability parameters align with normal control group baselines."
        )
        render_referral_panel("Parkinson's", pred)
        render_emergency_guidance("Parkinson's", pred)

# --- View: Risk History ---
elif page == "Risk History":
    st.markdown('<h1 class="main-title">Session Risk History</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Track assessment scores and vital progression across the active browser session</p>', unsafe_allow_html=True)

    if not st.session_state.risk_history:
        st.info("No assessments recorded in this session. Execute an assessment module to begin logging.")
    else:
        history_df = pd.DataFrame(st.session_state.risk_history)
        st.markdown('<div class="section-header">Assessment Timeline</div>', unsafe_allow_html=True)
        st.dataframe(history_df, hide_index=True, use_container_width=True)

        chart_df = history_df.copy()
        chart_df["Timestamp"] = pd.to_datetime(chart_df["Timestamp"])
        st.markdown('<div class="section-header">Confidence Score Trajectory</div>', unsafe_allow_html=True)
        st.line_chart(chart_df, x="Timestamp", y="Confidence Score (%)", color="Assessment")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Assessments", len(history_df))
        with c2:
            elevated_count = int((history_df["Risk Category"] == "Elevated Risk").sum())
            st.metric("Elevated Risk Results", elevated_count)
        with c3:
            st.metric("Latest Score", f'{history_df.iloc[-1]["Confidence Score (%)"]:.1f}%')

        st.download_button(
            "Export Session History (CSV)",
            history_df.to_csv(index=False).encode("utf-8"),
            "mediscan_session_history.csv",
            "text/csv",
            use_container_width=True,
        )

# --- View: Model Analytics ---
elif page == "Model Analytics":
    st.markdown('<h1 class="main-title">Model Specifications & Benchmark Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Validation metrics, training partitions, and feature schematics</p>', unsafe_allow_html=True)

    modules = {
        "Diabetes Risk Classifier": "diabetes",
        "Cardiovascular Classifier": "heart",
        "Parkinson's Acoustic Classifier": "parkinsons",
    }

    for label, key in modules.items():
        try:
            _, _, metrics = load_model(key)
            train_df = pd.read_csv(DATASET_DIR / f"{key}_train.csv")
            test_df = pd.read_csv(DATASET_DIR / f"{key}_test.csv")

            with st.expander(f"{label} (Hold-out Accuracy: {metrics['accuracy']*100:.1f}%)", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                for col, (num, color, lbl) in zip([c1, c2, c3, c4], [
                    (f"{metrics['accuracy']*100:.1f}%", "#60a5fa", "Hold-out Accuracy"),
                    (metrics["n_samples"], "#a78bfa", "Cohort Samples"),
                    (metrics["n_train"], "#34d399", "Training Split"),
                    (metrics["n_test"], "#fb923c", "Test Split"),
                ]):
                    with col:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number" style="color:{color};">{num}</div>
                            <div class="stat-label">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                t1, t2 = st.tabs(["Training Partition (Preview)", "Validation Partition (Preview)"])
                with t1: st.dataframe(train_df.head(8), use_container_width=True, hide_index=True)
                with t2: st.dataframe(test_df.head(8), use_container_width=True, hide_index=True)

                st.caption(f"Input Feature Vector: {', '.join(metrics['features'])}")
        except Exception as e:
            st.error(f"Error loading metadata for {label}: {e}")

# --- View: About ---
elif page == "About":
    st.markdown('<h1 class="main-title">About MediScan AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">System architecture, modeling methodology, and regulatory disclaimers</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>Algorithm & Modeling Architecture</h4>
            <p>MediScan AI applies <code>GradientBoostingClassifier</code> ensembles implemented in scikit-learn.
            Each estimator trains sequential decision trees optimizing deviance loss functions with shrinkage (learning rate: 0.1)
            and depth constraints (max depth: 5) to balance bias-variance tradeoffs.</p>
        </div>
        <div class="info-card">
            <h4>Data Normalization & Feature Pipeline</h4>
            <p>Continuous clinical markers (e.g. glucose in mg/dL, pressure in mmHg, acoustic frequencies in Hz)
            are standardized with <code>StandardScaler</code> fit strictly on training partitions to prevent data leakage.</p>
        </div>
        <div class="info-card">
            <h4>Session Security & Privacy</h4>
            <p>Inference executes in-memory. Patient demographic details and clinical inputs are maintained exclusively
            within the local Streamlit session state and are not persisted to external databases.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Technical Stack</h4>
            <p>
                <strong>Frontend:</strong> Streamlit<br>
                <strong>Inference & ML:</strong> scikit-learn, joblib<br>
                <strong>Data Processing:</strong> pandas, NumPy<br>
                <strong>Runtime Environment:</strong> Python 3.10+
            </p>
        </div>
        <div class="info-card">
            <h4>Clinical Decision Support Disclaimer</h4>
            <p>This software is a research prototype. It does not possess medical device certification under FDA (510k),
            EU MDR, or equivalent bodies. It is intended strictly for educational, technical demonstration, and investigational workflows.</p>
        </div>
        <div class="info-card">
            <h4>Dataset Acknowledgements</h4>
            <p>Feature distributions inspired by the National Institute of Diabetes and Digestive and Kidney Diseases
            (Pima Indians Diabetes), UCI Cleveland Heart Disease repository, and Oxford University Vocal Impairment studies.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:32px 0 16px;color:#6b7280;font-size:0.8rem;">
        MediScan AI — Developed by Ashish Balodia
    </div>""", unsafe_allow_html=True)
