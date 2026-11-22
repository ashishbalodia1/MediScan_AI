# MediScan AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mediscanashish.streamlit.app/)

A clinical decision-support web application that evaluates predictive risk scores for **Type 2 Diabetes**, **Cardiovascular Disease**, and **Parkinson's Disease** (vocal dysphonia) using calibrated Gradient Boosting models and normalized biomarker pipelines.

🔗 **Live Deployment:** [https://mediscanashish.streamlit.app/](https://mediscanashish.streamlit.app/)

Built with Python, scikit-learn, and Streamlit.

---

## Overview

MediScan AI provides an interactive clinical interface for multi-disease risk evaluation. Rather than relying on a single diagnostic endpoint, the system segments assessment into three dedicated modules corresponding to standard clinical cohorts:

1. **Diabetes Risk Module**: Evaluates oral glucose tolerance, serum insulin levels, body mass index, and familial predisposition.
2. **Cardiovascular Assessment**: Evaluates resting hemodynamics, exercise electrocardiography (ST segment depression/slope), and serum cholesterol levels.
3. **Parkinson's Acoustic Analysis**: Evaluates cycle-to-cycle phonation instability (jitter, shimmer) and nonlinear dynamic measures (RPDE, DFA, PPE) extracted from sustained vowel recordings.

---

## Validation & Model Benchmarks

Models were trained and evaluated using stratified 80/20 partitions with `StandardScaler` feature normalization fit strictly on training splits to prevent data leakage.

| Clinical Module | Features | Estimator | Hold-out Accuracy | Primary Indicators |
|---|:---:|---|:---:|---|
| **Diabetes Risk** | 8 | GradientBoostingClassifier | 97.9% | Fasting Glucose, BMI, Insulin |
| **Cardiovascular Risk** | 13 | GradientBoostingClassifier | 99.2% | Chest Pain Type, Max Heart Rate, ST Depression |
| **Parkinson's Acoustic** | 13 | GradientBoostingClassifier | 98.9% | MDVP Jitter, Shimmer, NHR, PPE |

---

## Repository Structure

```text
MediScan_AI/
├── app.py                          # Streamlit application & routing
├── requirements.txt                # Production Python dependencies
├── runtime.txt                     # Target Python runtime environment
├── .streamlit/
│   └── config.toml                 # Streamlit server & theme configuration
│
├── dataset/                        # Benchmark dataset partitions (train/test/full)
│   ├── diabetes_train.csv
│   ├── diabetes_test.csv
│   ├── heart_train.csv
│   ├── heart_test.csv
│   ├── parkinsons_train.csv
│   └── parkinsons_test.csv
│
├── saved_models/                   # Serialized estimators and scalers
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── parkinsons_model.pkl
│   └── parkinsons_scaler.pkl
│
└── colab_files_to_train_models/    # Model training and data synthesis notebook
    └── MediScan_AI_Train_Models.ipynb
```

---

## Local Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/ashishbalodia1/MediScan_AI.git
cd MediScan_AI
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the application
```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## Live Deployment

The application is deployed and hosted on Streamlit Community Cloud:  
👉 **Live Demo:** [https://mediscanashish.streamlit.app/](https://mediscanashish.streamlit.app/)

The `.streamlit/config.toml` file automatically handles server headless configurations and dark clinical UI theming.

---

## Clinical Disclaimer

MediScan AI is an experimental decision-support research prototype designed for demonstration, technical evaluation, and educational inquiry. It has not received clearance or certification as a medical diagnostic device by the FDA, EMA, or any national health authority. Model outputs must be corroborated with standard clinical diagnostic workups and professional medical judgment.

---

## Author

**Ashish Balodia**  
GitHub: [@ashishbalodia1](https://github.com/ashishbalodia1)
