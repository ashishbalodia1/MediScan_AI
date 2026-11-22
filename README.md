# MediScan AI — Disease Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mediscanashish.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clinical decision-support web application providing multi-pathology predictive risk scoring for **Type 2 Diabetes**, **Cardiovascular Disease**, and **Parkinson's Disease** (vocal phonation dysphonia) using Gradient Boosting Classifiers (`scikit-learn`) and normalized biomarker pipelines.

🔗 **Live Deployment:** [https://mediscanashish.streamlit.app/](https://mediscanashish.streamlit.app/)

> 💡 **Engineering & Provenance Note (HIPAA & Benchmark Context):**  
> In healthcare AI development, using raw patient records introduces strict **Protected Health Information (PHI)** and HIPAA compliance restrictions, preventing open public deployment. The core architectural objective of MediScan AI was to build, validate, and deploy a production-grade **end-to-end clinical decision-support pipeline** (isolated preprocessing, multi-pathology inference, dynamic specialist referral mapping, and emergency clinical triage).  
> To achieve this without PHI compliance issues, mathematical synthetic cohorts modeled strictly on published UCI/Oxford empirical distributions were utilized. Because synthetic distributions feature clean separability, tree-based models achieve **98–100% metrics**. By contrast, on noisy real-world clinical cohorts, established academic benchmarks typically achieve **~76–80% on Pima Diabetes**, **~82–88% on Cleveland Heart**, and **~85–92% on Oxford Parkinson's**.

---

## Table of Contents
- [Overview](#overview)
- [Dataset Provenance & Methodology](#dataset-provenance--methodology)
- [Data Hygiene & Leakage Audit](#data-hygiene--leakage-audit)
- [Comprehensive Model Evaluation](#comprehensive-model-evaluation)
- [Why Are Some Synthetic Metrics Near 100%?](#why-are-some-synthetic-metrics-near-100)
- [Application Features](#application-features)
- [Repository Structure](#repository-structure)
- [Local Development Setup](#local-development-setup)
- [Live Deployment](#live-deployment)
- [Clinical Disclaimer](#clinical-disclaimer)
- [GitHub Topics](#recommended-github-topics)

---

## Overview

MediScan AI provides an interactive clinical interface for multi-disease risk evaluation. Rather than relying on a single diagnostic endpoint, the system segments assessment into three dedicated modules corresponding to established clinical cohorts:

1. **Diabetes Risk Module**: Evaluates oral glucose tolerance, serum insulin levels, body mass index, and familial predisposition.
2. **Cardiovascular Assessment**: Evaluates resting hemodynamics, exercise electrocardiography (ST segment depression/slope), and serum cholesterol levels.
3. **Parkinson's Acoustic Analysis**: Evaluates phonation cycle-to-cycle perturbation (jitter, shimmer) and nonlinear dynamic measures (RPDE, DFA, PPE) extracted from sustained vowel recordings.

---

## Dataset Provenance & Methodology

To ensure open accessibility without storing or exposing HIPAA-restricted **Protected Health Information (PHI)**, the models were developed using **synthetically modeled benchmark cohorts**. Each dataset was parameterized mathematically to reflect the feature distributions, correlations, and clinical ranges documented in established medical research repositories:

| Module | Dataset Type | Total Samples | Train / Test Split | Empirical Reference & Source Link |
|---|---|:---:|:---:|---|
| **Diabetes** | Synthetic Benchmark | 1,200 | 960 / 240 (80/20) | [UCI Pima Indians Diabetes Database](https://archive.ics.uci.edu/dataset/34/diabetes) ([Kaggle Mirror](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)) |
| **Cardiovascular** | Synthetic Benchmark | 1,200 | 960 / 240 (80/20) | [UCI Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| **Parkinson's** | Synthetic Benchmark | 900 | 720 / 180 (80/20) | [UCI Parkinson's Disease Dataset (Oxford)](https://archive.ics.uci.edu/dataset/174/parkinsons) |

---

## Data Hygiene & Leakage Audit

A rigorous data hygiene verification was conducted across all training and hold-out splits:

- **Split & Synthesis Sequence**: Datasets were generated in full and partitioned using `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`. No post-split augmentation or oversampling was applied.
- **Duplicate Rows**: Verified **0 duplicate rows** in the training sets and **0 duplicate rows** in the test sets.
- **Train-Test Leakage**: Verified **0 overlapping rows** between training partitions and hold-out test partitions.
- **Preprocessing Isolation**: `StandardScaler` transformations are fit strictly on the training partitions (`X_train`) and only applied (`transform`) to test partitions and live user inputs, preventing data leakage.

---

## Comprehensive Model Evaluation

Medical decision support requires evaluating both discriminative power (ROC-AUC) and screening sensitivity (**Recall**). All models are evaluated below across both **5-Fold Stratified Cross-Validation** (on training data) and independent **Hold-out Test Partitions**:

### 1. 5-Fold Stratified Cross-Validation (Train Partition)

| Module | CV Accuracy (Mean ± Std) | CV Recall (Sensitivity) | CV Precision | CV F1-Score | CV ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Diabetes** | 98.3% ± 0.6% | 96.7% ± 2.0% | 98.5% ± 1.3% | 97.6% ± 0.9% | 0.998 ± 0.001 |
| **Cardiovascular** | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | 1.000 ± 0.000 |
| **Parkinson's** | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | 1.000 ± 0.000 |

### 2. Independent Hold-out Test Evaluation

| Module | Test Accuracy | Precision (PPV) | Recall (Sensitivity) | F1-Score | ROC-AUC | Brier Score Loss |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Diabetes** | **97.9%** | 96.4% | 97.6% | 97.0% | **0.999** | **0.0189** |
| **Cardiovascular** | **100.0%** | 100.0% | 100.0% | 100.0% | **1.000** | **0.0000** |
| **Parkinson's** | **100.0%** | 100.0% | 100.0% | 100.0% | **1.000** | **0.0000** |

*Note: Brier score measures the mean squared difference between predicted probabilities and actual outcomes (0.0 represents perfect probabilistic alignment).*

### 3. Hold-out Confusion Matrices

```text
Diabetes Test Partition (N=240)
                  Predicted Negative    Predicted Positive
Actual Negative          154 (TN)               3 (FP)
Actual Positive            2 (FN)              81 (TP)

Cardiovascular Test Partition (N=240)
                  Predicted Negative    Predicted Positive
Actual Negative          120 (TN)               0 (FP)
Actual Positive            0 (FN)             120 (TP)

Parkinson's Test Partition (N=180)
                  Predicted Negative    Predicted Positive
Actual Negative           43 (TN)               0 (FP)
Actual Positive            0 (FN)             137 (TP)
```

### 4. Probability Calibration & Brier Metric Analysis

In clinical machine learning, raw classifier outputs (`predict_proba`) from tree-based ensembles represent leaf vote aggregations rather than formally calibrated Bayesian posterior probabilities.
- **Reporting Fidelity**: To maintain strict technical transparency without making ungrounded claims, outputs are presented as **Decision Confidence Scores** rather than true calibrated probabilities.
- **Calibration Verification**: We measure probabilistic alignment using the **Brier Score Loss** (mean squared error between confidence values and true binary outcomes: `0.0189` for Diabetes, `0.0000` for Cardiovascular and Parkinson's).
- **Clinical Calibration Roadmap**: In clinical deployments requiring true population posteriors, a dedicated post-hoc calibration wrapper such as `CalibratedClassifierCV(method='sigmoid')` (Platt Scaling) or Isotonic Regression can be trained on a distinct calibration hold-out split.

---

## Why Are Some Synthetic Metrics Near 100%?

In commercial and academic evaluations, reporting ~100% accuracy on complex pathologies raises immediate questions. Here is the technical explanation:

1. **Synthetic Feature Boundary Separation**: The data generation functions in `colab_files_to_train_models/MediScan_AI_Train_Models.ipynb` model clinical risk using distinct class-conditional normal and uniform distributions (e.g., chest pain type `cp` in cardiovascular, or phonation frequency ranges in Parkinson's). Because the generator functions had limited noise overlap between positive and negative classes, tree-based estimators (`GradientBoostingClassifier`, 200 estimators, max depth 5) cleanly partition the feature space.
2. **Comparison with Real-World Clinical Benchmarks**:
   - On the original **UCI Pima Indians Diabetes** dataset with missing entries and noisy labs, standard models achieve **~76–84% accuracy**.
   - On the original **UCI Cleveland Heart Disease** dataset with real patient records, standard algorithms reach **~82–88% accuracy**.
   - On the original **Oxford Parkinson's** voice dataset, standard acoustic classifiers typically yield **~85–92% recall**.
   
MediScan AI serves as an architectural prototype demonstrating how an end-to-end normalization, inference, clinical risk categorization, and doctor consultation referral pipeline functions.

---

## Application Features

- **Multi-Module Risk Screening**: Independent forms tailored to Diabetes, Cardiology, and Neurology.
- **Session-Level Patient Profile**: Manage name, age, blood group, allergies, medications, and previous history in-memory.
- **Diagnostic Upload & Tabular Ingestion**: Upload lab CSV reports for automated summary statistics.
- **Clinical Follow-up & Referral Recommendations**: Automatically maps positive assessments to corresponding medical specialists (Endocrinologist, Cardiologist, Neurologist) and recommended secondary lab workups (HbA1c, 12-lead ECG, UPDRS).
- **Session Risk Timeline**: Interactive progression charts tracking predicted risk probabilities across the browser session, with CSV export.
- **Model Analytics Dashboard**: Live inspection of training/test partitions, feature lists, and 5-fold cross-validation metrics.

---

## Interface Preview

```text
┌────────────────────────────────────────────────────────────────────────┐
│  🏥 MediScan AI — Clinical Disease Risk Assessment System              │
├───────────────────┬────────────────────────────────────────────────────┤
│  Navigation       │  Cardiovascular Risk Assessment                    │
│  ──────────       │  ───────────────────────────────                   │
│  • Home           │  [ Baseline Demographics ]                         │
│  • Patient Profile│  Age: 52  |  Sex: Male  |  Chest Pain: Angina      │
│  • Diabetes Risk  │                                                    │
│  • Heart Disease  │  [ Hemodynamics & Lab Findings ]                   │
│  • Parkinson's    │  Resting BP: 130 mmHg  |  Cholesterol: 246 mg/dL   │
│  • Risk History   │  Max Heart Rate: 150   |  ST Depression: 1.0       │
│  • Analytics      │                                                    │
│  • About          │  [ Run Cardiovascular Assessment ]                 │
│                   │                                                    │
│                   │  ┌──────────────────────────────────────────────┐  │
│                   │  │  ⚠️ Coronary Disease Indicators Detected     │  │
│                   │  │  Confidence Score: 94.2%                     │  │
│                   │  └──────────────────────────────────────────────┘  │
└───────────────────┴────────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
MediScan_AI/
├── app.py                          # Streamlit application & inference pipeline
├── requirements.txt                # Pinned production Python dependencies
├── runtime.txt                     # Target Python runtime environment (3.10.12)
├── .streamlit/
│   └── config.toml                 # Streamlit headless server & dark theme configuration
│
├── dataset/                        # Benchmark datasets (verified 0 duplicates)
│   ├── diabetes_train.csv          # 960 rows
│   ├── diabetes_test.csv           # 240 rows
│   ├── diabetes_full.csv           # 1,200 rows
│   ├── heart_train.csv             # 960 rows
│   ├── heart_test.csv              # 240 rows
│   ├── heart_full.csv              # 1,200 rows
│   ├── parkinsons_train.csv        # 720 rows
│   ├── parkinsons_test.csv         # 180 rows
│   └── parkinsons_full.csv         # 900 rows
│
├── saved_models/                   # Serialized estimators, scalers, and metric manifests
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── diabetes_metrics.json       # CV, test metrics, and data audit
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── heart_metrics.json
│   ├── parkinsons_model.pkl
│   ├── parkinsons_scaler.pkl
│   └── parkinsons_metrics.json
│
└── colab_files_to_train_models/    # Training & data generation notebook
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

## Recommended GitHub Topics

To optimize discoverability on GitHub, add the following topics to your repository settings:
`machine-learning`, `streamlit`, `healthcare-ai`, `disease-prediction`, `scikit-learn`, `gradient-boosting`, `clinical-decision-support`, `python`

---

## Clinical Disclaimer

MediScan AI is an experimental decision-support research prototype designed for demonstration, technical evaluation, and educational inquiry. It has not received clearance or certification as a medical diagnostic device by the FDA, EMA, or any national health authority. Model outputs must be corroborated with standard clinical diagnostic workups and professional medical judgment.

---

## Author

**Ashish Balodia**  
GitHub: [@ashishbalodia1](https://github.com/ashishbalodia1)
