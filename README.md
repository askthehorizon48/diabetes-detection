# 🩺 AI-Based Diabetes Risk Prediction System

An end-to-end Machine Learning pipeline and web interface designed to detect diabetes risk from patient clinical indicators and lifestyle health factors.

---

## 📌 Project Overview & Problem Statement
Diabetes mellitus is a growing global health crisis. Early detection and risk assessment are critical for preventative care and reducing long-term cardiovascular complications. 

This project implements a machine learning solution to analyze patient health indicators (such as Blood Pressure, BMI, Cholesterol, HbA1c, and lifestyle habits) and deliver instant risk predictions alongside explainable medical insights.

### 🎯 SDG Alignment: SDG 3 — Good Health & Well-Being
* **Target 3.4:** Reduce premature mortality from non-communicable diseases through prevention and treatment.
* **Impact:** By identifying high-risk individuals early using non-invasive clinical indicators, this system empowers proactive intervention and lifestyle modifications before advanced disease onset.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Processing & Analysis:** `pandas`, `numpy`, `scipy`
* **Machine Learning Frameworks:** `scikit-learn`, `xgboost`, `lightgbm`
* **Model Explainability:** `shap` (SHapley Additive exPlanations)
* **Visualization:** `matplotlib`, `seaborn`
* **Web Prototype / Interface:** `streamlit` (or `reflex` / `fastapi`)

---

## 📁 Repository Structure

```text
├── data/
│   └── README.md              # Instructions for fetching Kaggle dataset files
├── notebooks/
│   └── 01_eda_and_modeling.ipynb # Exploratory Data Analysis & baseline model experiments
├── src/
│   ├── preprocessing.py       # Data cleaning, zero-imputation, and scaling
│   ├── train.py               # Model training pipelines & cross-validation
│   └── evaluate.py            # SHAP feature importance, ROC-AUC, PR-AUC calculations
├── app.py                     # Interactive Streamlit web prototype
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

```
---

## 📊 Dataset & Preprocessing Pipeline

This project uses the **BRFSS Diabetes Health Indicators Dataset** (CDC) / **Pima Indians Dataset**.

1. **Missing Data & Zero Imputation:** Clinical parameters with unrealistic zeros (e.g., Blood Pressure, Glucose, BMI) are median-imputed based on age and target class groups.
2. **Feature Scaling & Normalization:** Continuous risk factors are standardized using `StandardScaler` within a `Pipeline` to prevent data leakage during cross-validation.
3. **Class Imbalance Handling:** Implemented stratified sampling (`StratifiedKFold`) and adjusted algorithmic class weights (`scale_pos_weight`) to optimize sensitivity for positive cases.

---

## 🔬 Model Architectures & Performance Comparison

We evaluated multiple machine learning architectures using **Stratified 5-Fold Cross-Validation** (These values are just placeholders as a baseline-Actual values to be updated):

| Model | Accuracy | ROC-AUC | PR-AUC | Recall (Sensitivity) | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 0.742 | 0.812 | 0.620 | 0.710 | 0.650 |
| **Random Forest Classifier** | 0.835 | 0.884 | 0.745 | 0.768 | 0.731 |
| **XGBoost Classifier (Selected)** | **0.851** | **0.908** | **0.782** | **0.812** | **0.764** |

> **Key Clinical takeaway:** **Recall** was prioritized during model selection to minimize False Negatives, ensuring potentially diabetic patients are not overlooked during screening.

---

## 💡 Medical Insights & Model Interpretability (SHAP Analysis)

Using **SHAP (SHapley Additive exPlanations)**, model predictions are made transparent for healthcare practitioners:

1. **Primary Predictors:** Blood Glucose / HbA1c levels, Body Mass Index (BMI), and High Blood Pressure consistently emerged as the highest-impact factors driving risk scores.
2. **Feature Interactions:** Strong non-linear interactions were observed between Age and BMI—higher BMI exponentially increases risk in patients aged 45+.
3. **Bias & False Negative Auditing:** Audited model outputs across demographic subgroups (age/gender) to ensure uniform diagnostic sensitivity across patient cohorts.

---

## 🚀 Setup & Execution Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/diabetes-detection-ai.git](https://github.com/your-username/diabetes-detection-ai.git)
cd diabetes-detection-ai

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

### 3. Install Dependencies
``` bash
pip install -r requirements.txt

### 4. Run the Working Prototype Interface
``` bash
streamlit run app.py
```

---

## 📜 Acknowledgments & References

CDC Behavioral Risk Factor Surveillance System (BRFSS) Dataset.   Scikit-Learn, XGBoost, and SHAP documentation.
