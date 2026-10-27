import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="e-CRS Diabetes Risk Predictor", layout="wide")

# 1. Loading Model and Scaler


@st.cache_resource
def load_assets():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error(
        "Model files not found. Run train.py first to generate rf_model.pkl and scaler.pkl.")
    st.stop()

st.title("🩺 Explainable Clinical Risk Score (e-CRS) for Diabetes")
st.markdown(
    "Enter patient indicators below to generate a diagnostic risk assessment and explainability chart.")

# 2. Sidebar for User Input
with st.sidebar:
    st.header("Patient Clinical Profile")
    age = st.slider("Age Group (1=18-24, 13=80+)", 1, 13, 5)
    bmi = st.number_input("BMI", min_value=10.0, max_value=90.0, value=25.0)

    # Helper function for Yes/No dropdowns
    def ask_yes_no(prompt):
        return 1 if st.selectbox(prompt, ["No", "Yes"]) == "Yes" else 0

    high_bp = ask_yes_no("High Blood Pressure?")
    high_chol = ask_yes_no("High Cholesterol?")
    heart_disease = ask_yes_no("Heart Disease / Attack History?")
    stroke = ask_yes_no("History of Stroke?")

    st.header("Lifestyle Profile")
    smoker = ask_yes_no("Smoker?")
    phys_activity = ask_yes_no("Regular Physical Activity?")
    fruits = ask_yes_no("Consumes Fruits Daily?")
    veggies = ask_yes_no("Consumes Vegetables Daily?")

    ment_hlth = st.slider("Poor Mental Health Days (Past 30 Days)", 0, 30, 0)
    phys_hlth = st.slider("Poor Physical Health Days (Past 30 Days)", 0, 30, 0)

    # Defaults for remaining BRFSS features required by the model
    gen_hlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 2)
    chol_check = 1
    hvy_alcohol = 0
    any_healthcare = 1
    no_doc_bc_cost = 0
    diff_walk = 0
    sex = 1
    education = 5
    income = 6

if st.button("Generate Diagnostic Risk Score"):
    # 3. Constructing raw input dictionary matching BRFSS dataset
    input_data = {
        'HighBP': high_bp, 'HighChol': high_chol, 'CholCheck': chol_check, 'BMI': bmi,
        'Smoker': smoker, 'Stroke': stroke, 'HeartDiseaseorAttack': heart_disease,
        'PhysActivity': phys_activity, 'Fruits': fruits, 'Veggies': veggies,
        'HvyAlcoholConsump': hvy_alcohol, 'AnyHealthcare': any_healthcare,
        'NoDocbcCost': no_doc_bc_cost, 'GenHlth': gen_hlth, 'MentHlth': ment_hlth,
        'PhysHlth': phys_hlth, 'DiffWalk': diff_walk, 'Sex': sex, 'Age': age,
        'Education': education, 'Income': income
    }

    df_input = pd.DataFrame([input_data])

    # 4. Applying Novel Feature Engineering (Crucial for consistency)
    df_input['Metabolic_Risk_Score'] = df_input['HighBP'] + \
        df_input['HighChol'] + (df_input['BMI'] >= 30).astype(int)
    df_input['Comorbidity_Index'] = df_input['HeartDiseaseorAttack'] + \
        df_input['Stroke'] + df_input['HighBP']
    df_input['Age_BMI_Interaction'] = df_input['Age'] * df_input['BMI']
    df_input['Unhealthy_Lifestyle_Score'] = df_input['Smoker'] + \
        (1 - df_input['PhysActivity']) + \
        (1 - df_input['Fruits']) + (1 - df_input['Veggies'])

    # 5. Applying Scaling to continuous features
    continuous_features = ['BMI', 'MentHlth',
                           'PhysHlth', 'Age_BMI_Interaction']
    df_input[continuous_features] = scaler.transform(
        df_input[continuous_features])

    # Aligning column order with the trained Random Forest model
    expected_cols = model.feature_names_in_
    df_input = df_input[expected_cols]

    # 6. Predicting Risk Probability
    prediction_prob = model.predict_proba(df_input)[0][1]

    # 7. Displaying Results & XAI (Explainable AI)
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagnostic Output")

        # Adjusting Clinical Thresholds for Imbalanced Data
        if prediction_prob >= 0.50:
            st.error(
                f"⚠️ High Risk of Diabetes Detected (Probability: {prediction_prob:.1%})")
        elif prediction_prob >= 0.25:
            st.warning(
                f"🚸 Elevated Risk / Prediabetes Warning (Probability: {prediction_prob:.1%})")
        else:
            st.success(
                f"✅ Low Risk of Diabetes (Probability: {prediction_prob:.1%})")

        st.metric("Metabolic Risk Score (0-3)",
                  int(df_input['Metabolic_Risk_Score'].iloc[0]))
        st.metric("Unhealthy Lifestyle Score (0-4)",
                  int(df_input['Unhealthy_Lifestyle_Score'].iloc[0]))

    with col2:
        st.subheader("Clinical Feature Impact")
        st.write("SHAP Analysis: Factors driving this specific risk prediction.")

        # Generating SHAP Waterfall Plot
        explainer = shap.TreeExplainer(model)
        shap_obj = explainer(df_input)

        # Isolating the 0th sample, all features, and index 1 (the positive class)
        explanation = shap_obj[0, :, 1]

        fig, ax = plt.subplots(figsize=(6, 4))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)
