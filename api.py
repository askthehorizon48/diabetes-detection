# -*- coding: utf-8 -*-
"""
FastAPI backend for Diabetes Detection
Endpoints:
  POST /predict  — returns risk score, label, engineered features, and SHAP values
  GET  /health   — health check

Model: XGBoost + Random Forest ensemble (LightGBM excluded due to Windows crash)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import shap
import uvicorn
import traceback

# ──────────────────────────────────────────────
# Load model artifacts
# ──────────────────────────────────────────────
try:
    model    = joblib.load("rf_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    metadata = joblib.load("model_metadata.pkl")

    THRESHOLD          = metadata["threshold"]
    FEATURE_COLS       = metadata["feature_cols"]
    CONTINUOUS_FEATURES= metadata["continuous_features"]
    MODEL_NAME         = metadata["model_name"]
    DATASET_USED       = metadata["dataset_used"]
    print(f"[OK] Loaded: {MODEL_NAME} | Dataset: {DATASET_USED} | Threshold: {THRESHOLD:.2f}")
except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found. Run train.py first.\n{e}")


# ──────────────────────────────────────────────
# Feature Engineering (must mirror train.py)
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI"] = df["BMI"].replace(0, df["BMI"].median() if df["BMI"].median() != 0 else 25)

    df["Metabolic_Risk_Score"]      = df["HighBP"] + df["HighChol"] + (df["BMI"] >= 30).astype(int)
    df["Comorbidity_Index"]         = df["HeartDiseaseorAttack"] + df["Stroke"] + df["HighBP"]
    df["Age_BMI_Interaction"]       = df["Age"] * df["BMI"]
    df["Unhealthy_Lifestyle_Score"] = df["Smoker"] + (1 - df["PhysActivity"]) + (1 - df["Fruits"]) + (1 - df["Veggies"])
    df["Health_Access_Score"]       = df["AnyHealthcare"] - df["NoDocbcCost"]
    df["Poor_Health_Days_Total"]    = df["MentHlth"] + df["PhysHlth"]
    df["CardioRisk"]                = df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"] + df["Stroke"]
    df["BMI_Category"]              = int(pd.cut([df["BMI"].iloc[0]], bins=[0, 18.5, 25, 30, 35, 100], labels=[0,1,2,3,4])[0])
    df["Age_GenHlth_Interaction"]   = df["Age"] * df["GenHlth"]
    return df


# ──────────────────────────────────────────────
# SHAP explainer (cached)
# ──────────────────────────────────────────────
def get_shap_values(df_input: pd.DataFrame):
    try:
        # For ensemble VotingClassifier, use the first estimator
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model.estimators_[0])
        else:
            explainer = shap.TreeExplainer(model)

        shap_obj = explainer(df_input)

        # Distinguish multi-output vs single-output
        if shap_obj.values.ndim == 3:
            # shape: (n_samples, n_features, n_classes)
            vals = shap_obj.values[0, :, 1]
            base = float(shap_obj.base_values[0, 1])
        else:
            vals = shap_obj.values[0]
            base = float(shap_obj.base_values[0])

        feature_names = df_input.columns.tolist()
        shap_result = [
            {"feature": fn, "value": float(v), "feature_value": float(df_input[fn].iloc[0])}
            for fn, v in zip(feature_names, vals)
        ]
        # Sort by absolute importance, top 10
        shap_result.sort(key=lambda x: abs(x["value"]), reverse=True)
        return shap_result[:10], base
    except Exception as exc:
        print(f"SHAP warning: {exc}")
        return [], 0.0


# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="Diabetes Detection API",
    description="Explainable AI diabetes risk prediction API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────
class PatientInput(BaseModel):
    # Core vitals
    HighBP: int = Field(..., ge=0, le=1, description="High Blood Pressure (0=No, 1=Yes)")
    HighChol: int = Field(..., ge=0, le=1)
    CholCheck: int = Field(1, ge=0, le=1, description="Cholesterol check in past 5 years")
    BMI: float = Field(..., ge=10.0, le=100.0)

    # Medical history
    Stroke: int = Field(..., ge=0, le=1)
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1)
    DiffWalk: int = Field(0, ge=0, le=1, description="Difficulty walking or climbing stairs")

    # Lifestyle
    Smoker: int = Field(..., ge=0, le=1)
    HvyAlcoholConsump: int = Field(0, ge=0, le=1)
    PhysActivity: int = Field(..., ge=0, le=1)
    Fruits: int = Field(..., ge=0, le=1)
    Veggies: int = Field(..., ge=0, le=1)

    # Healthcare access
    AnyHealthcare: int = Field(1, ge=0, le=1)
    NoDocbcCost: int = Field(0, ge=0, le=1, description="Could not see doctor due to cost")

    # General health
    GenHlth: int = Field(..., ge=1, le=5, description="1=Excellent → 5=Poor")
    MentHlth: int = Field(0, ge=0, le=30)
    PhysHlth: int = Field(0, ge=0, le=30)

    # Demographics
    Sex: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    Age: int = Field(..., ge=1, le=13, description="1=18-24 … 13=80+")
    Education: int = Field(5, ge=1, le=6)
    Income: int = Field(6, ge=1, le=8)


class SHAPEntry(BaseModel):
    feature: str
    value: float
    feature_value: float


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: str          # "low" | "elevated" | "high"
    risk_percentage: str
    metabolic_risk_score: int
    unhealthy_lifestyle_score: int
    comorbidity_index: int
    cardio_risk: int
    shap_values: list[SHAPEntry]
    shap_base_value: float
    model_info: dict


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dataset": DATASET_USED,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    try:
        raw = patient.model_dump()
        df_raw = pd.DataFrame([raw])

        # Feature engineering
        df_fe = engineer_features(df_raw)

        # Extract engineered metrics before scaling
        metabolic_risk_score      = int(df_fe["Metabolic_Risk_Score"].iloc[0])
        unhealthy_lifestyle_score = int(df_fe["Unhealthy_Lifestyle_Score"].iloc[0])
        comorbidity_index         = int(df_fe["Comorbidity_Index"].iloc[0])
        cardio_risk               = int(df_fe["CardioRisk"].iloc[0])

        # Scale continuous features
        df_fe[CONTINUOUS_FEATURES] = scaler.transform(df_fe[CONTINUOUS_FEATURES])

        # Align columns
        df_input = df_fe[FEATURE_COLS]

        # Predict
        prob = float(model.predict_proba(df_input)[0][1])

        # Risk labelling with tuned threshold
        if prob >= THRESHOLD:
            risk_label = "high"
        elif prob >= THRESHOLD * 0.5:
            risk_label = "elevated"
        else:
            risk_label = "low"

        # SHAP
        shap_values, base_value = get_shap_values(df_input)

        return PredictionResponse(
            risk_probability=round(prob, 4),
            risk_label=risk_label,
            risk_percentage=f"{prob:.1%}",
            metabolic_risk_score=metabolic_risk_score,
            unhealthy_lifestyle_score=unhealthy_lifestyle_score,
            comorbidity_index=comorbidity_index,
            cardio_risk=cardio_risk,
            shap_values=[SHAPEntry(**s) for s in shap_values],
            shap_base_value=round(base_value, 4),
            model_info={
                "name": MODEL_NAME,
                "dataset": DATASET_USED,
                "threshold": round(THRESHOLD, 2),
            },
        )

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
