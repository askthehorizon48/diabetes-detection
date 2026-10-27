# -*- coding: utf-8 -*-
"""
Diabetes Detection - Improved Training Pipeline
================================================
Strategy:
  1. Train on BOTH datasets (full imbalanced + balanced 50/50 split)
  2. Advanced feature engineering (9 derived features)
  3. Proper nested 5-fold CV - scaler fit inside each fold (no leakage)
  4. Models: XGBoost variants + Random Forest + XGB/RF VotingEnsemble
  5. OOF threshold tuning - maximise F1 without class bias
  6. Save best model + scaler + threshold for FastAPI serving

Note: LightGBM excluded - known Windows OpenMP access-violation crash
on this machine regardless of dataset size.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score,
    balanced_accuracy_score, classification_report,
)
from xgboost import XGBClassifier


# ──────────────────────────────────────────────
# 1. Feature Engineering
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI"] = df["BMI"].replace(0, df["BMI"].median())

    df["Metabolic_Risk_Score"]      = df["HighBP"] + df["HighChol"] + (df["BMI"] >= 30).astype(int)
    df["Comorbidity_Index"]         = df["HeartDiseaseorAttack"] + df["Stroke"] + df["HighBP"]
    df["Age_BMI_Interaction"]       = df["Age"] * df["BMI"]
    df["Unhealthy_Lifestyle_Score"] = df["Smoker"] + (1 - df["PhysActivity"]) + (1 - df["Fruits"]) + (1 - df["Veggies"])
    df["Health_Access_Score"]       = df["AnyHealthcare"] - df["NoDocbcCost"]
    df["Poor_Health_Days_Total"]    = df["MentHlth"] + df["PhysHlth"]
    df["CardioRisk"]                = df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"] + df["Stroke"]
    df["BMI_Category"]              = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, 35, 100], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df["Age_GenHlth_Interaction"]   = df["Age"] * df["GenHlth"]
    return df


CONTINUOUS_FEATURES = [
    "BMI", "MentHlth", "PhysHlth",
    "Age_BMI_Interaction", "Poor_Health_Days_Total", "Age_GenHlth_Interaction"
]


# ──────────────────────────────────────────────
# 2. Load datasets
# ──────────────────────────────────────────────
print("=" * 70)
print("DIABETES DETECTION - ADVANCED TRAINING PIPELINE")
print("=" * 70)

TARGET = "Diabetes_binary"

print("\n[1/5] Loading datasets...")
df_full     = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")
df_balanced = pd.read_csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

print(f"  Full dataset     : {len(df_full):,} rows | {df_full[TARGET].value_counts().to_dict()}")
print(f"  Balanced dataset : {len(df_balanced):,} rows | {df_balanced[TARGET].value_counts().to_dict()}")

print("\n[2/5] Engineering features...")
df_full_fe     = engineer_features(df_full)
df_balanced_fe = engineer_features(df_balanced)

X_full     = df_full_fe.drop(TARGET, axis=1)
y_full     = df_full_fe[TARGET].astype(np.int32)
X_balanced = df_balanced_fe.drop(TARGET, axis=1)
y_balanced = df_balanced_fe[TARGET].astype(np.int32)

# Class imbalance ratio for full dataset
neg, pos = np.bincount(y_full)
spw = round(neg / pos, 1)
print(f"  Full dataset scale_pos_weight = {spw}")


# ──────────────────────────────────────────────
# 3. Define models
# ──────────────────────────────────────────────
print("\n[3/5] Defining models & ensembles...")

# --- Full (imbalanced) dataset models ---
# XGBoost with mild class weighting - boosts minority recall without sacrificing accuracy
xgb_full_mild = XGBClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.75,
    min_child_weight=5, gamma=0.1,
    reg_alpha=0.05, reg_lambda=2.0,
    scale_pos_weight=2,           # mild - keeps accuracy high, improves recall
    eval_metric="logloss", tree_method="hist",
    random_state=42, n_jobs=-1,
)

# XGBoost with no weighting - maximises raw accuracy
xgb_full_noweight = XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.75,
    min_child_weight=3, gamma=0.05,
    reg_alpha=0.1, reg_lambda=2.0,
    scale_pos_weight=1,
    eval_metric="logloss", tree_method="hist",
    random_state=42, n_jobs=-1,
)

rf_full = RandomForestClassifier(
    n_estimators=300, max_depth=15,
    min_samples_split=8, min_samples_leaf=4,
    max_features="sqrt", class_weight="balanced_subsample",
    random_state=42, n_jobs=-1,
)

# Ensemble: mild-weighted XGB + RF (full dataset)
ensemble_full = VotingClassifier(
    estimators=[
        ("xgb_mild", XGBClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            min_child_weight=5, gamma=0.1,
            reg_alpha=0.05, reg_lambda=2.0,
            scale_pos_weight=2,
            eval_metric="logloss", tree_method="hist",
            random_state=42, n_jobs=-1)),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=15,
            min_samples_split=8, min_samples_leaf=4,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=42, n_jobs=-1)),
    ],
    voting="soft",
)

# --- Balanced dataset models ---
xgb_bal = XGBClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.75,
    min_child_weight=3, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.5,
    scale_pos_weight=1,
    eval_metric="logloss", tree_method="hist",
    random_state=42, n_jobs=-1,
)

rf_bal = RandomForestClassifier(
    n_estimators=300, max_depth=12,
    min_samples_split=10, min_samples_leaf=5,
    max_features="sqrt", class_weight="balanced",
    random_state=42, n_jobs=-1,
)

ensemble_bal = VotingClassifier(
    estimators=[
        ("xgb", XGBClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            min_child_weight=3, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=1,
            eval_metric="logloss", tree_method="hist",
            random_state=42, n_jobs=-1)),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=12,
            min_samples_split=10, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ],
    voting="soft",
)

# Experiment plan: (label, X, y, {name: model})
experiment_plan = [
    ("Full (Imbalanced)", X_full, y_full, {
        "XGBoost (spw=2)"    : xgb_full_mild,
        "XGBoost (spw=1)"    : xgb_full_noweight,
        "Random Forest"      : rf_full,
        "Ensemble XGB+RF"    : ensemble_full,
    }),
    ("Balanced 50/50", X_balanced, y_balanced, {
        "XGBoost"            : xgb_bal,
        "Random Forest"      : rf_bal,
        "Ensemble XGB+RF"    : ensemble_bal,
    }),
]


# ──────────────────────────────────────────────
# 4. Stratified 5-Fold cross-validation
# ──────────────────────────────────────────────
print("\n[4/5] Cross-validating all models on both datasets...")
print("=" * 92)
print(f"{'Dataset':<20} {'Model':<22} {'Acc':>7} {'Bal-Acc':>8} {'ROC-AUC':>8} {'F1':>7} {'Recall':>7}")
print("=" * 92)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_score  = 0.0
best_config = None  # (ds_name, m_name, model, X, y, feature_cols)

for ds_name, X, y, models in experiment_plan:
    for m_name, model in models.items():
        fold_acc, fold_bal, fold_roc, fold_f1, fold_rec = [], [], [], [], []

        for train_idx, test_idx in skf.split(X, y):
            X_tr = X.iloc[train_idx].copy()
            X_te = X.iloc[test_idx].copy()
            y_tr = y.iloc[train_idx]
            y_te = y.iloc[test_idx]

            scaler_fold = StandardScaler()
            X_tr[CONTINUOUS_FEATURES] = scaler_fold.fit_transform(X_tr[CONTINUOUS_FEATURES])
            X_te[CONTINUOUS_FEATURES] = scaler_fold.transform(X_te[CONTINUOUS_FEATURES])

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]

            fold_acc.append(accuracy_score(y_te, y_pred))
            fold_bal.append(balanced_accuracy_score(y_te, y_pred))
            fold_roc.append(roc_auc_score(y_te, y_prob))
            fold_f1.append(f1_score(y_te, y_pred))
            fold_rec.append(recall_score(y_te, y_pred))

        ma = np.mean(fold_acc)
        mb = np.mean(fold_bal)
        mr = np.mean(fold_roc)
        mf = np.mean(fold_f1)
        mc = np.mean(fold_rec)

        print(f"{ds_name:<20} {m_name:<22} {ma:>7.4f} {mb:>8.4f} {mr:>8.4f} {mf:>7.4f} {mc:>7.4f}")

        # Composite score: reward both accuracy AND balanced accuracy (anti-gaming)
        # A model must be good on BOTH to win
        composite = (2 * ma * mb) / (ma + mb + 1e-9) * mr
        if composite > best_score:
            best_score  = composite
            best_config = (ds_name, m_name, model, X, y, X.columns.tolist())

print("=" * 92)


# ──────────────────────────────────────────────
# 5. Retrain best + OOF threshold tuning
# ──────────────────────────────────────────────
ds_name, m_name, best_model, X_best, y_best, feature_cols = best_config
print(f"\n[5/5] Best config -> [{ds_name}] [{m_name}]  (composite: {best_score:.4f})")
print("Retraining on full data and tuning decision threshold...")

final_scaler = StandardScaler()
X_final = X_best.copy()
X_final[CONTINUOUS_FEATURES] = final_scaler.fit_transform(X_final[CONTINUOUS_FEATURES])

# OOF threshold search
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
oof_probs  = np.zeros(len(y_best))
oof_labels = np.zeros(len(y_best))

for train_idx, test_idx in skf2.split(X_final, y_best):
    X_tr, X_te = X_final.iloc[train_idx], X_final.iloc[test_idx]
    best_model.fit(X_tr, y_best.iloc[train_idx])
    oof_probs[test_idx]  = best_model.predict_proba(X_te)[:, 1]
    oof_labels[test_idx] = y_best.iloc[test_idx].values

best_thresh, best_f1 = 0.5, 0.0
for t in np.arange(0.25, 0.76, 0.01):
    preds = (oof_probs >= t).astype(int)
    score = f1_score(oof_labels, preds)
    if score > best_f1:
        best_f1, best_thresh = score, t

print(f"  Optimal threshold: {best_thresh:.2f}  (OOF F1 = {best_f1:.4f})")

# Final fit on full training data
best_model.fit(X_final, y_best)

final_preds = (best_model.predict_proba(X_final)[:, 1] >= best_thresh).astype(int)
print("\n-- Final Training-set Classification Report --")
print(classification_report(y_best, final_preds))


# ──────────────────────────────────────────────
# 6. Export artifacts
# ──────────────────────────────────────────────
print("\nExporting model artifacts...")
joblib.dump(best_model,   "rf_model.pkl")
joblib.dump(final_scaler, "scaler.pkl")
joblib.dump({
    "threshold"          : best_thresh,
    "feature_cols"       : feature_cols,
    "continuous_features": CONTINUOUS_FEATURES,
    "dataset_used"       : ds_name,
    "model_name"         : m_name,
}, "model_metadata.pkl")

print(f"\nDone! Saved: rf_model.pkl, scaler.pkl, model_metadata.pkl")
print(f"  Model    : {m_name}")
print(f"  Dataset  : {ds_name}")
print(f"  Threshold: {best_thresh:.2f}")
