// Results panel — shows risk score, metrics, SHAP chart
import React from 'react';

const RISK_CONFIG = {
  low: {
    icon: '✅',
    label: 'Low Risk',
    title: 'Diabetes Risk: Low',
    subtitle: 'Current indicators suggest a low likelihood of diabetes.',
    color: 'low',
  },
  elevated: {
    icon: '⚠️',
    label: 'Elevated Risk',
    title: 'Prediabetes / Elevated Risk',
    subtitle: 'Some risk factors detected. Lifestyle changes recommended.',
    color: 'elevated',
  },
  high: {
    icon: '🚨',
    label: 'High Risk',
    title: 'High Risk of Diabetes',
    subtitle: 'Multiple risk factors identified. Consult a healthcare provider.',
    color: 'high',
  },
};

/** SHAP horizontal bar chart */
function SHAPChart({ values }) {
  if (!values || values.length === 0) return null;

  const maxAbs = Math.max(...values.map((v) => Math.abs(v.value)), 0.001);

  return (
    <div className="card fade-up">
      <div className="card-title">Clinical Feature Impact (SHAP Analysis)</div>
      <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 16 }}>
        Factors driving this specific risk prediction. Blue = increases risk, grey = decreases risk.
      </p>
      {values.map((item) => {
        const absVal = Math.abs(item.value);
        const pct = (absVal / maxAbs) * 100;
        const isPositive = item.value > 0;

        return (
          <div className="progress-row" key={item.feature}>
            <span className="progress-label" title={item.feature}>
              {formatFeatureName(item.feature)}
            </span>
            <div className="progress-bar-track">
              <div
                className={`progress-bar-fill ${isPositive ? 'positive' : 'negative'}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="progress-val">{item.value.toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
}

function formatFeatureName(name) {
  const map = {
    HighBP: 'High Blood Pressure',
    HighChol: 'High Cholesterol',
    BMI: 'BMI',
    Smoker: 'Smoker',
    Stroke: 'Stroke History',
    HeartDiseaseorAttack: 'Heart Disease/Attack',
    PhysActivity: 'Physical Activity',
    Fruits: 'Daily Fruits',
    Veggies: 'Daily Vegetables',
    HvyAlcoholConsump: 'Heavy Alcohol',
    AnyHealthcare: 'Has Healthcare',
    NoDocbcCost: 'Skipped Doctor (Cost)',
    GenHlth: 'General Health',
    MentHlth: 'Mental Health Days',
    PhysHlth: 'Physical Health Days',
    DiffWalk: 'Difficulty Walking',
    Sex: 'Sex',
    Age: 'Age Group',
    Education: 'Education Level',
    Income: 'Income Level',
    Metabolic_Risk_Score: 'Metabolic Risk Score',
    Comorbidity_Index: 'Comorbidity Index',
    Age_BMI_Interaction: 'Age × BMI',
    Unhealthy_Lifestyle_Score: 'Lifestyle Risk Score',
    Health_Access_Score: 'Healthcare Access',
    Poor_Health_Days_Total: 'Poor Health Days Total',
    CardioRisk: 'Cardiovascular Risk',
    BMI_Category: 'BMI Category',
    Age_GenHlth_Interaction: 'Age × Gen. Health',
    CholCheck: 'Cholesterol Check',
  };
  return map[name] || name.replace(/_/g, ' ');
}

export default function ResultsPanel({ result, loading, error }) {
  if (loading) {
    return (
      <div className="results-panel">
        <div className="spinner-overlay">
          <div className="spinner" />
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Running diagnostic analysis…
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="results-panel">
        <div className="error-msg">
          <strong>Error:</strong> {error}
          <br />
          <span style={{ opacity: 0.7 }}>Make sure the FastAPI backend is running on port 8000.</span>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="results-panel">
        <div className="empty-state">
          <div className="empty-icon">🩺</div>
          <h3>Awaiting Patient Data</h3>
          <p>
            Fill in the patient clinical profile on the left and click{' '}
            <strong>Analyse Risk</strong> to generate a prediction.
          </p>
        </div>
      </div>
    );
  }

  const cfg = RISK_CONFIG[result.risk_label] || RISK_CONFIG.low;

  return (
    <div className="results-panel">
      {/* Risk banner */}
      <div className={`risk-banner ${cfg.color} fade-up`}>
        <div className="risk-icon">{cfg.icon}</div>
        <div className="risk-content">
          <div className="risk-label">{cfg.label}</div>
          <div className="risk-title">{cfg.title}</div>
          <div className="risk-subtitle">{cfg.subtitle}</div>
          <div style={{ marginTop: 10 }}>
            <span className="model-chip">
              🤖 {result.model_info?.name ?? 'Ensemble'} · threshold {result.model_info?.threshold}
            </span>
          </div>
        </div>
        <div className="risk-probability" style={{ color: `var(--risk-${cfg.color})` }}>
          {result.risk_percentage}
        </div>
      </div>

      {/* Engineered metric cards */}
      <div className="card fade-up">
        <div className="card-title">Derived Clinical Metrics</div>
        <div className="metric-grid">
          <MetricCard label="Metabolic Risk" value={result.metabolic_risk_score} max="/ 3" />
          <MetricCard label="Lifestyle Risk" value={result.unhealthy_lifestyle_score} max="/ 4" />
          <MetricCard label="Comorbidity Index" value={result.comorbidity_index} max="/ 3" />
          <MetricCard label="Cardiovascular Risk" value={result.cardio_risk} max="/ 4" />
        </div>
      </div>

      {/* SHAP chart */}
      <SHAPChart values={result.shap_values} />

      <div className="disclaimer fade-up">
        ⚠️ This tool is for educational and research purposes only. It is <strong>not</strong> a
        substitute for professional medical advice, diagnosis, or treatment. Always consult a
        qualified healthcare provider.
      </div>
    </div>
  );
}

function MetricCard({ label, value, max }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-max">{max}</div>
    </div>
  );
}
