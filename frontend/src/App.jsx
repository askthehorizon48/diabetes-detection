import React, { useState } from 'react';
import PatientForm, { DEFAULT_FORM } from './components/PatientForm';
import ResultsPanel from './components/ResultsPanel';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleAnalyse() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Unknown error. Is the API running?');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="app-header">
        <div className="header-logo">🩺</div>
        <div className="header-title">
          <h1>e-CRS Diabetes Risk Predictor</h1>
          <p>Explainable Clinical Risk Score · Powered by XGBoost + LightGBM Ensemble</p>
        </div>
        <span className="header-badge">AI Diagnostic Tool</span>
      </header>

      {/* Sidebar */}
      <PatientForm
        form={form}
        setForm={setForm}
        onSubmit={handleAnalyse}
        loading={loading}
      />

      {/* Results */}
      <ResultsPanel result={result} loading={loading} error={error} />
    </div>
  );
}
