// Patient form sidebar
import React from 'react';
import { YesNoToggle, Slider, Select, NumberInput } from './FormControls';

const AGE_OPTIONS = [
  { value: 1,  label: '18–24' },
  { value: 2,  label: '25–29' },
  { value: 3,  label: '30–34' },
  { value: 4,  label: '35–39' },
  { value: 5,  label: '40–44' },
  { value: 6,  label: '45–49' },
  { value: 7,  label: '50–54' },
  { value: 8,  label: '55–59' },
  { value: 9,  label: '60–64' },
  { value: 10, label: '65–69' },
  { value: 11, label: '70–74' },
  { value: 12, label: '75–79' },
  { value: 13, label: '80+' },
];

const EDU_OPTIONS = [
  { value: 1, label: 'Never attended school' },
  { value: 2, label: 'Elementary school' },
  { value: 3, label: 'Some high school' },
  { value: 4, label: 'High school graduate' },
  { value: 5, label: 'Some college' },
  { value: 6, label: 'College graduate' },
];

const INCOME_OPTIONS = [
  { value: 1, label: '< $10,000' },
  { value: 2, label: '$10,000–$15,000' },
  { value: 3, label: '$15,000–$20,000' },
  { value: 4, label: '$20,000–$25,000' },
  { value: 5, label: '$25,000–$35,000' },
  { value: 6, label: '$35,000–$50,000' },
  { value: 7, label: '$50,000–$75,000' },
  { value: 8, label: '≥ $75,000' },
];

const GEN_HLTH_OPTIONS = [
  { value: 1, label: '1 – Excellent' },
  { value: 2, label: '2 – Very Good' },
  { value: 3, label: '3 – Good' },
  { value: 4, label: '4 – Fair' },
  { value: 5, label: '5 – Poor' },
];

export const DEFAULT_FORM = {
  HighBP: 0,
  HighChol: 0,
  CholCheck: 1,
  BMI: 26.0,
  Stroke: 0,
  HeartDiseaseorAttack: 0,
  DiffWalk: 0,
  Smoker: 0,
  HvyAlcoholConsump: 0,
  PhysActivity: 1,
  Fruits: 1,
  Veggies: 1,
  AnyHealthcare: 1,
  NoDocbcCost: 0,
  GenHlth: 2,
  MentHlth: 0,
  PhysHlth: 0,
  Sex: 1,
  Age: 5,
  Education: 5,
  Income: 6,
};

export default function PatientForm({ form, setForm, onSubmit, loading }) {
  const set = (key) => (val) => setForm((prev) => ({ ...prev, [key]: val }));

  return (
    <div className="sidebar">
      {/* ── Core Vitals ── */}
      <div className="section-label">Core Vitals</div>

      <Select label="Age Group" value={form.Age} onChange={set('Age')} options={AGE_OPTIONS} />
      <NumberInput label="BMI" value={form.BMI} onChange={set('BMI')} min={10} max={99} step={0.5} hint="kg/m²" />
      <YesNoToggle label="High Blood Pressure?" value={form.HighBP} onChange={set('HighBP')} />
      <YesNoToggle label="High Cholesterol?" value={form.HighChol} onChange={set('HighChol')} />
      <YesNoToggle label="Cholesterol Checked (5 years)?" value={form.CholCheck} onChange={set('CholCheck')} />

      {/* ── Medical History ── */}
      <div className="section-label">Medical History</div>

      <YesNoToggle label="Heart Disease / Attack History?" value={form.HeartDiseaseorAttack} onChange={set('HeartDiseaseorAttack')} />
      <YesNoToggle label="History of Stroke?" value={form.Stroke} onChange={set('Stroke')} />
      <YesNoToggle label="Difficulty Walking / Climbing Stairs?" value={form.DiffWalk} onChange={set('DiffWalk')} />

      {/* ── Lifestyle ── */}
      <div className="section-label">Lifestyle</div>

      <YesNoToggle label="Smoker?" value={form.Smoker} onChange={set('Smoker')} />
      <YesNoToggle label="Heavy Alcohol Consumption?" value={form.HvyAlcoholConsump} onChange={set('HvyAlcoholConsump')} />
      <YesNoToggle label="Regular Physical Activity?" value={form.PhysActivity} onChange={set('PhysActivity')} />
      <YesNoToggle label="Consumes Fruits Daily?" value={form.Fruits} onChange={set('Fruits')} />
      <YesNoToggle label="Consumes Vegetables Daily?" value={form.Veggies} onChange={set('Veggies')} />

      {/* ── Health Status ── */}
      <div className="section-label">Health Status</div>

      <Select label="General Health" value={form.GenHlth} onChange={set('GenHlth')} options={GEN_HLTH_OPTIONS} />
      <Slider label="Poor Mental Health Days" min={0} max={30} value={form.MentHlth} onChange={set('MentHlth')} hint="past 30 days" />
      <Slider label="Poor Physical Health Days" min={0} max={30} value={form.PhysHlth} onChange={set('PhysHlth')} hint="past 30 days" />

      {/* ── Healthcare Access ── */}
      <div className="section-label">Healthcare Access</div>

      <YesNoToggle label="Has Any Healthcare Coverage?" value={form.AnyHealthcare} onChange={set('AnyHealthcare')} />
      <YesNoToggle label="Skipped Doctor Due to Cost?" value={form.NoDocbcCost} onChange={set('NoDocbcCost')} />

      {/* ── Demographics ── */}
      <div className="section-label">Demographics</div>

      <Select
        label="Sex"
        value={form.Sex}
        onChange={set('Sex')}
        options={[{ value: 0, label: 'Female' }, { value: 1, label: 'Male' }]}
      />
      <Select label="Education Level" value={form.Education} onChange={set('Education')} options={EDU_OPTIONS} />
      <Select label="Household Income" value={form.Income} onChange={set('Income')} options={INCOME_OPTIONS} />

      {/* ── Submit ── */}
      <button
        id="analyse-btn"
        className="submit-btn"
        onClick={onSubmit}
        disabled={loading}
      >
        {loading ? 'Analysing…' : '🔬 Analyse Risk'}
      </button>

      <p className="disclaimer">
        All data is processed locally and never stored.
      </p>
    </div>
  );
}
