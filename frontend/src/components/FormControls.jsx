// Reusable form control components
import React from 'react';

/** Yes/No toggle button pair */
export function YesNoToggle({ label, value, onChange }) {
  return (
    <div className="form-group">
      <label className="form-label">{label}</label>
      <div className="toggle-group">
        <button
          type="button"
          className={`toggle-btn ${value === 0 ? 'active' : ''}`}
          onClick={() => onChange(0)}
        >
          No
        </button>
        <button
          type="button"
          className={`toggle-btn ${value === 1 ? 'active' : ''}`}
          onClick={() => onChange(1)}
        >
          Yes
        </button>
      </div>
    </div>
  );
}

/** Range slider with live value display */
export function Slider({ label, min, max, value, onChange, hint }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="form-group">
      <label className="form-label">
        {label}
        {hint && <span className="badge">{hint}</span>}
      </label>
      <div className="slider-container">
        <div className="slider-track">
          <input
            type="range"
            min={min}
            max={max}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            style={{ '--fill': `${pct}%` }}
          />
          <span className="slider-value">{value}</span>
        </div>
      </div>
    </div>
  );
}

/** Dropdown select */
export function Select({ label, value, onChange, options, hint }) {
  return (
    <div className="form-group">
      <label className="form-label">
        {label}
        {hint && <span className="badge">{hint}</span>}
      </label>
      <select
        className="form-select"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

/** Number input */
export function NumberInput({ label, value, onChange, min, max, step = 0.1, hint }) {
  return (
    <div className="form-group">
      <label className="form-label">
        {label}
        {hint && <span className="badge">{hint}</span>}
      </label>
      <input
        type="number"
        className="form-input"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
