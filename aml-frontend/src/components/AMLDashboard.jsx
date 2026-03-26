import React, { useState } from 'react';
import { useApi } from '../hooks/useApi';
import { ENDPOINTS } from '../api/endpoints';
import { mockMetrics, mockFeatures, mockThresholds, mockShap } from '../data/mock';

import GlowCard from './shared/GlowCard';
import MetricBadge from './shared/MetricBadge';
import ApiPill from './shared/ApiPill';
import ExportButton from './shared/ExportButton';
import RadarChart from './charts/RadarChart';
import FeatureBarChart from './charts/FeatureBarChart';
import ThresholdChart from './charts/ThresholdChart';
import ModelCompareChart from './charts/ModelCompareChart';
import ShapPanel from './shap/ShapPanel';
import UploadPanel from './inference/UploadPanel';

const MODES   = ['safe', 'no_leak', 'full'];
const MODELS  = [
  { id: 'xgb', label: 'XGBoost' },
  { id: 'lgb', label: 'LightGBM' },
  { id: 'rf',  label: 'Random Forest' },
];

const btnBase = {
  fontFamily: 'var(--font-mono)', fontSize: 12, padding: '6px 14px',
  border: '1px solid var(--border)', borderRadius: 'var(--radius)',
  background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer', transition: 'all 0.2s',
};
const btnActive = { borderColor: 'var(--neon-cyan)', color: 'var(--neon-cyan)', background: '#00f0ff0d' };

export default function AMLDashboard() {
  const [mode,  setMode]  = useState('safe');
  const [model, setModel] = useState('xgb');

  const metrics   = useApi(ENDPOINTS.metrics(mode, model),   mockMetrics[mode]);
  const features  = useApi(ENDPOINTS.features(mode, model),  mockFeatures[mode]);
  const thresholds = useApi(ENDPOINTS.thresholds(mode, model), mockThresholds[mode]);
  const shap      = useApi(ENDPOINTS.shap(mode, model),      mockShap[mode]);
  const compare   = useApi(ENDPOINTS.compare(mode),          null);

  const noResults = metrics.data?.error;

  return (
    <div className="dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div>
          <div className="dashboard-title">AML FRAUD DETECTION</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
            BitoPro · XGBoost / LightGBM / Random Forest · SHAP Explainability
          </div>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          {/* Mode selector */}
          <div style={{ display: 'flex', gap: 6 }}>
            {MODES.map(m => (
              <button key={m} style={{ ...btnBase, ...(m === mode ? btnActive : {}) }} onClick={() => setMode(m)}>{m}</button>
            ))}
          </div>
          {/* Model selector */}
          <div style={{ display: 'flex', gap: 6 }}>
            {MODELS.map(m => (
              <button key={m.id} style={{ ...btnBase, ...(m.id === model ? { ...btnActive, borderColor: 'var(--neon-purple)', color: 'var(--neon-purple)', background: '#a855f711' } : {}) }}
                onClick={() => setModel(m.id)}>{m.label}</button>
            ))}
          </div>
        </div>
      </div>

      {/* No results banner */}
      {noResults && (
        <div style={{ padding: '16px 20px', background: '#ff336611', border: '1px solid #ff336633', borderRadius: 8, marginBottom: 20, fontSize: 13, color: 'var(--neon-yellow)' }}>
          ⚠ 尚未找到模型結果。請先執行：
          <code style={{ marginLeft: 8, color: 'var(--neon-cyan)', background: '#00f0ff11', padding: '2px 8px', borderRadius: 4 }}>
            python feature_engineering.py
          </code>
          <span style={{ margin: '0 6px' }}>→</span>
          <code style={{ color: 'var(--neon-cyan)', background: '#00f0ff11', padding: '2px 8px', borderRadius: 4 }}>
            python run_all_models.py --no-optuna
          </code>
        </div>
      )}

      {/* Metrics row */}
      <div className="section-title">Model Performance — {MODELS.find(m => m.id === model)?.label} / {mode}</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <ApiPill loading={metrics.loading} error={metrics.error} isMock={metrics.isMock} />
        {metrics.data && !noResults && (
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            threshold={metrics.data.threshold} · features={metrics.data.n_features}
          </span>
        )}
      </div>
      <div className="grid-4" style={{ marginBottom: 24 }}>
        <GlowCard><MetricBadge label="F1 Score"  value={metrics.data?.f1}        color="var(--neon-cyan)"   /></GlowCard>
        <GlowCard><MetricBadge label="Precision" value={metrics.data?.precision} color="var(--neon-green)"  /></GlowCard>
        <GlowCard><MetricBadge label="Recall"    value={metrics.data?.recall}    color="var(--neon-purple)" /></GlowCard>
        <GlowCard><MetricBadge label="AUC-ROC"   value={metrics.data?.auc}       color="var(--neon-orange)" /></GlowCard>
      </div>

      {/* Model comparison + Radar */}
      <div className="grid-2">
        <GlowCard title="Model Comparison (F1 / AUC / Precision / Recall)">
          <ApiPill loading={compare.loading} error={compare.error} isMock={compare.isMock} />
          <ModelCompareChart data={compare.data} />
        </GlowCard>
        <GlowCard title="Radar Overview">
          <RadarChart metrics={metrics.data} />
        </GlowCard>
      </div>

      {/* Feature Importance + Threshold */}
      <div className="grid-2">
        <GlowCard title="Feature Importance (Top 20)">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <ApiPill loading={features.loading} error={features.error} isMock={features.isMock} />
            <ExportButton data={features.data} filename={`features_${model}_${mode}.csv`} />
          </div>
          <FeatureBarChart features={features.data} />
        </GlowCard>
        <GlowCard title="Threshold Analysis">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <ApiPill loading={thresholds.loading} error={thresholds.error} isMock={thresholds.isMock} />
            <ExportButton data={thresholds.data} filename={`thresholds_${model}_${mode}.csv`} />
          </div>
          <ThresholdChart data={thresholds.data} bestThreshold={metrics.data?.threshold} />
        </GlowCard>
      </div>

      {/* SHAP */}
      <GlowCard title="SHAP Explainability — Feature Contributions">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <ApiPill loading={shap.loading} error={shap.error} isMock={shap.isMock} />
          <ExportButton data={shap.data} filename={`shap_${model}_${mode}.csv`} />
        </div>
        <ShapPanel shapData={shap.data} />
      </GlowCard>

      {/* Inference */}
      <div style={{ marginTop: 20 }}>
        <GlowCard title="Batch Inference — Upload CSV">
          <UploadPanel model={model} mode={mode} />
        </GlowCard>
      </div>
    </div>
  );
}
