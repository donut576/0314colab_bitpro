import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, Cell,
} from 'recharts';

const MODEL_COLORS = { xgb: '#00f0ff', lgb: '#00ff88', rf: '#a855f7' };
const METRICS = ['f1', 'auc', 'precision', 'recall'];

/**
 * ModelCompareChart — grouped bar chart comparing XGBoost / LightGBM / RF.
 * Pure props: { data } = [{ model, f1, auc, precision, recall }]
 */
export default function ModelCompareChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
        Run models to see comparison
      </div>
    );
  }

  // Reshape: one row per metric, columns per model
  const chartData = METRICS.map(metric => {
    const row = { metric: metric.toUpperCase() };
    data.forEach(d => { row[d.model] = d[metric]; });
    return row;
  });

  const models = data.map(d => d.model);

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={chartData} margin={{ left: 0, right: 10, top: 10, bottom: 4 }}>
        <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'var(--font-mono)' }} axisLine={{ stroke: '#1e293b' }} tickLine={false} />
        <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }} axisLine={{ stroke: '#1e293b' }} tickLine={false} />
        <Tooltip
          contentStyle={{ background: '#111827', border: '1px solid #1e293b', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: 11 }}
          formatter={(val) => val?.toFixed(4)}
          cursor={{ fill: '#ffffff06' }}
        />
        <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#94a3b8' }} />
        {models.map(m => (
          <Bar key={m} dataKey={m} fill={MODEL_COLORS[m] || '#888'} radius={[3, 3, 0, 0]} maxBarSize={28} fillOpacity={0.85} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
