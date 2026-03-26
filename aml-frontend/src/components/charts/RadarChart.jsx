import React from 'react';
import {
  Radar, RadarChart as RechartsRadar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip,
} from 'recharts';

/**
 * RadarChart — displays model metrics as a radar polygon.
 * Pure props: { metrics } where metrics = { f1, precision, recall, auc, accuracy }
 */
export default function RadarChart({ metrics }) {
  if (!metrics) return null;

  const data = [
    { metric: 'F1', value: metrics.f1 || 0 },
    { metric: 'Precision', value: metrics.precision || 0 },
    { metric: 'Recall', value: metrics.recall || 0 },
    { metric: 'AUC', value: metrics.auc || 0 },
    { metric: 'Accuracy', value: metrics.accuracy || 0 },
  ];

  return (
    <ResponsiveContainer width="100%" height={280}>
      <RechartsRadar data={data} cx="50%" cy="50%" outerRadius="75%">
        <PolarGrid stroke="#1e293b" />
        <PolarAngleAxis
          dataKey="metric"
          tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'var(--font-mono)' }}
        />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 1]}
          tick={{ fill: '#64748b', fontSize: 9 }}
          tickCount={5}
        />
        <Radar
          dataKey="value"
          stroke="#00f0ff"
          fill="#00f0ff"
          fillOpacity={0.15}
          strokeWidth={2}
        />
        <Tooltip
          contentStyle={{
            background: '#111827', border: '1px solid #1e293b',
            borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: 11,
          }}
          formatter={(val) => val.toFixed(4)}
        />
      </RechartsRadar>
    </ResponsiveContainer>
  );
}
