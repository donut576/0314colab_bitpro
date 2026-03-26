import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';

/**
 * FeatureBarChart — horizontal bar chart of feature importances.
 * Pure props: { features } where features = [{ feature, importance }]
 */
export default function FeatureBarChart({ features, barColor = '#00f0ff' }) {
  if (!features || features.length === 0) return null;

  const sorted = [...features].sort((a, b) => b.importance - a.importance).slice(0, 15);

  return (
    <ResponsiveContainer width="100%" height={Math.max(240, sorted.length * 28)}>
      <BarChart data={sorted} layout="vertical" margin={{ left: 120, right: 20, top: 4, bottom: 4 }}>
        <XAxis
          type="number"
          domain={[0, 'auto']}
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          axisLine={{ stroke: '#1e293b' }}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'var(--font-mono)' }}
          axisLine={false}
          tickLine={false}
          width={115}
        />
        <Tooltip
          contentStyle={{
            background: '#111827', border: '1px solid #1e293b',
            borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: 11,
          }}
          formatter={(val) => val.toFixed(4)}
          cursor={{ fill: '#ffffff08' }}
        />
        <Bar dataKey="importance" radius={[0, 4, 4, 0]} maxBarSize={18}>
          {sorted.map((_, i) => (
            <Cell key={i} fill={barColor} fillOpacity={1 - i * 0.05} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
