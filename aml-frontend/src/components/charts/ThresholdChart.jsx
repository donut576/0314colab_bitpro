import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts';

/**
 * ThresholdChart — precision/recall/F1 curves across thresholds.
 * Pure props: { data, bestThreshold }
 *   data = [{ threshold, precision, recall, f1 }]
 */
export default function ThresholdChart({ data, bestThreshold }) {
  if (!data || data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ left: 10, right: 20, top: 10, bottom: 4 }}>
        <XAxis
          dataKey="threshold"
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          axisLine={{ stroke: '#1e293b' }}
          tickLine={false}
          label={{ value: 'Threshold', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 10 }}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          axisLine={{ stroke: '#1e293b' }}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            background: '#111827', border: '1px solid #1e293b',
            borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: 11,
          }}
          formatter={(val) => val.toFixed(3)}
        />
        <Legend
          wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#94a3b8' }}
        />
        <Line type="monotone" dataKey="precision" stroke="#00f0ff" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="recall" stroke="#00ff88" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="f1" stroke="#a855f7" strokeWidth={2} dot={false} />
        {bestThreshold != null && (
          <ReferenceLine
            x={bestThreshold}
            stroke="#ff3366"
            strokeDasharray="4 4"
            label={{ value: `Best: ${bestThreshold}`, fill: '#ff3366', fontSize: 10, position: 'top' }}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
