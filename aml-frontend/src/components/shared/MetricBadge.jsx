import React from 'react';
import { useCountUp } from '../../hooks/useCountUp';

export default function MetricBadge({ label, value, color = 'var(--neon-cyan)', decimals = 4 }) {
  const animated = useCountUp(value, 800, decimals);
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 700, color, textShadow: `0 0 10px ${color}44` }}>{animated}</div>
    </div>
  );
}
