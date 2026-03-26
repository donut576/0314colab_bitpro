import React from 'react';

export default function ScoreBar({ value = 0, max = 1, color = 'var(--neon-cyan)', height = 8 }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ width: '100%', height, background: '#1e293b', borderRadius: height / 2, overflow: 'hidden' }}>
      <div style={{
        width: `${pct}%`, height: '100%', borderRadius: height / 2,
        background: `linear-gradient(90deg, ${color}88, ${color})`,
        boxShadow: `0 0 8px ${color}44`,
        transition: 'width 0.6s ease',
      }} />
    </div>
  );
}
