import React, { useState } from 'react';
import ShapDrillDown from './ShapDrillDown';

/**
 * ShapPanel — waterfall-style SHAP visualization.
 * Pure props: { shapData } where shapData = [{ feature, shap_value, direction }]
 */
export default function ShapPanel({ shapData }) {
  const [selected, setSelected] = useState(null);

  if (!shapData || shapData.length === 0) return <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No SHAP data</div>;

  const sorted = [...shapData].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
  const maxVal = Math.max(...sorted.map(d => Math.abs(d.shap_value)));

  return (
    <div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {sorted.map((item) => {
          const pct = (Math.abs(item.shap_value) / maxVal) * 100;
          const isPositive = item.shap_value >= 0;
          const color = isPositive ? 'var(--neon-red)' : 'var(--neon-green)';
          return (
            <div
              key={item.feature}
              style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', padding: '4px 0' }}
              onClick={() => setSelected(selected === item.feature ? null : item.feature)}
              role="button"
              tabIndex={0}
              aria-label={`SHAP detail for ${item.feature}`}
              onKeyDown={e => e.key === 'Enter' && setSelected(selected === item.feature ? null : item.feature)}
            >
              <span style={{ width: 140, fontSize: 11, color: 'var(--text-secondary)', textAlign: 'right', flexShrink: 0 }}>
                {item.feature}
              </span>
              <div style={{ flex: 1, height: 14, background: '#1e293b', borderRadius: 3, overflow: 'hidden', position: 'relative' }}>
                <div style={{
                  position: 'absolute',
                  [isPositive ? 'left' : 'right']: '50%',
                  width: `${pct / 2}%`,
                  height: '100%',
                  background: `linear-gradient(${isPositive ? '90deg' : '270deg'}, ${color}44, ${color})`,
                  borderRadius: 3,
                  boxShadow: `0 0 6px ${color}33`,
                  transition: 'width 0.4s ease',
                }} />
                <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: '#334155' }} />
              </div>
              <span style={{ width: 55, fontSize: 11, color, textAlign: 'right', fontWeight: 600 }}>
                {item.shap_value > 0 ? '+' : ''}{item.shap_value.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
      {selected && <ShapDrillDown feature={selected} shapData={shapData} />}
    </div>
  );
}
