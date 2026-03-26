import React from 'react';

/**
 * ShapDrillDown — detail panel for a selected SHAP feature.
 * Pure props: { feature, shapData }
 */
export default function ShapDrillDown({ feature, shapData }) {
  const item = shapData?.find(d => d.feature === feature);
  if (!item) return null;

  const isPositive = item.shap_value >= 0;
  const color = isPositive ? 'var(--neon-red)' : 'var(--neon-green)';
  const impact = isPositive ? '增加風險' : '降低風險';

  return (
    <div style={{
      marginTop: 12, padding: 14, background: '#0f172a',
      border: '1px solid var(--border)', borderRadius: 'var(--radius)',
      fontSize: 12, color: 'var(--text-secondary)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{feature}</span>
        <span style={{ color, fontWeight: 600 }}>SHAP: {item.shap_value > 0 ? '+' : ''}{item.shap_value.toFixed(4)}</span>
      </div>
      <div style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>
        此特徵對模型預測的影響方向為 <span style={{ color }}>{impact}</span>。
        SHAP 值的絕對值越大，代表該特徵對最終風險分數的貢獻越顯著。
        排名：#{shapData.sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)).findIndex(d => d.feature === feature) + 1}
      </div>
    </div>
  );
}
