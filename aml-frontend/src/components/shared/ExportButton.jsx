import React from 'react';
import { exportCsv } from '../../utils/csv';

const btnStyle = {
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  padding: '5px 12px',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius)',
  background: 'transparent',
  color: 'var(--text-secondary)',
  cursor: 'pointer',
  transition: 'all 0.2s',
};

export default function ExportButton({ data, filename = 'export.csv', label = '⬇ CSV' }) {
  const handleClick = () => exportCsv(data, filename);
  return (
    <button
      style={btnStyle}
      onClick={handleClick}
      disabled={!data || data.length === 0}
      onMouseEnter={e => { e.target.style.borderColor = 'var(--neon-cyan)'; e.target.style.color = 'var(--neon-cyan)'; }}
      onMouseLeave={e => { e.target.style.borderColor = 'var(--border)'; e.target.style.color = 'var(--text-secondary)'; }}
      aria-label={`Export ${filename}`}
    >
      {label}
    </button>
  );
}
