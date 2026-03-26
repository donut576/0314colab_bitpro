import React from 'react';

const style = {
  background: 'var(--bg-card)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-lg)',
  padding: '20px',
  transition: 'border-color 0.2s, box-shadow 0.2s',
};

const hoverStyle = {
  borderColor: '#00f0ff44',
  boxShadow: 'var(--glow-cyan)',
};

export default function GlowCard({ children, className = '', title }) {
  const [hovered, setHovered] = React.useState(false);
  return (
    <div
      className={className}
      style={{ ...style, ...(hovered ? hoverStyle : {}) }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {title && <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 2, marginBottom: 12 }}>{title}</div>}
      {children}
    </div>
  );
}
