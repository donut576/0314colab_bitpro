import React from 'react';

export default function ApiPill({ loading, error, isMock }) {
  if (loading) return <span className="loading-pulse" style={{ fontSize: 10, color: 'var(--text-muted)' }}>LOADING...</span>;
  if (isMock) return <span className="mock-badge">MOCK</span>;
  if (error) return <span className="error-text" style={{ fontSize: 10 }}>ERR</span>;
  return <span style={{ fontSize: 10, color: 'var(--neon-green)' }}>● LIVE</span>;
}
