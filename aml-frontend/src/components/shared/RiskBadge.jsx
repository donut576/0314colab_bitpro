import React from 'react';

const LEVELS = {
  HIGH:   { color: 'var(--neon-red)',    bg: '#ff336611', border: '#ff336633' },
  MEDIUM: { color: 'var(--neon-yellow)', bg: '#ffcc0011', border: '#ffcc0033' },
  LOW:    { color: 'var(--neon-green)',  bg: '#00ff8811', border: '#00ff8833' },
};

export default function RiskBadge({ level = 'LOW' }) {
  const s = LEVELS[level] || LEVELS.LOW;
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, padding: '3px 10px',
      borderRadius: 4, color: s.color, background: s.bg,
      border: `1px solid ${s.border}`, letterSpacing: 1,
    }}>
      {level}
    </span>
  );
}
