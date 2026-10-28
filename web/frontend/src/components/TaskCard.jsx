import React from 'react';

export default function TaskCard({ title, value, sub, borderColor = '#3b82f6' }) {
  return (
    <div style={{ backgroundColor: '#141a24', border: '1px solid #2a364a', borderTop: `4px solid ${borderColor}`, borderRadius: '6px', padding: '16px' }}>
      <div style={{ fontSize: '11px', fontWeight: 'bold', color: '#94a3b8', letterSpacing: '0.5px' }}>{title}</div>
      <div style={{ fontSize: '24px', fontWeight: 'bold', fontFamily: 'Consolas, monospace', margin: '8px 0 4px 0', color: '#fff' }}>{value}</div>
      <div style={{ fontSize: '11px', color: '#94a3b8' }}>{sub}</div>
    </div>
  );
}
