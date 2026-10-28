import React from 'react';
import { UserCheck } from 'lucide-react';
import { useNids } from '../context/NidsContext';

export default function Header() {
  const { apiOnline } = useNids();
  const color = apiOnline ? '#10b981' : apiOnline === false ? '#ef4444' : '#94a3b8';
  return (
    <header style={{ backgroundColor: '#1a2230', borderBottom: '2px solid #ff6900', padding: '12px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span style={{ fontSize: '16px', fontWeight: '900', color: '#fff' }}>3K<span style={{ color: '#ff6900' }}>&gt;</span> enterprise security</span>
        <span style={{ fontSize: '12px', color: '#94a3b8', borderLeft: '1px solid #2a364a', paddingLeft: '12px' }}>UNSW-NB15 Hierarchical NIDS Pipeline</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px', fontSize: '12px' }}>
        <div style={{ color, fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '8px', height: '8px', backgroundColor: color, borderRadius: '50%', display: 'inline-block' }}></span>
          {apiOnline ? 'API ONLINE' : apiOnline === false ? 'API OFFLINE' : 'CONNECTING…'}
        </div>
        <div style={{ backgroundColor: '#141a24', border: '1px solid #2a364a', padding: '6px 12px', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <UserCheck size={14} color="#ff6900" /> SOC Lead Analyst
        </div>
      </div>
    </header>
  );
}
