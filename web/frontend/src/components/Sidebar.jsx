import React from 'react';
import { NavLink } from 'react-router-dom';
import { ShieldAlert, BarChart3, Sliders, ListFilter } from 'lucide-react';

export default function Sidebar() {
  const linkStyle = ({ isActive }) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '12px 16px',
    borderRadius: '4px',
    color: isActive ? '#fff' : '#94a3b8',
    backgroundColor: isActive ? '#222c3d' : 'transparent',
    textDecoration: 'none',
    fontSize: '13px',
    fontWeight: '600',
    marginBottom: '6px'
  });

  return (
    <aside style={{ width: '260px', backgroundColor: '#141a24', borderRight: '1px solid #2a364a', display: 'flex', flexDirection: 'column', padding: '20px' }}>
      <div style={{ fontSize: '15px', fontWeight: '800', color: '#ff6900', marginBottom: '30px', letterSpacing: '0.5px' }}>
        NETWORK ANALYSIS AI
      </div>
      <nav style={{ display: 'flex', flexDirection: 'column' }}>
        <NavLink to="/" style={linkStyle}><ShieldAlert size={16} /> SOC Command Center</NavLink>
        <NavLink to="/analytics" style={linkStyle}><BarChart3 size={16} /> Model Performance</NavLink>
        <NavLink to="/tuning" style={linkStyle}><Sliders size={16} /> Threshold Tuning</NavLink>
        <NavLink to="/logs" style={linkStyle}><ListFilter size={16} /> Event Stream Logs</NavLink>
      </nav>
    </aside>
  );
}
