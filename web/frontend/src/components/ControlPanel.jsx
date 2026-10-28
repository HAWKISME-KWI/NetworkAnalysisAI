import React from 'react';
import { Sliders } from 'lucide-react';
import { useNids } from '../context/NidsContext';
import { card, h3 } from '../utils/ui';

export default function ControlPanel() {
  const { s1Threshold, setS1Threshold, confThreshold, setConfThreshold, selectiveMode, setSelectiveMode } = useNids();
  const label = { fontSize: '12px', fontWeight: '600', display: 'flex', justifyContent: 'space-between' };
  const val = { color: '#ff6900', fontFamily: 'monospace' };
  return (
    <div style={card}>
      <h3 style={h3}><Sliders size={16} /> Hierarchical NIDS Model Tuning & Thresholds</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', alignItems: 'center' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <label style={label}><span>Stage 1 Binary Threshold:</span><span style={val}>{s1Threshold.toFixed(2)}</span></label>
          <input type="range" min="0.10" max="0.90" step="0.05" value={s1Threshold} onChange={e => setS1Threshold(parseFloat(e.target.value))} style={{ accentColor: '#ff6900' }} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <label style={label}><span>Stage 2 Confidence Cutoff (&tau;):</span><span style={val}>{confThreshold.toFixed(2)}</span></label>
          <input type="range" min="0.30" max="0.90" step="0.05" value={confThreshold} onChange={e => setConfThreshold(parseFloat(e.target.value))} style={{ accentColor: '#ff6900' }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#0b0e14', border: '1px solid #2a364a', padding: '10px 14px', borderRadius: '4px' }}>
          <span style={{ fontSize: '12px', fontWeight: '600' }}>Selective Mode:</span>
          <button onClick={() => setSelectiveMode(!selectiveMode)} style={{ backgroundColor: selectiveMode ? 'rgba(16,185,129,0.2)' : '#334155', color: selectiveMode ? '#10b981' : '#cbd5e1', border: 'none', padding: '4px 10px', borderRadius: '4px', fontWeight: 'bold', fontSize: '11px', cursor: 'pointer' }}>
            {selectiveMode ? 'ENABLED' : 'DISABLED'}
          </button>
        </div>
      </div>
    </div>
  );
}
