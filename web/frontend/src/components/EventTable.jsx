import React, { useMemo, useState } from 'react';
import { ShieldAlert } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useNids } from '../context/NidsContext';
import { classify } from '../utils/nidsLogic';
import { card, h3, muted, STATUS_COLOR } from '../utils/ui';
import Modal from './Modal';

export default function EventTable({ limit = 200, showFilters = true, viewAll = false }) {
  const { rows, s1Threshold, confThreshold, selectiveMode } = useNids();
  const [filter, setFilter] = useState('all');
  const [selected, setSelected] = useState(null);
  const flagged = useMemo(() => rows.map(r => ({ r, c: classify(r, s1Threshold, confThreshold, selectiveMode) })).filter(x => x.c.status !== 'NORMAL'), [rows, s1Threshold, confThreshold, selectiveMode]);
  const classes = useMemo(() => [...new Set(flagged.map(x => x.c.cls))], [flagged]);
  const view = flagged.filter(x => filter === 'all' || x.c.cls === filter).slice(0, limit);
  const td = { padding: '10px' };
  return (
    <div style={card}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <h3 style={h3}><ShieldAlert size={16} /> Flagged Threat Log Stream ({flagged.length})</h3>
        {showFilters && <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '15px' }}>
          {['all', ...classes].map(f => <button key={f} onClick={() => setFilter(f)} style={{ background: filter === f ? '#ff6900' : '#0b0e14', color: filter === f ? '#fff' : '#94a3b8', border: '1px solid #2a364a', borderRadius: '4px', padding: '4px 10px', fontSize: '11px', cursor: 'pointer' }}>{f === 'all' ? 'All' : f}</button>)}
        </div>}
        {viewAll && <Link to="/logs" style={{ ...muted, color: '#38bdf8', marginBottom: '15px' }}>View all →</Link>}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontFamily: 'Consolas, monospace', fontSize: '11px' }}>
          <thead><tr style={{ backgroundColor: '#0b0e14', color: '#94a3b8' }}>
            {['TIMESTAMP', 'SRC IP', 'DST IP', 'PROTO', 'S1 PROB', 'STAGE 2 CLASS', 'CONF', 'STATUS', 'PAYLOAD'].map(h => <th key={h} style={td}>{h}</th>)}
          </tr></thead>
          <tbody>
            {view.length ? view.map(({ r, c }) => (
              <tr key={r.id} style={{ borderBottom: '1px solid #2a364a' }}>
                <td style={td}>{r.time ?? '–'}</td><td style={{ ...td, color: '#38bdf8' }}>{r.srcip ?? '–'}</td><td style={td}>{r.dstip ?? '–'}</td>
                <td style={{ ...td, textTransform: 'uppercase' }}>{r.proto ?? '–'}</td><td style={td}>{r.s1Prob.toFixed(2)}</td>
                <td style={{ ...td, fontWeight: 'bold' }}>{c.cls}</td><td style={td}>{r.conf.toFixed(2)}</td>
                <td style={td}><span style={{ padding: '2px 6px', borderRadius: '3px', fontSize: '10px', fontWeight: 'bold', color: STATUS_COLOR[c.status], backgroundColor: STATUS_COLOR[c.status] + '33' }}>{c.status}</span></td>
                <td style={td}><button onClick={() => setSelected({ ...r, decision: c })} style={{ background: '#0b0e14', border: '1px solid #2a364a', color: '#e2e8f0', padding: '4px 8px', borderRadius: '3px', cursor: 'pointer' }}>Inspect</button></td>
              </tr>
            )) : <tr><td colSpan={9} style={{ ...td, textAlign: 'center', ...muted }}>{rows.length ? 'No events match.' : 'No data. Upload a CSV or load demo data.'}</td></tr>}
          </tbody>
        </table>
      </div>
      <Modal isOpen={!!selected} onClose={() => setSelected(null)} data={selected} />
    </div>
  );
}
