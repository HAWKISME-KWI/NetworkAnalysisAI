import React, { useState } from 'react';
import { Upload } from 'lucide-react';
import { useNids } from '../context/NidsContext';
import { card, h3, btn, muted } from '../utils/ui';

export default function UploadPanel() {
  const { models, model, setModel, analyze, loadDemo, loading, error, source, apiOnline } = useNids();
  const [file, setFile] = useState(null);
  return (
    <div style={card}>
      <h3 style={h3}><Upload size={16} /> Data Source</h3>
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
        <label style={{ border: '1px dashed #2a364a', borderRadius: '4px', padding: '8px 14px', fontSize: '12px', cursor: 'pointer' }}>
          {file ? file.name : 'Choose CSV file'}
          <input type="file" accept=".csv" hidden onChange={e => setFile(e.target.files[0] || null)} />
        </label>
        <select value={model} onChange={e => setModel(e.target.value)} style={{ backgroundColor: '#0b0e14', color: '#e2e8f0', border: '1px solid #2a364a', borderRadius: '4px', padding: '7px' }}>
          {models.length ? models.map(m => <option key={m.name}>{m.name}</option>) : <option>No model</option>}
        </select>
        <button style={{ ...btn, opacity: file && model && !loading ? 1 : 0.45 }} disabled={!file || !model || loading} onClick={() => analyze(file)}>
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
        <button style={{ ...btn, backgroundColor: '#222c3d' }} disabled={loading} onClick={loadDemo}>Load demo data</button>
        <span style={{ ...muted, color: error ? '#ef4444' : '#94a3b8' }}>
          {error || (source ? `Source: ${source}` : apiOnline === false ? 'Backend offline — you can still try the demo data.' : 'Upload a traffic CSV to start.')}
        </span>
      </div>
    </div>
  );
}
