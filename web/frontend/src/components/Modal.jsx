import React from 'react';
import { X, Copy, Code } from 'lucide-react';

export default function Modal({ isOpen, onClose, data }) {
  if (!isOpen || !data) return null;

  return (
    <div style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
      <div style={{ backgroundColor: '#141a24', border: '1px solid #2a364a', borderRadius: '6px', width: '500px', padding: '20px', boxShadow: '0 5px 20px rgba(0,0,0,0.5)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #2a364a', paddingBottom: '10px', marginBottom: '15px' }}>
          <h4 style={{ fontSize: '13px', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
            <Code size={16} color="#ff6900" /> Packet Feature Raw Payload
          </h4>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}><X size={18} /></button>
        </div>
        <pre style={{ backgroundColor: '#0b0e14', border: '1px solid #2a364a', padding: '12px', borderRadius: '4px', color: '#38bdf8', fontFamily: 'monospace', fontSize: '12px', maxHeight: '250px', overflowY: 'auto' }}>
          {JSON.stringify(data, null, 2)}
        </pre>
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '15px' }}>
          <button onClick={() => { navigator.clipboard.writeText(JSON.stringify(data, null, 2)); alert('Copied to clipboard!'); }} style={{ backgroundColor: '#ff6900', color: '#fff', border: 'none', padding: '8px 16px', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '12px' }}>
            <Copy size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Copy JSON
          </button>
        </div>
      </div>
    </div>
  );
}
