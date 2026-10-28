import React from 'react';
import { useNids } from '../context/NidsContext';

export default function AllTasks() {
  const { metrics } = useNids();
  return (
    <div style={{ backgroundColor: '#141a24', border: '1px solid #2a364a', borderRadius: '6px', padding: '20px' }}>
      <h3 style={{ fontSize: '16px', color: '#ff6900', marginBottom: '15px' }}>Model Performance & Evaluation</h3>
      <p style={{ color: '#94a3b8', fontSize: '13px', lineHeight: '1.6' }}>
        Detailed performance metrics for the UNSW-NB15 hierarchical network intrusion detection model. 
        Stage 1 binary classification achieves robust attack recall (94.94%), while Stage 2 multiclass model utilizes confidence gating ($	au$) to maintain high precision.
      </p>
    </div>
  );
}
