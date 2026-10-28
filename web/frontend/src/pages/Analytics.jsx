import React from 'react';
import { useNids } from '../context/NidsContext';
import MetricCard from '../components/TaskCard';
import { ImportanceChart, SweepChart } from '../components/Charts';
import { formatPercent } from '../utils/formatters';
import { muted } from '../utils/ui';

export default function Analytics() {
  const { metrics: m } = useNids();
  const v = x => (x != null ? formatPercent(x) : 'n/a');
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={muted}>Model performance is computed from the uploaded CSV and requires a label column. Upload data on the SOC Command Center page.</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
        <MetricCard title="STAGE 1 ACCURACY" value={v(m?.accuracy)} sub="Normal vs Attack" borderColor="#ff6900" />
        <MetricCard title="STAGE 1 RECALL" value={v(m?.recall)} sub="Attacks caught" borderColor="#ef4444" />
        <MetricCard title="SELECTIVE MACRO-F1" value={m?.f1 != null ? m.f1.toFixed(4) : 'n/a'} sub="Stage 2, confident predictions" borderColor="#3b82f6" />
        <MetricCard title="COVERAGE" value={m ? formatPercent(m.coverage) : 'n/a'} sub="Attacks above τ" borderColor="#10b981" />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
        <ImportanceChart /><SweepChart />
      </div>
    </div>
  );
}
