import React from 'react';
import { useNids } from '../context/NidsContext';
import MetricCard from '../components/TaskCard';
import UploadPanel from '../components/UploadPanel';
import ControlPanel from '../components/ControlPanel';
import EventTable from '../components/EventTable';
import { TimelineChart, AttackPie } from '../components/Charts';
import { formatNumber, formatPercent } from '../utils/formatters';

export default function Dashboard() {
  const { metrics: m } = useNids();
  const dash = '–';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <UploadPanel />
      <ControlPanel />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
        <MetricCard title="TOTAL EVENTS INGESTED" value={m ? formatNumber(m.total) : dash} sub="from uploaded file" borderColor="#3b82f6" />
        <MetricCard title="ATTACKS DETECTED" value={m ? formatNumber(m.attacks) : dash} sub={m ? `${formatPercent(m.attacks / m.total)} of traffic` : dash} borderColor="#ef4444" />
        <MetricCard title="ATTACK_UNKNOWN (SOC)" value={m ? formatNumber(m.unknown) : dash} sub={m ? `${formatPercent(m.unknown / Math.max(m.attacks, 1))} of attacks below cutoff` : dash} borderColor="#f59e0b" />
        <MetricCard title="STAGE 1 ACCURACY" value={m?.accuracy != null ? formatPercent(m.accuracy) : 'n/a'} sub={m?.recall != null ? `Recall: ${formatPercent(m.recall)}` : 'No label column'} borderColor="#ff6900" />
        <MetricCard title="SELECTIVE MACRO-F1" value={m?.f1 != null ? m.f1.toFixed(4) : 'n/a'} sub={m ? `Coverage: ${formatPercent(m.coverage)}` : dash} borderColor="#10b981" />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
        <TimelineChart /><AttackPie />
      </div>
      <EventTable limit={8} showFilters={false} viewAll />
    </div>
  );
}
