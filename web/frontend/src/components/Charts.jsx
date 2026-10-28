import React, { useMemo } from 'react';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import { useNids } from '../context/NidsContext';
import { timeline } from '../utils/nidsLogic';
import { AX, PALETTE } from '../utils/chartSetup';
import { card, h3, muted } from '../utils/ui';

const legend = { labels: { color: '#e2e8f0' } };
const Box = ({ title, icon, children, h = 240 }) => (
  <div style={{ ...card, minWidth: 0 }}><h3 style={h3}>{icon}{title}</h3><div style={{ position: 'relative', height: h }}>{children}</div></div>
);
const Empty = ({ text }) => <div style={{ ...muted, display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center' }}>{text}</div>;

export function TimelineChart() {
  const { rows, s1Threshold, confThreshold, selectiveMode } = useNids();
  const t = useMemo(() => timeline(rows, s1Threshold, confThreshold, selectiveMode), [rows, s1Threshold, confThreshold, selectiveMode]);
  const ds = (label, key, borderColor, extra = {}) => ({ label, data: t.map(x => x[key]), borderColor, tension: 0.3, ...extra });
  return (
    <Box title="Network Traffic & Threat Ingestion">
      {t.length ? <Line data={{ labels: t.map(x => x.label), datasets: [ds('Normal', 'normal', '#10b981'), ds('Attacks', 'attack', '#ef4444'), ds('Attack_Unknown', 'unknown', '#f59e0b', { borderDash: [5, 5] })] }}
        options={{ responsive: true, maintainAspectRatio: false, plugins: { legend }, scales: { x: AX, y: AX } }} /> : <Empty text="No data yet" />}
    </Box>
  );
}

export function AttackPie() {
  const { metrics } = useNids();
  const names = metrics ? Object.keys(metrics.counts) : [];
  return (
    <Box title="Multiclass Attack Breakdown">
      {names.length ? <Doughnut data={{ labels: names, datasets: [{ data: names.map(n => metrics.counts[n]), backgroundColor: names.map((_, i) => PALETTE[i % PALETTE.length]) }] }}
        options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', ...legend } } }} /> : <Empty text="No attacks classified" />}
    </Box>
  );
}

export function ImportanceChart() {
  const { importance } = useNids();
  return (
    <Box title="Top Feature Importances">
      {importance.length ? <Bar data={{ labels: importance.map(f => f.name), datasets: [{ data: importance.map(f => f.value), backgroundColor: '#38bdf8' }] }}
        options={{ indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: AX, y: { ...AX, ticks: { color: '#e2e8f0' } } } }} /> : <Empty text="Backend did not return feature importance" />}
    </Box>
  );
}

export function SweepChart() {
  const { sweepData } = useNids();
  return (
    <Box title="Selective F1 vs Coverage (τ sweep)">
      {sweepData.length ? <Line data={{ labels: sweepData.map(x => x.tau), datasets: [
        { label: 'Selective Macro-F1', data: sweepData.map(x => x.f1), borderColor: '#3b82f6', yAxisID: 'y' },
        { label: 'Coverage', data: sweepData.map(x => x.coverage), borderColor: '#ff6900', borderDash: [4, 4], yAxisID: 'y1' }] }}
        options={{ responsive: true, maintainAspectRatio: false, plugins: { legend }, scales: { x: AX, y: { ...AX, position: 'left' }, y1: { position: 'right', ticks: { color: '#ff6900' }, grid: { drawOnChartArea: false } } } }} /> : <Empty text="Needs a label column (true class) in the CSV" />}
    </Box>
  );
}
