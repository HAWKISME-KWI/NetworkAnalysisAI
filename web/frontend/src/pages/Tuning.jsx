import React from 'react';
import ControlPanel from '../components/ControlPanel';
import { SweepChart } from '../components/Charts';

export default function Tuning() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <ControlPanel />
      <SweepChart />
    </div>
  );
}
