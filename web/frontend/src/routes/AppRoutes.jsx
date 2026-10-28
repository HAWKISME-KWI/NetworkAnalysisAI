import React from 'react';
import { Routes, Route } from 'react-router-dom';
import MainLayout from '../layouts/MainLayout';
import PrivateRoute from '../guard/PrivateRoute';
import Dashboard from '../pages/Dashboard';
import AllTasks from '../pages/AllTasks';
import Analytics from '../pages/Analytics';
import Tuning from '../pages/Tuning';
import Logs from '../pages/Logs';

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<PrivateRoute><MainLayout /></PrivateRoute>}>
        <Route index element={<Dashboard />} />
        <Route path="tasks" element={<AllTasks />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="tuning" element={<Tuning />} />
        <Route path="logs" element={<Logs />} />
      </Route>
    </Routes>
  );
}
