import React from 'react';
import { Navigate } from 'react-router-dom';

export default function PrivateRoute({ children }) {
  const isAuthorized = true; // SOC Analyst session active
  return isAuthorized ? children : <Navigate to="/login" replace />;
}
