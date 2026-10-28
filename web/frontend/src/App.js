import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import AppRoutes from './routes/AppRoutes';
import { NidsProvider } from './context/NidsContext';

function App() {
  return (
    <BrowserRouter>
      <NidsProvider>
        <AppRoutes />
      </NidsProvider>
    </BrowserRouter>
  );
}

export default App;
