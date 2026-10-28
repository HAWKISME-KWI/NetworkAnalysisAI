import React, { createContext, useState, useEffect, useContext, useMemo, useCallback } from 'react';
import { nidsService } from '../services/nidsService';
import { computeStats, sweep } from '../utils/nidsLogic';

const NidsContext = createContext();

export const NidsProvider = ({ children }) => {
  const [s1Threshold, setS1Threshold] = useState(0.5);
  const [confThreshold, setConfThreshold] = useState(0.8);
  const [selectiveMode, setSelectiveMode] = useState(true);
  const [rows, setRows] = useState([]);
  const [importance, setImportance] = useState([]);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState('');
  const [apiOnline, setApiOnline] = useState(null); // null = đang kiểm tra
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [source, setSource] = useState('');

  useEffect(() => {
    nidsService.getModels()
      .then(list => { setModels(list); setModel(list[0]?.name || ''); setApiOnline(true); })
      .catch(() => setApiOnline(false));
  }, []);

  const load = useCallback(async (fn, label) => {
    setLoading(true); setError('');
    try { const d = await fn(); setRows(d.rows); setImportance(d.importance); setSource(label); }
    catch (e) { setError(e.message || 'Không kết nối được backend'); }
    setLoading(false);
  }, []);
  const analyze = useCallback((file) => load(() => nidsService.predict(file, model), file.name), [load, model]);
  const loadDemo = useCallback(() => load(async () => nidsService.demo(), 'Dữ liệu minh hoạ'), [load]);

  // Kéo thanh trượt chỉ tính lại ở client, không gọi lại API
  const metrics = useMemo(() => (rows.length ? computeStats(rows, s1Threshold, confThreshold, selectiveMode) : null), [rows, s1Threshold, confThreshold, selectiveMode]);
  const sweepData = useMemo(() => (metrics && metrics.labeled ? sweep(rows, s1Threshold) : []), [rows, s1Threshold, metrics]);

  return (
    <NidsContext.Provider value={{
      s1Threshold, setS1Threshold, confThreshold, setConfThreshold, selectiveMode, setSelectiveMode,
      rows, importance, models, model, setModel, apiOnline, loading, error, source, analyze, loadDemo, metrics, sweepData,
    }}>
      {children}
    </NidsContext.Provider>
  );
};

export const useNids = () => useContext(NidsContext);
