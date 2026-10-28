import { API_BASE_URL } from '../config/apiConfig';

/* Backend cần có:
   GET  /models  -> [{ "name": "xgb_hier.joblib" }]
   POST /predict (form-data: file, model) -> { "rows": [...], "importance": [{ "name": "sttl", "value": 0.13 }] }
   Mỗi row: { id, time, srcip, dstip, sport, dport, proto, s1Prob, s2Class, conf, trueClass?, payload? }
   (chấp nhận cả snake_case: s1_prob, s2_class, s2_conf, true_class, timestamp)
   Với dòng Stage 1 cho là Normal, vẫn nên trả s2Class/conf thật để thanh trượt Stage 1 hoạt động. */

const normalize = (r, i) => ({
  id: r.id ?? `EVT-${i + 1}`,
  time: r.time ?? r.timestamp ?? null,
  srcip: r.srcip, dstip: r.dstip, sport: r.sport, dport: r.dport, proto: r.proto,
  s1Prob: Number(r.s1Prob ?? r.s1_prob),
  s2Class: r.s2Class ?? r.s2_class,
  conf: Number(r.conf ?? r.s2_conf),
  trueClass: r.trueClass ?? r.true_class ?? null,
  payload: r.payload ?? null,
});

async function request(path, options) {
  const res = await fetch(`${API_BASE_URL}${path}`, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.error || `Lỗi ${res.status}`);
  return data;
}

export const nidsService = {
  getModels: () => request('/models'),
  async predict(file, model) {
    const fd = new FormData();
    fd.append('file', file); fd.append('model', model);
    const d = await request('/predict', { method: 'POST', body: fd });
    return { rows: (d.rows || []).map(normalize), importance: (d.importance || []).slice(0, 10) };
  },
  // Dữ liệu minh hoạ để xem giao diện khi chưa có backend
  demo() {
    const cls = ['Exploitation', 'Generic', 'DoS', 'Reconnaissance', 'Backdoor'], rnd = Math.random;
    const rows = Array.from({ length: 600 }, (_, i) => {
      const attack = rnd() < 0.35, t = cls[Math.floor(rnd() * cls.length)];
      return normalize({
        id: `DEMO-${i + 1}`, time: new Date(Date.now() - (600 - i) * 60000).toISOString().replace('T', ' ').slice(0, 19),
        srcip: `175.45.176.${Math.floor(rnd() * 20)}`, dstip: '149.171.126.18', dport: [80, 443, 22, 53][Math.floor(rnd() * 4)], proto: rnd() < 0.7 ? 'tcp' : 'udp',
        s1Prob: attack ? 0.5 + rnd() * 0.5 : rnd() * 0.55, s2Class: rnd() < 0.8 ? t : cls[Math.floor(rnd() * cls.length)],
        conf: 0.45 + rnd() * 0.55, trueClass: attack ? t : 'Normal',
      }, i);
    });
    const importance = [['sttl', .134], ['sload', .102], ['dload', .086], ['ct_state_ttl', .084], ['sbytes', .067], ['dbytes', .059], ['proto', .058], ['service', .044]].map(([name, value]) => ({ name, value }));
    return { rows, importance };
  },
};
