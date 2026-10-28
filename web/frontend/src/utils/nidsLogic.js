export const NORMAL = 'Normal';

// Áp dụng 2 ngưỡng của mô hình phân tầng lên một sự kiện (chạy ngay trên trình duyệt)
export function classify(r, s1, tau, selective) {
  if (r.s1Prob < s1) return { cls: NORMAL, status: 'NORMAL' };
  if (selective && r.conf < tau) return { cls: 'Attack_Unknown', status: 'UNKNOWN' };
  return { cls: r.s2Class, status: 'ATTACK' };
}

export function macroF1(pairs) { // pairs: [nhãn thật, nhãn dự đoán]
  const labels = [...new Set(pairs.map(p => p[0]))];
  if (!labels.length) return 0;
  return labels.reduce((sum, c) => {
    const tp = pairs.filter(p => p[0] === c && p[1] === c).length;
    const fp = pairs.filter(p => p[0] !== c && p[1] === c).length;
    const fn = pairs.filter(p => p[0] === c && p[1] !== c).length;
    return sum + (tp ? (2 * tp) / (2 * tp + fp + fn) : 0);
  }, 0) / labels.length;
}

export function computeStats(rows, s1, tau, selective) {
  const o = { total: rows.length, attacks: 0, unknown: 0, counts: {}, labeled: 0, correct: 0, tp: 0, fn: 0, pairs: [] };
  for (const r of rows) {
    const c = classify(r, s1, tau, selective);
    if (c.status !== 'NORMAL') {
      o.attacks++;
      if (c.status === 'UNKNOWN') o.unknown++; else o.counts[c.cls] = (o.counts[c.cls] || 0) + 1;
    }
    if (r.trueClass != null) {
      o.labeled++;
      const ta = r.trueClass !== NORMAL, pa = c.status !== 'NORMAL';
      if (ta === pa) o.correct++;
      if (ta && pa) o.tp++;
      if (ta && !pa) o.fn++;
      if (ta && c.status === 'ATTACK') o.pairs.push([r.trueClass, c.cls]);
    }
  }
  o.coverage = o.attacks ? (o.attacks - o.unknown) / o.attacks : 0;
  o.accuracy = o.labeled ? o.correct / o.labeled : null;
  o.recall = o.labeled ? o.tp / Math.max(o.tp + o.fn, 1) : null;
  o.f1 = o.pairs.length ? macroF1(o.pairs) : null;
  return o;
}

// Đường cong F1 / Coverage theo tau (cần nhãn thật)
export function sweep(rows, s1) {
  const out = [];
  for (let t = 0.3; t <= 0.951; t += 0.05) {
    const s = computeStats(rows, s1, t, true);
    out.push({ tau: t.toFixed(2), f1: s.f1 ?? 0, coverage: s.coverage });
  }
  return out;
}

// Gộp thành n nhóm liên tiếp cho biểu đồ thời gian
export function timeline(rows, s1, tau, selective, n = 12) {
  const size = Math.max(1, Math.ceil(rows.length / n)), out = [];
  for (let i = 0; i < rows.length; i += size) {
    const g = rows.slice(i, i + size); let a = 0, u = 0;
    g.forEach(r => { const c = classify(r, s1, tau, selective); if (c.status === 'ATTACK') a++; if (c.status === 'UNKNOWN') u++; });
    out.push({ label: g[0].time ? String(g[0].time).slice(11, 19) || `#${i + 1}` : `#${i + 1}`, normal: g.length - a - u, attack: a, unknown: u });
  }
  return out;
}
