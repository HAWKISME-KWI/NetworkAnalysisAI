export const formatNumber = (num) => {
  return new Intl.NumberFormat('en-US').format(num);
};

export const formatPercent = (val) => {
  return `${(val * 100).toFixed(2)}%`;
};
