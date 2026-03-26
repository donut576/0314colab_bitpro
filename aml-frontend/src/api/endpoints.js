/** API endpoint definitions */

const BASE = '';

export const ENDPOINTS = {
  metrics:    (mode, model = 'xgb') => `${BASE}/metrics?mode=${mode}&model=${model}`,
  compare:    (mode)                 => `${BASE}/metrics/compare?mode=${mode}`,
  features:   (mode, model = 'xgb') => `${BASE}/features?mode=${mode}&model=${model}`,
  thresholds: (mode, model = 'xgb') => `${BASE}/thresholds?mode=${mode}&model=${model}`,
  shap:       (mode, model = 'xgb') => `${BASE}/shap?mode=${mode}&model=${model}`,
  summary:    ()                     => `${BASE}/summary`,
  infer:      ()                     => `${BASE}/infer`,
};
