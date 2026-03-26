/** Mock data — fallback when API is unavailable */

export const mockMetrics = {
  safe: {
    f1: 0.8234, precision: 0.7891, recall: 0.8612,
    auc: 0.9145, accuracy: 0.9523, threshold: 0.42,
    total_samples: 12480, positive_samples: 1247,
    mode: 'safe',
  },
  no_leak: {
    f1: 0.8567, precision: 0.8123, recall: 0.9056,
    auc: 0.9312, accuracy: 0.9601, threshold: 0.38,
    total_samples: 12480, positive_samples: 1247,
    mode: 'no_leak',
  },
  full: {
    f1: 0.9123, precision: 0.8901, recall: 0.9356,
    auc: 0.9678, accuracy: 0.9789, threshold: 0.35,
    total_samples: 12480, positive_samples: 1247,
    mode: 'full',
  },
};

export const mockFeatures = {
  safe: [
    { feature: 'twd_out_total', importance: 0.142 },
    { feature: 'crypto_unique_chains', importance: 0.118 },
    { feature: 'tx_frequency_7d', importance: 0.105 },
    { feature: 'avg_amount_ratio', importance: 0.098 },
    { feature: 'ip_unique_count', importance: 0.087 },
    { feature: 'fast_in_out_flag', importance: 0.076 },
    { feature: 'network_degree', importance: 0.065 },
    { feature: 'kyc_time_gap_hr', importance: 0.058 },
    { feature: 'swap_volume_30d', importance: 0.049 },
    { feature: 'isolation_score', importance: 0.041 },
  ],
  no_leak: [
    { feature: 'twd_out_total', importance: 0.155 },
    { feature: 'crypto_unique_chains', importance: 0.128 },
    { feature: 'tx_frequency_7d', importance: 0.112 },
    { feature: 'avg_amount_ratio', importance: 0.101 },
    { feature: 'ip_unique_count', importance: 0.092 },
    { feature: 'fast_in_out_flag', importance: 0.081 },
    { feature: 'network_degree', importance: 0.072 },
    { feature: 'kyc_time_gap_hr', importance: 0.063 },
    { feature: 'swap_volume_30d', importance: 0.054 },
    { feature: 'isolation_score', importance: 0.046 },
  ],
  full: [
    { feature: 'twd_out_total', importance: 0.168 },
    { feature: 'crypto_unique_chains', importance: 0.139 },
    { feature: 'tx_frequency_7d', importance: 0.121 },
    { feature: 'avg_amount_ratio', importance: 0.108 },
    { feature: 'ip_unique_count', importance: 0.095 },
    { feature: 'fast_in_out_flag', importance: 0.084 },
    { feature: 'network_degree', importance: 0.073 },
    { feature: 'kyc_time_gap_hr', importance: 0.066 },
    { feature: 'swap_volume_30d', importance: 0.057 },
    { feature: 'isolation_score', importance: 0.048 },
  ],
};

export const mockThresholds = {
  safe: Array.from({ length: 19 }, (_, i) => {
    const t = (i + 1) * 0.05;
    return {
      threshold: +t.toFixed(2),
      precision: +(0.5 + t * 0.45).toFixed(3),
      recall: +(1.0 - t * 0.7).toFixed(3),
      f1: +(2 * (0.5 + t * 0.45) * (1.0 - t * 0.7) / ((0.5 + t * 0.45) + (1.0 - t * 0.7))).toFixed(3),
    };
  }),
  no_leak: Array.from({ length: 19 }, (_, i) => {
    const t = (i + 1) * 0.05;
    return {
      threshold: +t.toFixed(2),
      precision: +(0.55 + t * 0.42).toFixed(3),
      recall: +(1.0 - t * 0.65).toFixed(3),
      f1: +(2 * (0.55 + t * 0.42) * (1.0 - t * 0.65) / ((0.55 + t * 0.42) + (1.0 - t * 0.65))).toFixed(3),
    };
  }),
  full: Array.from({ length: 19 }, (_, i) => {
    const t = (i + 1) * 0.05;
    return {
      threshold: +t.toFixed(2),
      precision: +(0.6 + t * 0.38).toFixed(3),
      recall: +(1.0 - t * 0.55).toFixed(3),
      f1: +(2 * (0.6 + t * 0.38) * (1.0 - t * 0.55) / ((0.6 + t * 0.38) + (1.0 - t * 0.55))).toFixed(3),
    };
  }),
};

export const mockShap = {
  safe: [
    { feature: 'twd_out_total', shap_value: 0.32, direction: 'positive' },
    { feature: 'crypto_unique_chains', shap_value: 0.25, direction: 'positive' },
    { feature: 'tx_frequency_7d', shap_value: 0.18, direction: 'positive' },
    { feature: 'avg_amount_ratio', shap_value: -0.12, direction: 'negative' },
    { feature: 'ip_unique_count', shap_value: 0.15, direction: 'positive' },
    { feature: 'fast_in_out_flag', shap_value: 0.22, direction: 'positive' },
    { feature: 'network_degree', shap_value: -0.08, direction: 'negative' },
    { feature: 'kyc_time_gap_hr', shap_value: 0.11, direction: 'positive' },
  ],
  no_leak: [
    { feature: 'twd_out_total', shap_value: 0.35, direction: 'positive' },
    { feature: 'crypto_unique_chains', shap_value: 0.28, direction: 'positive' },
    { feature: 'tx_frequency_7d', shap_value: 0.21, direction: 'positive' },
    { feature: 'avg_amount_ratio', shap_value: -0.14, direction: 'negative' },
    { feature: 'ip_unique_count', shap_value: 0.17, direction: 'positive' },
    { feature: 'fast_in_out_flag', shap_value: 0.24, direction: 'positive' },
    { feature: 'network_degree', shap_value: -0.09, direction: 'negative' },
    { feature: 'kyc_time_gap_hr', shap_value: 0.13, direction: 'positive' },
  ],
  full: [
    { feature: 'twd_out_total', shap_value: 0.38, direction: 'positive' },
    { feature: 'crypto_unique_chains', shap_value: 0.31, direction: 'positive' },
    { feature: 'tx_frequency_7d', shap_value: 0.24, direction: 'positive' },
    { feature: 'avg_amount_ratio', shap_value: -0.16, direction: 'negative' },
    { feature: 'ip_unique_count', shap_value: 0.19, direction: 'positive' },
    { feature: 'fast_in_out_flag', shap_value: 0.27, direction: 'positive' },
    { feature: 'network_degree', shap_value: -0.11, direction: 'negative' },
    { feature: 'kyc_time_gap_hr', shap_value: 0.15, direction: 'positive' },
  ],
};
