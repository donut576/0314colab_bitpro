import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/metrics': 'http://localhost:8000',
      '/features': 'http://localhost:8000',
      '/thresholds': 'http://localhost:8000',
      '/shap': 'http://localhost:8000',
      '/infer': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/cache': 'http://localhost:8000',
    },
  },
});
