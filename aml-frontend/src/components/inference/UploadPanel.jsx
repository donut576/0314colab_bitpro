import React, { useState, useRef } from 'react';
import { ENDPOINTS } from '../../api/endpoints';
import RiskBadge from '../shared/RiskBadge';
import ScoreBar from '../shared/ScoreBar';

/**
 * UploadPanel — CSV upload for batch inference via POST /infer.
 * Handles file upload, API call, and result display.
 */
export default function UploadPanel({ model = 'xgb', mode = 'safe' }) {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${ENDPOINTS.infer()}?model=${model}&mode=${mode}`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setResults(json);
    } catch (err) {
      setError(err.message);
      // Mock fallback
      setResults({
        predictions: [
          { user_id: 'USR_001', risk_score: 0.92, risk_level: 'HIGH' },
          { user_id: 'USR_002', risk_score: 0.45, risk_level: 'MEDIUM' },
          { user_id: 'USR_003', risk_score: 0.12, risk_level: 'LOW' },
          { user_id: 'USR_004', risk_score: 0.88, risk_level: 'HIGH' },
          { user_id: 'USR_005', risk_score: 0.31, risk_level: 'LOW' },
        ],
        total: 5,
        mode: 'mock',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 16 }}>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          onChange={e => setFile(e.target.files?.[0] || null)}
          style={{ display: 'none' }}
          aria-label="Upload CSV file"
        />
        <button
          onClick={() => inputRef.current?.click()}
          style={{
            fontFamily: 'var(--font-mono)', fontSize: 12, padding: '8px 16px',
            border: '1px solid var(--border)', borderRadius: 'var(--radius)',
            background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer',
          }}
        >
          📁 選擇 CSV
        </button>
        {file && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{file.name}</span>}
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          style={{
            fontFamily: 'var(--font-mono)', fontSize: 12, padding: '8px 18px',
            border: '1px solid var(--neon-cyan)', borderRadius: 'var(--radius)',
            background: loading ? '#00f0ff11' : 'transparent',
            color: 'var(--neon-cyan)', cursor: file && !loading ? 'pointer' : 'not-allowed',
            opacity: file && !loading ? 1 : 0.5,
          }}
        >
          {loading ? '推論中...' : '🚀 執行推論'}
        </button>
      </div>

      {error && <div style={{ fontSize: 11, color: 'var(--neon-yellow)', marginBottom: 8 }}>⚠ API 失敗，顯示 mock 結果</div>}

      {results?.predictions && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <th style={thStyle}>User ID</th>
                <th style={thStyle}>Risk Score</th>
                <th style={thStyle}>Level</th>
                <th style={{ ...thStyle, width: '30%' }}>Score Bar</th>
              </tr>
            </thead>
            <tbody>
              {results.predictions.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #1e293b22' }}>
                  <td style={tdStyle}>{p.user_id}</td>
                  <td style={{ ...tdStyle, color: 'var(--neon-cyan)', fontWeight: 600 }}>{p.risk_score.toFixed(4)}</td>
                  <td style={tdStyle}><RiskBadge level={p.risk_level} /></td>
                  <td style={tdStyle}>
                    <ScoreBar
                      value={p.risk_score}
                      color={p.risk_level === 'HIGH' ? 'var(--neon-red)' : p.risk_level === 'MEDIUM' ? 'var(--neon-yellow)' : 'var(--neon-green)'}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8 }}>
            共 {results.total} 筆 {results.mode === 'mock' ? '(mock)' : ''}
          </div>
        </div>
      )}
    </div>
  );
}

const thStyle = { textAlign: 'left', padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 500, fontSize: 11 };
const tdStyle = { padding: '8px 10px', color: 'var(--text-secondary)' };
