import { useState, useEffect, useCallback } from 'react';

/**
 * useApi — Layer 1 hook for all data fetching.
 * Handles loading, error, and mock fallback.
 *
 * @param {string} url        — API endpoint URL
 * @param {*}      mockData   — fallback data when API fails
 * @param {object} [options]  — { enabled, interval }
 */
export function useApi(url, mockData, options = {}) {
  const { enabled = true, interval = null } = options;
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isMock, setIsMock] = useState(false);

  const fetchData = useCallback(async () => {
    if (!enabled || !url) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setIsMock(false);
    } catch (err) {
      setError(err.message);
      if (mockData !== undefined) {
        setData(mockData);
        setIsMock(true);
      }
    } finally {
      setLoading(false);
    }
  }, [url, mockData, enabled]);

  useEffect(() => {
    fetchData();
    if (interval && interval > 0) {
      const id = setInterval(fetchData, interval);
      return () => clearInterval(id);
    }
  }, [fetchData, interval]);

  return { data, loading, error, isMock, refetch: fetchData };
}
