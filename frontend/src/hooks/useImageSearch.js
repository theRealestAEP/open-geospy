import { useState } from 'react';

import { formatRetrievalModelLabel } from '../lib/format';

/**
 * Image-search state machine. Lives at App level so search results persist
 * while the user navigates between pages.
 */
export default function useImageSearch({
  retrievalFile,
  searchEmbeddingBase,
  retrievalMinSimilarity,
  clearSearchFallbackMarker,
  focusRetrievalResult,
  loadRetrievalStats
}) {
  const [retrievalSearchTopK, setRetrievalSearchTopK] = useState(12);
  const [retrievalStatus, setRetrievalStatus] = useState('');
  const [retrievalBusy, setRetrievalBusy] = useState(false);
  const [retrievalResults, setRetrievalResults] = useState([]);
  const [topSearchCaptureId, setTopSearchCaptureId] = useState(null);

  const runImageSearch = async () => {
    if (!retrievalFile) {
      setRetrievalStatus('Pick a reference image first.');
      return;
    }
    setRetrievalBusy(true);
    clearSearchFallbackMarker();
    setRetrievalStatus('Searching similar captures...');
    try {
      const formData = new FormData();
      formData.append('image', retrievalFile);
      formData.append('top_k', String(Math.max(1, Number(retrievalSearchTopK) || 12)));
      formData.append('embedding_base', searchEmbeddingBase);
      const minSimilarityRaw = String(retrievalMinSimilarity ?? '').trim();
      if (minSimilarityRaw !== '') {
        const parsedSimilarity = Number(minSimilarityRaw);
        if (!Number.isNaN(parsedSimilarity)) {
          formData.append('min_similarity', String(parsedSimilarity));
        }
      }
      const res = await fetch('/api/retrieval/search-by-image', {
        method: 'POST',
        body: formData
      });
      const body = await res.json();
      if (!res.ok) {
        setRetrievalStatus(`Search failed: ${body?.detail || 'unknown error'}`);
        setRetrievalResults([]);
        setTopSearchCaptureId(null);
        return;
      }
      const matches = Array.isArray(body.matches) ? body.matches : [];
      setRetrievalResults(matches);
      setTopSearchCaptureId(matches.length ? Number(matches[0].capture_id) : null);
      if (matches.length) {
        await focusRetrievalResult(matches[0]);
      }
      const modelLabel = formatRetrievalModelLabel(body.model_name, body.model_version);
      setRetrievalStatus(
        `Found ${matches.length} matches${modelLabel ? ` (${modelLabel})` : ''}.`
      );
      await loadRetrievalStats();
    } catch (error) {
      setRetrievalStatus(`Search failed: ${error.message}`);
      setRetrievalResults([]);
      setTopSearchCaptureId(null);
    } finally {
      setRetrievalBusy(false);
    }
  };

  const clearSearch = () => {
    setRetrievalResults([]);
    setRetrievalStatus('');
    setTopSearchCaptureId(null);
    clearSearchFallbackMarker();
  };

  return {
    retrievalSearchTopK,
    setRetrievalSearchTopK,
    retrievalStatus,
    retrievalBusy,
    retrievalResults,
    topSearchCaptureId,
    runImageSearch,
    clearSearch
  };
}
