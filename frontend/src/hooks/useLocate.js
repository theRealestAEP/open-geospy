import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';

import useBattlefrontSequence from './useBattlefrontSequence';
import useLocateProgress from './useLocateProgress';
import { locateBattlefrontRevealStartDelayMs } from '../lib/battlefront';
import {
  buildLocatePipelineStages,
  buildLocatePipelineStagesFromProgress,
  orbPopupMoments
} from '../lib/locatePipeline';
import {
  buildStreetViewUrl,
  createClientRetrievalId,
  formatRetrievalModelLabel
} from '../lib/format';

/**
 * Locate request state machine: form settings, the locate request itself
 * (with live progress + pipeline stages), ORB rerank results and popup state,
 * the battlefront reveal sequence, and result focusing on the shared map.
 */
export default function useLocate({
  mapRef,
  isLocatePage,
  currentPathRef,
  retrievalFile,
  retrievalMinSimilarity,
  locateEmbeddingBase,
  showPreview,
  loadRetrievalStats
}) {
  const locateOrbPopupSuppressedRef = useRef(false);
  const locateRequestAbortRef = useRef(null);
  const searchFallbackMarkerRef = useRef(null);

  const [locateTopK, setLocateTopK] = useState(8);
  const [locateBattlefrontModeEnabled, setLocateBattlefrontModeEnabled] = useState(false);
  const [locateOrbEnabled, setLocateOrbEnabled] = useState(false);
  const [locateOrbTopN, setLocateOrbTopN] = useState(100);
  const [locateOrbWeight, setLocateOrbWeight] = useState(0.75);
  const [locateOrbFeatureCount, setLocateOrbFeatureCount] = useState(500);
  const [locateOrbRansacTopK, setLocateOrbRansacTopK] = useState(10);
  const [locateOrbIgnoreBottomRatio, setLocateOrbIgnoreBottomRatio] = useState(0.28);
  const [locateSam2MaskCars, setLocateSam2MaskCars] = useState(false);
  const [locateSam2MaskTrees, setLocateSam2MaskTrees] = useState(false);
  const [locateStatus, setLocateStatus] = useState('');
  const [locateBusy, setLocateBusy] = useState(false);
  const [locateResults, setLocateResults] = useState([]);
  const [topLocateFamilyId, setTopLocateFamilyId] = useState(null);
  const [locateViewTab, setLocateViewTab] = useState('families');
  const [locateOrbStats, setLocateOrbStats] = useState(null);
  const [locateOrbComparisons, setLocateOrbComparisons] = useState([]);
  const [locateOrbPopupEnabled, setLocateOrbPopupEnabled] = useState(false);
  const [locateOrbPopupOpen, setLocateOrbPopupOpen] = useState(false);
  const [locateOrbPopupPhase, setLocateOrbPopupPhase] = useState('idle');
  const [locateOrbPopupMoment, setLocateOrbPopupMoment] = useState(0);
  const [locateProgressId, setLocateProgressId] = useState('');

  const {
    liveProgress: locateLiveProgress,
    pipeline: locatePipeline,
    setLiveProgress: setLocateLiveProgress,
    setPipeline: setLocatePipeline
  } = useLocateProgress({
    locateBusy,
    locateProgressId,
    buildStagesFromProgress: buildLocatePipelineStagesFromProgress,
    onMessage: setLocateStatus
  });

  const {
    sequence: locateBattlefrontSequence,
    clearSequence: clearLocateBattlefrontSequence,
    beginSearch: beginLocateBattlefrontSearch,
    waitForSearchReady: waitForLocateBattlefrontSearchReady,
    startReveal: startLocateBattlefrontReveal
  } = useBattlefrontSequence({ mapRef });

  const locateBattlefrontSequenceActive =
    isLocatePage && Boolean(locateBattlefrontModeEnabled && locateBattlefrontSequence);
  const battlefrontMapModeActive =
    isLocatePage && Boolean(locateBattlefrontModeEnabled && (locateBusy || locateBattlefrontSequence));
  const battlefrontMarkersHidden = battlefrontMapModeActive;
  const locateActionLocked = locateBusy || locateBattlefrontSequenceActive;

  useEffect(() => {
    if (isLocatePage) {
      return;
    }
    clearLocateBattlefrontSequence({ stopMap: true });
    locateOrbPopupSuppressedRef.current = true;
    setLocateOrbPopupOpen(false);
    setLocateOrbPopupPhase('idle');
    setLocateOrbPopupMoment(0);
    setLocateLiveProgress(null);
    setLocateProgressId('');
    if (locateRequestAbortRef.current) {
      locateRequestAbortRef.current.abort();
      locateRequestAbortRef.current = null;
    }
    setLocateBusy(false);
  }, [isLocatePage, clearLocateBattlefrontSequence, setLocateLiveProgress]);

  useEffect(() => {
    if (!locateOrbPopupOpen || locateOrbPopupPhase !== 'processing') {
      setLocateOrbPopupMoment(0);
      return undefined;
    }
    const intervalId = window.setInterval(() => {
      setLocateOrbPopupMoment((prev) => (prev + 1) % orbPopupMoments.length);
    }, 1400);
    return () => window.clearInterval(intervalId);
  }, [locateOrbPopupOpen, locateOrbPopupPhase]);

  const clearSearchFallbackMarker = () => {
    const map = mapRef.current;
    if (!map || !searchFallbackMarkerRef.current) return;
    map.removeLayer(searchFallbackMarkerRef.current);
    searchFallbackMarkerRef.current = null;
  };

  const focusRetrievalResult = async (result, options = {}) => {
    const {
      skipMapMove = false,
      keepBattlefrontReveal = false,
      preferredZoom = null,
      skipPreview = false
    } = options;
    if (!keepBattlefrontReveal) {
      clearLocateBattlefrontSequence({ stopMap: true });
    }
    const map = mapRef.current;
    const lat = Number(result?.lat ?? result?.family_center_lat);
    const lon = Number(result?.lon ?? result?.family_center_lon);
    const hasLocalImage = Boolean(String(result?.web_path || '').trim());
    clearSearchFallbackMarker();
    if (map && Number.isFinite(lat) && Number.isFinite(lon)) {
      if (!skipMapMove) {
        if (typeof map.stop === 'function') {
          map.stop();
        }
        const resolvedZoom = Number.isFinite(Number(preferredZoom))
          ? Number(preferredZoom)
          : Math.max(16, map.getZoom());
        map.setView([lat, lon], resolvedZoom, {
          animate: false
        });
      }
      const streetViewUrl = buildStreetViewUrl(result);
      const marker = L.circleMarker([lat, lon], {
        radius: 7,
        fillColor: hasLocalImage ? '#4cc9f0' : '#ffd166',
        fillOpacity: 0.95,
        color: '#111',
        weight: 2
      }).addTo(map);
      if (streetViewUrl) {
        marker.bindTooltip(
          hasLocalImage
            ? 'Matched capture. Click to open Street View'
            : 'No local image. Click to open Street View'
        );
        marker.on('click', () => {
          window.open(streetViewUrl, '_blank', 'noopener,noreferrer');
        });
      } else {
        marker.bindTooltip(
          hasLocalImage ? 'Matched capture location' : 'No local image available'
        );
      }
      searchFallbackMarkerRef.current = marker;
    }
    if (!skipPreview && result?.panorama_id != null) {
      await showPreview({ id: result.panorama_id, pano_id: result.pano_id || '' });
    }
  };

  const runImageLocate = async () => {
    if (locateActionLocked) {
      return;
    }
    if (!retrievalFile) {
      setLocateStatus('Pick a reference image first.');
      return;
    }
    if (locateRequestAbortRef.current) {
      locateRequestAbortRef.current.abort();
    }
    const requestController = new AbortController();
    locateRequestAbortRef.current = requestController;
    const retrievalId = createClientRetrievalId();
    setLocateBusy(true);
    setLocateProgressId(retrievalId);
    setLocateLiveProgress(null);
    setLocateViewTab('families');
    setLocateOrbStats(null);
    setLocateOrbComparisons([]);
    setLocateOrbPopupMoment(0);
    clearLocateBattlefrontSequence({ stopMap: true });
    locateOrbPopupSuppressedRef.current = false;
    clearSearchFallbackMarker();
    setLocatePipeline({ stages: buildLocatePipelineStages('vector_search', 'running') });
    setLocateStatus(
      locateOrbEnabled ? 'Locating image with ORB rerank...' : 'Locating image...'
    );
    if (locateBattlefrontModeEnabled) {
      beginLocateBattlefrontSearch();
      setLocateOrbPopupOpen(false);
      setLocateOrbPopupPhase('idle');
    } else if (locateOrbEnabled && locateOrbPopupEnabled) {
      setLocateOrbPopupOpen(true);
      setLocateOrbPopupPhase('processing');
    } else {
      setLocateOrbPopupOpen(false);
      setLocateOrbPopupPhase('idle');
    }
    try {
      const formData = new FormData();
      formData.append('image', retrievalFile);
      formData.append('client_retrieval_id', retrievalId);
      formData.append('top_k', String(Math.max(1, Number(locateTopK) || 8)));
      formData.append('embedding_base', locateEmbeddingBase);
      formData.append('orb_enabled', locateOrbEnabled ? '1' : '0');
      if (locateOrbEnabled) {
        formData.append('orb_top_n', String(Math.max(1, Number(locateOrbTopN) || 100)));
        const parsedOrbWeight = Number(locateOrbWeight);
        if (!Number.isNaN(parsedOrbWeight)) {
          formData.append('orb_weight', String(parsedOrbWeight));
        }
        const parsedFeatureCount = Number(locateOrbFeatureCount);
        if (!Number.isNaN(parsedFeatureCount)) {
          formData.append('orb_feature_count', String(Math.max(100, Math.min(2000, parsedFeatureCount))));
        }
        const parsedRansacTopK = Number(locateOrbRansacTopK);
        if (!Number.isNaN(parsedRansacTopK)) {
          formData.append(
            'orb_ransac_top_k',
            String(Math.max(0, Math.min(Math.max(1, Number(locateOrbTopN) || 100), parsedRansacTopK)))
          );
        }
        const parsedIgnoreBottomRatio = Number(locateOrbIgnoreBottomRatio);
        if (!Number.isNaN(parsedIgnoreBottomRatio)) {
          formData.append(
            'orb_ignore_bottom_ratio',
            String(Math.max(0, Math.min(0.6, parsedIgnoreBottomRatio)))
          );
        }
        formData.append('sam2_mask_cars', locateSam2MaskCars ? '1' : '0');
        formData.append('sam2_mask_trees', locateSam2MaskTrees ? '1' : '0');
      }
      const minSimilarityRaw = String(retrievalMinSimilarity ?? '').trim();
      if (minSimilarityRaw !== '') {
        const parsedSimilarity = Number(minSimilarityRaw);
        if (!Number.isNaN(parsedSimilarity)) {
          formData.append('min_similarity', String(parsedSimilarity));
        }
      }
      const res = await fetch('/api/retrieval/locate-by-image', {
        method: 'POST',
        body: formData,
        signal: requestController.signal
      });
      const body = await res.json();
      if (currentPathRef.current !== '/locate') {
        return;
      }
      if (!res.ok) {
        setLocateStatus(`Locate failed: ${body?.detail || 'unknown error'}`);
        setLocateResults([]);
        setTopLocateFamilyId(null);
        setLocateOrbStats(null);
        setLocateOrbComparisons([]);
        clearLocateBattlefrontSequence({ stopMap: true });
        setLocatePipeline({ stages: buildLocatePipelineStages('vector_search', 'failed') });
        setLocateOrbPopupPhase('error');
        return;
      }
      const matches = Array.isArray(body.matches) ? body.matches : [];
      const pipelineStages = Array.isArray(body?.pipeline?.stages) ? body.pipeline.stages : [];
      const orbStats = body?.orb?.stats && typeof body.orb.stats === 'object' ? body.orb.stats : null;
      const orbComparisons = Array.isArray(orbStats?.comparisons) ? orbStats.comparisons : [];
      setLocateLiveProgress(null);
      setLocateResults(matches);
      setTopLocateFamilyId(matches.length ? String(matches[0].family_id || '') : null);
      setLocateOrbStats(orbStats);
      setLocateOrbComparisons(orbComparisons);
      setLocatePipeline({
        stages: pipelineStages.length
          ? pipelineStages
          : buildLocatePipelineStages('family_rank', 'completed')
      });
      const modelLabel = formatRetrievalModelLabel(body.model_name, body.model_version);
      setLocateStatus(
        `Found ${matches.length} location families from ${Number(body.capture_candidates || 0)} capture candidates${locateOrbEnabled ? ` with ORB top ${Math.max(1, Number(locateOrbTopN) || 100)}` : ''}${modelLabel ? ` (${modelLabel})` : ''}.`
      );
      const shouldPlayBattlefrontReveal = locateBattlefrontModeEnabled && matches.length > 0;
      const shouldOpenOrbPopupOnFinish =
        locateOrbEnabled && locateOrbPopupEnabled && !locateOrbPopupSuppressedRef.current;
      if (shouldPlayBattlefrontReveal) {
        setLocateOrbPopupOpen(false);
        setLocateOrbPopupPhase(shouldOpenOrbPopupOnFinish ? 'results' : 'idle');
        const startReveal = async () => {
          await waitForLocateBattlefrontSearchReady();
          if (currentPathRef.current !== '/locate') {
            return;
          }
          await new Promise((resolve) => {
            window.setTimeout(resolve, locateBattlefrontRevealStartDelayMs);
          });
          if (currentPathRef.current !== '/locate') {
            return;
          }
          void startLocateBattlefrontReveal(matches[0], {
            focusResult: focusRetrievalResult,
            clearSearchFallbackMarker,
            onComplete: () => {
              if (shouldOpenOrbPopupOnFinish && !locateOrbPopupSuppressedRef.current) {
                setLocateOrbPopupPhase('results');
                setLocateOrbPopupOpen(true);
              }
            }
          });
        };
        void startReveal();
      } else {
        if (locateBattlefrontModeEnabled) {
          clearLocateBattlefrontSequence({ stopMap: true });
        }
        if (matches.length) {
          await focusRetrievalResult(matches[0]);
        }
        if (shouldOpenOrbPopupOnFinish) {
          setLocateOrbPopupOpen(true);
          setLocateOrbPopupPhase('results');
        } else {
          setLocateOrbPopupPhase('idle');
        }
      }
      await loadRetrievalStats();
    } catch (error) {
      if (error?.name === 'AbortError') {
        return;
      }
      setLocateStatus(`Locate failed: ${error.message}`);
      setLocateResults([]);
      setTopLocateFamilyId(null);
      setLocateOrbStats(null);
      setLocateOrbComparisons([]);
      clearLocateBattlefrontSequence({ stopMap: true });
      setLocatePipeline({ stages: buildLocatePipelineStages('vector_search', 'failed') });
      setLocateOrbPopupPhase('error');
    } finally {
      if (locateRequestAbortRef.current === requestController) {
        locateRequestAbortRef.current = null;
      }
      setLocateBusy(false);
    }
  };

  const clearLocate = () => {
    clearLocateBattlefrontSequence({ stopMap: true });
    setLocateResults([]);
    setLocateStatus('');
    setTopLocateFamilyId(null);
    setLocatePipeline({ stages: [] });
    setLocateViewTab('families');
    setLocateOrbStats(null);
    setLocateOrbComparisons([]);
    setLocateLiveProgress(null);
    setLocateProgressId('');
    setLocateOrbPopupOpen(false);
    setLocateOrbPopupPhase('idle');
    locateOrbPopupSuppressedRef.current = false;
    clearSearchFallbackMarker();
  };

  const openOrbPopup = () => {
    locateOrbPopupSuppressedRef.current = false;
    setLocateOrbPopupPhase('results');
    setLocateOrbPopupOpen(true);
  };

  const dismissOrbPopup = () => {
    locateOrbPopupSuppressedRef.current = true;
    setLocateOrbPopupOpen(false);
  };

  const openOrbTabFromPopup = () => {
    locateOrbPopupSuppressedRef.current = true;
    setLocateViewTab('orb');
    setLocateOrbPopupOpen(false);
  };

  // Shared derived view state (used by the locate page, ORB popup, and
  // battlefront overlay).
  const liveLocateProgress =
    locateLiveProgress && typeof locateLiveProgress === 'object' ? locateLiveProgress : null;
  const liveLocateOrbStats =
    liveLocateProgress?.orb && typeof liveLocateProgress.orb === 'object'
      ? liveLocateProgress.orb
      : null;
  const displayedLocateOrbStats =
    locateBusy && liveLocateOrbStats
      ? { ...(locateOrbStats || {}), ...liveLocateOrbStats }
      : locateOrbStats;
  const liveLocateComparison =
    locateOrbPopupPhase === 'processing' &&
    liveLocateOrbStats?.latest_comparison &&
    typeof liveLocateOrbStats.latest_comparison === 'object'
      ? liveLocateOrbStats.latest_comparison
      : null;
  const hasLocateOrbComparisons = locateOrbComparisons.length > 0;
  const liveLocatePhase = String(liveLocateProgress?.phase || '').trim().toLowerCase();
  const locateOrbAppliedIgnoreBottomRatio =
    locateOrbPopupPhase === 'processing'
      ? Math.max(
          0,
          Math.min(
            0.6,
            Number(liveLocateOrbStats?.ignore_bottom_ratio ?? locateOrbIgnoreBottomRatio) || 0
          )
        )
      : Number(displayedLocateOrbStats?.ignore_bottom_ratio || 0);
  const locateOrbMaskPercent = Math.max(0, Math.round(locateOrbAppliedIgnoreBottomRatio * 100));
  const locateSam2Enabled =
    locateOrbPopupPhase === 'processing'
      ? Boolean(
          liveLocateOrbStats?.sam2_enabled ??
            (locateSam2MaskCars || locateSam2MaskTrees)
        )
      : Boolean(displayedLocateOrbStats?.sam2_enabled);
  const locateSam2CarsEnabled =
    locateOrbPopupPhase === 'processing'
      ? Boolean(liveLocateOrbStats?.sam2_mask_cars ?? locateSam2MaskCars)
      : Boolean(displayedLocateOrbStats?.sam2_mask_cars);
  const locateSam2TreesEnabled =
    locateOrbPopupPhase === 'processing'
      ? Boolean(liveLocateOrbStats?.sam2_mask_trees ?? locateSam2MaskTrees)
      : Boolean(displayedLocateOrbStats?.sam2_mask_trees);
  const locateSam2VehicleBoxes = Number(
    displayedLocateOrbStats?.sam2_vehicle_boxes || liveLocateOrbStats?.sam2_vehicle_boxes || 0
  );
  const locateSam2TreeBoxes = Number(
    displayedLocateOrbStats?.sam2_tree_boxes || liveLocateOrbStats?.sam2_tree_boxes || 0
  );
  const locateOrbProcessedCandidates = Number(liveLocateOrbStats?.processed_candidates || 0);
  const locateOrbCandidateCount = Number(
    liveLocateOrbStats?.candidate_count || displayedLocateOrbStats?.candidate_count || 0
  );
  const locateOrbResolvedFeatureCount = Math.max(
    0,
    Number(displayedLocateOrbStats?.feature_count || locateOrbFeatureCount || 0)
  );

  return {
    // form settings
    locateTopK,
    setLocateTopK,
    locateBattlefrontModeEnabled,
    setLocateBattlefrontModeEnabled,
    locateOrbEnabled,
    setLocateOrbEnabled,
    locateOrbTopN,
    setLocateOrbTopN,
    locateOrbWeight,
    setLocateOrbWeight,
    locateOrbFeatureCount,
    setLocateOrbFeatureCount,
    locateOrbRansacTopK,
    setLocateOrbRansacTopK,
    locateOrbIgnoreBottomRatio,
    setLocateOrbIgnoreBottomRatio,
    locateSam2MaskCars,
    setLocateSam2MaskCars,
    locateSam2MaskTrees,
    setLocateSam2MaskTrees,
    locateOrbPopupEnabled,
    setLocateOrbPopupEnabled,
    // request state
    locateStatus,
    locateBusy,
    locateResults,
    topLocateFamilyId,
    locateViewTab,
    setLocateViewTab,
    locatePipeline,
    locateOrbComparisons,
    // popup state
    locateOrbPopupOpen,
    locateOrbPopupPhase,
    locateOrbPopupMoment,
    openOrbPopup,
    dismissOrbPopup,
    openOrbTabFromPopup,
    // battlefront
    locateBattlefrontSequence,
    locateBattlefrontSequenceActive,
    battlefrontMapModeActive,
    battlefrontMarkersHidden,
    locateActionLocked,
    // actions
    runImageLocate,
    clearLocate,
    focusRetrievalResult,
    clearSearchFallbackMarker,
    // shared derived view state
    liveLocateProgress,
    liveLocateOrbStats,
    displayedLocateOrbStats,
    liveLocateComparison,
    hasLocateOrbComparisons,
    liveLocatePhase,
    locateOrbMaskPercent,
    locateSam2Enabled,
    locateSam2CarsEnabled,
    locateSam2TreesEnabled,
    locateSam2VehicleBoxes,
    locateSam2TreeBoxes,
    locateOrbProcessedCandidates,
    locateOrbCandidateCount,
    locateOrbResolvedFeatureCount
  };
}
