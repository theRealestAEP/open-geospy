import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';
import 'leaflet-draw';
import { useLocation, useNavigate } from 'react-router-dom';

const defaultScanForm = {
  minLat: 37.784,
  minLon: -122.438,
  maxLat: 37.801,
  maxLon: -122.422,
  workers: 4,
  stepMeters: 50,
  dedupRadius: 25,
  mode: 'modal',
  jobType: 'scan',
  captureProfile: 'high_v1',
  fillGapMeters: 40,
  enrichMissingOnly: true
};
const MIN_POLYGON_POINT_DISTANCE_METERS = 2.0;
const ACTIVE_SCANS_STORAGE_KEY = 'geospy.active_scan_ids';
const LEGACY_ACTIVE_SCAN_STORAGE_KEY = 'geospy.active_scan_id';
const retrievalControlHelp = {
  reference_image: 'Upload the photo you want to search.',
  search_top_k: 'How many nearest results to return in Search mode.',
  locate_top_k: 'How many location families to return in Locate mode. The ORB gallery uses the same count for compared capture images.',
  min_similarity: 'Optional floor for match similarity. Higher values are stricter and may reduce recall.',
  embedding_base: 'Choose which embedding set this action uses.',
  locate_battlefront_mode: 'Play a cinematic map reveal after a successful locate, with a progressive zoom from regional scale down to the matched street.',
  locate_orb_enabled: 'Run ORB local-feature reranking on the top vector candidates before panorama aggregation.',
  locate_orb_top_n: 'How many top vector candidates should be reranked with ORB.',
  locate_orb_weight: 'How strongly the ORB score boosts the vector score during locate reranking.',
  locate_orb_feature_count: 'How many ORB keypoints to extract per image during reranking. Higher values are slower but can preserve more detail.',
  locate_orb_ransac_top_k: 'How many top ORB candidates should be geometry-checked with RANSAC.',
  locate_orb_ignore_bottom_ratio: 'Optional fraction of the lower frame to ignore during ORB feature extraction.',
  locate_sam2_mask_cars: 'Use local SAM 2 masking to remove cars and other road vehicles from the query image and top ORB review candidates before feature extraction.',
  locate_sam2_mask_trees: 'Experimental: use local SAM 2 masking to remove trees and other vegetation from the query image and top ORB review candidates before feature extraction.',
  locate_orb_popup: 'Optionally auto-open the ORB fingerprint popup while comparisons are running.'
};
const orbPopupMoments = [
  'Extracting keypoints from the query frame.',
  'Sweeping candidate facades for matching fingerprints.',
  'Linking feature pairs and pruning weak matches.',
  'Testing the strongest alignments for geometric consistency.'
];
const orbPopupPreviewStages = ['keypoints', 'scan', 'link', 'geometry'];
const locateBattlefrontAudioPath = '/assets/loading.mp3';
const locateBattlefrontStageLineDurationMs = 3000;
const locateBattlefrontStagePauseMs = 700;
const locateBattlefrontStageZoomDurationSeconds = 1.9;
const locateBattlefrontStageSettleMs = 260;
const locateBattlefrontRevealStartDelayMs = 1000;
const orbPopupPreviewPoints = [
  { x: 14, y: 18, delay: 0.1 },
  { x: 31, y: 24, delay: 0.35 },
  { x: 52, y: 21, delay: 0.2 },
  { x: 71, y: 17, delay: 0.55 },
  { x: 83, y: 28, delay: 0.75 },
  { x: 24, y: 43, delay: 0.4 },
  { x: 45, y: 39, delay: 0.15 },
  { x: 63, y: 46, delay: 0.6 },
  { x: 78, y: 41, delay: 0.3 },
  { x: 18, y: 62, delay: 0.8 },
  { x: 39, y: 68, delay: 0.5 },
  { x: 58, y: 61, delay: 0.95 },
  { x: 74, y: 66, delay: 0.7 },
  { x: 86, y: 54, delay: 1.05 }
];
const locateBattlefrontRevealStageBlueprints = [
  {
    key: 'acquiring',
    label: 'Country lock',
    title: 'Country lock acquired',
    subtitle: 'Resolving the winning family inside a national-scale tactical grid.',
    zoom: 5,
    lineDurationMs: locateBattlefrontStageLineDurationMs,
    pauseAfterLinesMs: locateBattlefrontStagePauseMs,
    zoomDurationSeconds: locateBattlefrontStageZoomDurationSeconds,
    settleMs: locateBattlefrontStageSettleMs
  },
  {
    key: 'approach',
    label: 'State corridor',
    title: 'State corridor resolved',
    subtitle: 'Compressing the search volume into a state-level ingress lane.',
    zoom: 8,
    lineDurationMs: locateBattlefrontStageLineDurationMs,
    pauseAfterLinesMs: locateBattlefrontStagePauseMs,
    zoomDurationSeconds: locateBattlefrontStageZoomDurationSeconds,
    settleMs: locateBattlefrontStageSettleMs
  },
  {
    key: 'descent',
    label: 'City / county',
    title: 'City / county descent',
    subtitle: 'Tightening the corridor onto the metro and county grid.',
    zoom: 12,
    lineDurationMs: locateBattlefrontStageLineDurationMs,
    pauseAfterLinesMs: locateBattlefrontStagePauseMs,
    zoomDurationSeconds: locateBattlefrontStageZoomDurationSeconds,
    settleMs: locateBattlefrontStageSettleMs
  },
  {
    key: 'terminal',
    label: 'Location lock',
    title: 'Location lock',
    subtitle: 'Finalizing the recovered panorama and committing the street position.',
    zoom: 21,
    lineDurationMs: locateBattlefrontStageLineDurationMs,
    pauseAfterLinesMs: locateBattlefrontStagePauseMs,
    zoomDurationSeconds: locateBattlefrontStageZoomDurationSeconds,
    settleMs: locateBattlefrontStageSettleMs
  }
];
const locateBattlefrontRevealStages = [];
let locateBattlefrontRevealCursorMs = 0;
locateBattlefrontRevealStageBlueprints.forEach((stage) => {
  const zoomDelayMs = Number(stage.lineDurationMs || 0) + Number(stage.pauseAfterLinesMs || 0);
  const stageDurationMs =
    zoomDelayMs + Math.round(Number(stage.zoomDurationSeconds || 0) * 1000) + Number(stage.settleMs || 0);
  locateBattlefrontRevealStages.push({
    ...stage,
    at: locateBattlefrontRevealCursorMs,
    lineLeadMs: zoomDelayMs,
    zoomDelayMs,
    duration: Number(stage.zoomDurationSeconds || 0),
    stageDurationMs
  });
  locateBattlefrontRevealCursorMs += stageDurationMs;
});
const locateBattlefrontLockDelayMs = locateBattlefrontRevealCursorMs;
const locateBattlefrontClearDelayMs = locateBattlefrontLockDelayMs + 1800;
const locateBattlefrontSearchZoomDurationMs = 4400;

function buildLocatePipelineStages(activeKey = 'vector_search', status = 'running') {
  const stageOrder = [
    ['vector_search', 'Vector search'],
    ['orb_rerank', 'ORB rerank'],
    ['panorama_rerank', 'Panorama aggregation'],
    ['family_rank', 'Panorama-family ranking']
  ];
  let seenActive = false;
  return stageOrder.map(([key, title]) => {
    let stageStatus = 'pending';
    if (key === activeKey) {
      stageStatus = status;
      seenActive = true;
    } else if (!seenActive) {
      stageStatus = 'completed';
    }
    return { key, title, status: stageStatus, detail: '' };
  });
}

function formatRetrievalModelLabel(modelName, modelVersion) {
  const name = String(modelName || '').trim();
  const version = String(modelVersion || '').trim().toLowerCase();
  if (!name) {
    return '';
  }
  if (!version || version === 'open_clip') {
    return name;
  }
  if (version === 'open_clip_place') {
    return `${name} place`;
  }
  return `${name} ${modelVersion}`;
}

function formatMetric(value, digits = 3) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : '0.000';
}

function formatCoordinate(value, digits = 5) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : 'n/a';
}

function createClientRetrievalId() {
  if (
    typeof window !== 'undefined' &&
    window.crypto &&
    typeof window.crypto.randomUUID === 'function'
  ) {
    return `locate-${window.crypto.randomUUID()}`;
  }
  return `locate-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function mapLocateProgressPhaseToPreviewStage(phase, orbProgress) {
  const normalizedPhase = String(phase || '').trim().toLowerCase();
  const processedCandidates = Number(orbProgress?.processed_candidates || 0);
  const latestComparison = Boolean(orbProgress?.latest_comparison);
  if (normalizedPhase === 'vector_search') return 'scan';
  if (normalizedPhase === 'orb_rerank') {
    if (latestComparison) {
      return processedCandidates > 2 ? 'geometry' : 'link';
    }
    return 'keypoints';
  }
  if (normalizedPhase === 'panorama_rerank' || normalizedPhase === 'family_rank') {
    return 'geometry';
  }
  return 'keypoints';
}

function formatLocateProgressStageBadge(phase, orbEnabled) {
  const normalizedPhase = String(phase || '').trim().toLowerCase();
  if (normalizedPhase === 'vector_search') return 'Vector search';
  if (normalizedPhase === 'orb_rerank') return 'ORB rerank';
  if (normalizedPhase === 'panorama_rerank') return orbEnabled ? 'Panorama merge' : 'Panorama aggregation';
  if (normalizedPhase === 'family_rank') return 'Family ranking';
  if (normalizedPhase === 'completed') return 'Wrapping up';
  return orbEnabled ? 'Preparing ORB' : 'Preparing locate';
}

function buildLocatePipelineStagesFromProgress(progress) {
  const phase = String(progress?.phase || 'starting').trim().toLowerCase();
  const status = String(progress?.status || 'processing').trim().toLowerCase();
  const orb = progress?.orb && typeof progress.orb === 'object' ? progress.orb : {};
  const orbEnabled = Boolean(orb.enabled);
  const stageStatus =
    status === 'error' ? 'failed' : status === 'completed' ? 'completed' : 'running';
  let activeKey = 'vector_search';
  if (phase === 'orb_rerank') {
    activeKey = 'orb_rerank';
  } else if (phase === 'panorama_rerank') {
    activeKey = 'panorama_rerank';
  } else if (phase === 'family_rank' || phase === 'completed') {
    activeKey = 'family_rank';
  }
  return buildLocatePipelineStages(activeKey, stageStatus).map((stage) => {
    const next = { ...stage };
    if (stage.key === 'vector_search') {
      const vectorCandidates = Number(progress?.vector_candidates || 0);
      if (vectorCandidates > 0) {
        next.detail = `Retrieved ${vectorCandidates} raw vector candidates from the selected model.`;
      } else if (phase === 'vector_search' || phase === 'starting') {
        next.detail = String(progress?.message || 'Searching embeddings for nearby matches.');
      }
    }
    if (stage.key === 'orb_rerank') {
      if (!orbEnabled && phase !== 'starting' && phase !== 'vector_search') {
        next.status = 'completed';
        next.detail = 'Skipped for this run.';
      } else if (orbEnabled) {
        const processedCandidates = Number(orb.processed_candidates || 0);
        const candidateCount = Number(orb.candidate_count || 0);
        if (processedCandidates > 0 && candidateCount > 0) {
          next.detail = `Compared ${processedCandidates} of ${candidateCount} candidates with ORB fingerprints.`;
        } else if (phase === 'orb_rerank') {
          next.detail = String(progress?.message || 'Extracting query features and comparing candidates.');
        }
      }
    }
    if (stage.key === 'panorama_rerank') {
      const panoramaCandidates = Number(progress?.panorama_candidates || 0);
      if (panoramaCandidates > 0) {
        next.detail = `Collapsed capture hits into ${panoramaCandidates} panorama candidates.`;
      } else if (phase === 'panorama_rerank') {
        next.detail = String(progress?.message || 'Collapsing reranked captures into panoramas.');
      }
    }
    if (stage.key === 'family_rank') {
      const familyMatches = Number(progress?.matches || 0);
      if (familyMatches > 0) {
        next.detail = `Ranking ${familyMatches} nearby location families for the final result.`;
      } else if (phase === 'family_rank' || phase === 'completed') {
        next.detail = String(progress?.message || 'Grouping panoramas into nearby location families.');
      }
    }
    return next;
  });
}

function getOrbPopupTimelineState(index, phase, activeMoment) {
  if (phase === 'processing') {
    if (index === activeMoment) return 'active';
    if (index < activeMoment) return 'done';
    return '';
  }
  if (phase === 'results') {
    return 'done';
  }
  if (phase === 'error') {
    return index === 0 ? 'failed' : '';
  }
  return '';
}

function interpolateColor(c1, c2, ratio) {
  const hex = (s) => parseInt(s, 16);
  const r = Math.round(hex(c1.slice(1, 3)) + (hex(c2.slice(1, 3)) - hex(c1.slice(1, 3))) * ratio);
  const g = Math.round(hex(c1.slice(3, 5)) + (hex(c2.slice(3, 5)) - hex(c1.slice(3, 5))) * ratio);
  const b = Math.round(hex(c1.slice(5, 7)) + (hex(c2.slice(5, 7)) - hex(c1.slice(5, 7))) * ratio);
  return `rgb(${r},${g},${b})`;
}

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef(null);
  const drawnItemsRef = useRef(null);
  const oneShotMarkerRef = useRef(null);
  const searchFallbackMarkerRef = useRef(null);
  const pointModeEnabledRef = useRef(false);
  const freeDrawEnabledRef = useRef(false);
  const freeDrawDrawingRef = useRef(false);
  const freeDrawLatLngsRef = useRef([]);
  const freeDrawPreviewRef = useRef(null);
  const polygonModeEnabledRef = useRef(false);
  const polygonDraftLatLngsRef = useRef([]);
  const polygonDraftPreviewRef = useRef(null);
  const polygonDraftVertexLayerRef = useRef(null);
  const locateOrbPopupSuppressedRef = useRef(false);
  const locateRequestAbortRef = useRef(null);
  const locateBattlefrontTimeoutsRef = useRef([]);
  const currentPathRef = useRef(location.pathname);
  const pointModeButtonRef = useRef(null);
  const polygonModeButtonRef = useRef(null);
  const freeDrawButtonRef = useRef(null);
  const hasAutoFittedBoundsRef = useRef(false);
  const autoRefreshRef = useRef(null);
  const viewportAbortRef = useRef(null);
  const viewportDebounceRef = useRef(null);
  const scanPollRef = useRef(null);
  const activeScanIdRef = useRef(null);
  const trackedScanIdsRef = useRef([]);
  const markerRendererRef = useRef(null);
  const battlefrontMapModeActiveRef = useRef(false);
  const battlefrontMarkersHiddenRef = useRef(false);
  const locateBattlefrontAudioRef = useRef(null);
  const locateBattlefrontAudioStopTimeoutRef = useRef(null);
  const locateBattlefrontSearchStartedAtRef = useRef(0);
  const locateBattlefrontSearchReadyPromiseRef = useRef(Promise.resolve());
  const locateBattlefrontSearchReadyTimeoutRef = useRef(null);
  const locateBattlefrontSearchMoveEndHandlerRef = useRef(null);

  const [geojsonData, setGeojsonData] = useState(null);
  const [stats, setStats] = useState({ total_panoramas: 0, total_captures: 0, bounds: null });
  const [colorMode, setColorMode] = useState('recency');
  const [preview, setPreview] = useState({ title: 'Click a marker to preview captures', captures: [] });

  const [oneShotLat, setOneShotLat] = useState('');
  const [oneShotLon, setOneShotLon] = useState('');
  const [oneShotStatus, setOneShotStatus] = useState('');

  const [scanForm, setScanForm] = useState(defaultScanForm);
  const [scanShapeType, setScanShapeType] = useState(null);
  const [selectedPolygonCoords, setSelectedPolygonCoords] = useState([]);
  const [scanStatusText, setScanStatusText] = useState('');
  const [activeScanId, setActiveScanId] = useState(null);
  const [trackedScanIds, setTrackedScanIds] = useState([]);
  const [scanJobs, setScanJobs] = useState([]);
  const [scanProgress, setScanProgress] = useState({
    pending: 0,
    inProgress: 0,
    done: 0,
    skipped: 0,
    failed: 0,
    workers: 0
  });
  const [retrievalFile, setRetrievalFile] = useState(null);
  const [retrievalSearchTopK, setRetrievalSearchTopK] = useState(12);
  const [retrievalMinSimilarity, setRetrievalMinSimilarity] = useState('');
  const [retrievalStatus, setRetrievalStatus] = useState('');
  const [retrievalBusy, setRetrievalBusy] = useState(false);
  const [retrievalResults, setRetrievalResults] = useState([]);
  const [topSearchCaptureId, setTopSearchCaptureId] = useState(null);
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
  const [locatePipeline, setLocatePipeline] = useState({ stages: [] });
  const [locateViewTab, setLocateViewTab] = useState('families');
  const [locateOrbStats, setLocateOrbStats] = useState(null);
  const [locateOrbComparisons, setLocateOrbComparisons] = useState([]);
  const [locateOrbPopupEnabled, setLocateOrbPopupEnabled] = useState(false);
  const [locateOrbPopupOpen, setLocateOrbPopupOpen] = useState(false);
  const [locateOrbPopupPhase, setLocateOrbPopupPhase] = useState('idle');
  const [locateOrbPopupMoment, setLocateOrbPopupMoment] = useState(0);
  const [locateProgressId, setLocateProgressId] = useState('');
  const [locateLiveProgress, setLocateLiveProgress] = useState(null);
  const [locateBattlefrontSequence, setLocateBattlefrontSequence] = useState(null);
  const [searchEmbeddingBase, setSearchEmbeddingBase] = useState('');
  const [locateEmbeddingBase, setLocateEmbeddingBase] = useState('');
  const [retrievalEmbeddingBaseOptions, setRetrievalEmbeddingBaseOptions] = useState([
    { value: 'clip', label: 'CLIP' }
  ]);
  const [retrievalFilePreviewUrl, setRetrievalFilePreviewUrl] = useState('');
  const [retrievalStats, setRetrievalStats] = useState({
    model_name: '',
    model_version: '',
    total_captures: 0,
    embedded_captures: 0,
    pending_captures: 0,
    models: []
  });
  const isScanPage = location.pathname === '/scan' || location.pathname === '/';
  const isSearchPage = location.pathname === '/search';
  const isLocatePage = location.pathname === '/locate';
  const currentWorkerLimit = scanForm.mode === 'modal' ? 100 : 32;
  const battlefrontMapModeActive =
    isLocatePage && Boolean(locateBattlefrontModeEnabled && (locateBusy || locateBattlefrontSequence));
  const battlefrontMarkersHidden = battlefrontMapModeActive;

  useEffect(() => {
    battlefrontMapModeActiveRef.current = battlefrontMapModeActive;
    battlefrontMarkersHiddenRef.current = battlefrontMarkersHidden;
    if (!battlefrontMarkersHidden) {
      return;
    }
    if (viewportDebounceRef.current) {
      window.clearTimeout(viewportDebounceRef.current);
      viewportDebounceRef.current = null;
    }
    if (viewportAbortRef.current) {
      viewportAbortRef.current.abort();
      viewportAbortRef.current = null;
    }
    if (markersRef.current) {
      markersRef.current.clearLayers();
    }
  }, [battlefrontMapModeActive, battlefrontMarkersHidden]);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, { preferCanvas: true }).setView([37.785, -122.43], 14);
    mapRef.current = map;
    markerRendererRef.current = L.canvas({ padding: 0.5 });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    markersRef.current = L.layerGroup().addTo(map);
    drawnItemsRef.current = new L.FeatureGroup();
    map.addLayer(drawnItemsRef.current);
    polygonDraftVertexLayerRef.current = L.layerGroup().addTo(map);

    const drawControl = new L.Control.Draw({
      edit: {
        featureGroup: drawnItemsRef.current,
        edit: {},
        remove: true
      },
      draw: {
        polyline: false,
        marker: false,
        circle: false,
        circlemarker: false,
        rectangle: false,
        polygon: false
      }
    });
    map.addControl(drawControl);

    const applyLayerBounds = (layer, type) => {
      const bounds = layer.getBounds();
      const next = {
        ...scanForm,
        minLat: Number(bounds.getSouth().toFixed(6)),
        minLon: Number(bounds.getWest().toFixed(6)),
        maxLat: Number(bounds.getNorth().toFixed(6)),
        maxLon: Number(bounds.getEast().toFixed(6))
      };
      setScanForm(next);
      setScanShapeType(type);

      if (type === 'polygon') {
        const latlngs = layer.getLatLngs();
        const ring = Array.isArray(latlngs?.[0]) ? latlngs[0] : [];
        setSelectedPolygonCoords(ring.map((ll) => [Number(ll.lat.toFixed(6)), Number(ll.lng.toFixed(6))]));
      } else {
        setSelectedPolygonCoords([]);
      }
    };

    const syncModeButtonStyles = () => {
      if (pointModeButtonRef.current) {
        pointModeButtonRef.current.classList.toggle('active', pointModeEnabledRef.current);
      }
      if (polygonModeButtonRef.current) {
        polygonModeButtonRef.current.classList.toggle('active', polygonModeEnabledRef.current);
      }
      if (freeDrawButtonRef.current) {
        freeDrawButtonRef.current.classList.toggle('active', freeDrawEnabledRef.current);
      }
    };

    const clearFreeDrawPreview = () => {
      if (freeDrawPreviewRef.current) {
        map.removeLayer(freeDrawPreviewRef.current);
        freeDrawPreviewRef.current = null;
      }
      freeDrawLatLngsRef.current = [];
      freeDrawDrawingRef.current = false;
      map.dragging.enable();
    };

    const renderPolygonDraftPreview = (cursorLatLng = null) => {
      if (polygonDraftPreviewRef.current) {
        map.removeLayer(polygonDraftPreviewRef.current);
        polygonDraftPreviewRef.current = null;
      }
      if (polygonDraftVertexLayerRef.current) {
        polygonDraftVertexLayerRef.current.clearLayers();
      }

      const points = polygonDraftLatLngsRef.current;
      points.forEach((point) => {
        if (!polygonDraftVertexLayerRef.current) return;
        L.circleMarker(point, {
          radius: 4,
          color: '#2f8cff',
          weight: 2,
          fillColor: '#ffffff',
          fillOpacity: 1
        }).addTo(polygonDraftVertexLayerRef.current);
      });

      const previewPoints = cursorLatLng ? [...points, cursorLatLng] : [...points];
      if (previewPoints.length >= 3) {
        polygonDraftPreviewRef.current = L.polygon(previewPoints, {
          color: '#2f8cff',
          weight: 2,
          opacity: 0.95,
          fillColor: '#2f8cff',
          fillOpacity: 0.16,
          dashArray: '6 4'
        }).addTo(map);
      } else if (previewPoints.length >= 2) {
        polygonDraftPreviewRef.current = L.polyline(previewPoints, {
          color: '#2f8cff',
          weight: 3,
          opacity: 0.85,
          dashArray: '6 4'
        }).addTo(map);
      }
    };

    const clearPolygonDraftPreview = () => {
      if (polygonDraftPreviewRef.current) {
        map.removeLayer(polygonDraftPreviewRef.current);
        polygonDraftPreviewRef.current = null;
      }
      if (polygonDraftVertexLayerRef.current) {
        polygonDraftVertexLayerRef.current.clearLayers();
      }
      polygonDraftLatLngsRef.current = [];
      map.doubleClickZoom.enable();
    };

    const finishPolygonDraft = () => {
      const points = polygonDraftLatLngsRef.current;
      if (!polygonModeEnabledRef.current) return;
      if (points.length < 3) {
        setScanStatusText('Polygon mode needs at least 3 points.');
        return;
      }
      const polygonLayer = L.polygon(points, {
        color: '#e94560',
        weight: 2,
        opacity: 0.95,
        fillColor: '#e94560',
        fillOpacity: 0.18
      });
      drawnItemsRef.current.clearLayers();
      drawnItemsRef.current.addLayer(polygonLayer);
      applyLayerBounds(polygonLayer, 'polygon');
      setScanStatusText(`Polygon shape ready (${points.length} points).`);
      polygonModeEnabledRef.current = false;
      map.getContainer().style.cursor = '';
      clearPolygonDraftPreview();
      syncModeButtonStyles();
    };

    const finishFreeDrawShape = (releaseLatLng) => {
      if (!freeDrawEnabledRef.current || !freeDrawDrawingRef.current) return;
      freeDrawDrawingRef.current = false;
      map.dragging.enable();
      if (releaseLatLng) {
        const points = freeDrawLatLngsRef.current;
        const last = points[points.length - 1];
        if (!last || map.distance(last, releaseLatLng) >= 0.25) {
          points.push(releaseLatLng);
        }
      }
      if (freeDrawPreviewRef.current) {
        map.removeLayer(freeDrawPreviewRef.current);
        freeDrawPreviewRef.current = null;
      }
      const points = freeDrawLatLngsRef.current;
      if (points.length < 3) {
        setScanStatusText('Free draw needs a larger gesture (at least 3 points).');
        freeDrawLatLngsRef.current = [];
        return;
      }

      const polygonLayer = L.polygon(points, {
        color: '#e94560',
        weight: 2,
        opacity: 0.95,
        fillColor: '#e94560',
        fillOpacity: 0.18
      });
      drawnItemsRef.current.clearLayers();
      drawnItemsRef.current.addLayer(polygonLayer);
      applyLayerBounds(polygonLayer, 'polygon');
      setScanStatusText(`Free draw shape ready (${points.length} points).`);
      freeDrawLatLngsRef.current = [];
    };

    const togglePointMode = () => {
      pointModeEnabledRef.current = !pointModeEnabledRef.current;
      if (pointModeEnabledRef.current) {
        polygonModeEnabledRef.current = false;
        freeDrawEnabledRef.current = false;
        clearPolygonDraftPreview();
        clearFreeDrawPreview();
        setScanStatusText('Point mode enabled. Click map to set one-shot target.');
      } else {
        setScanStatusText('');
      }
      map.getContainer().style.cursor = pointModeEnabledRef.current ? 'crosshair' : '';
      syncModeButtonStyles();
    };

    const togglePolygonMode = () => {
      polygonModeEnabledRef.current = !polygonModeEnabledRef.current;
      if (polygonModeEnabledRef.current) {
        pointModeEnabledRef.current = false;
        freeDrawEnabledRef.current = false;
        clearFreeDrawPreview();
        clearPolygonDraftPreview();
        map.doubleClickZoom.disable();
        setScanStatusText('Polygon mode enabled. Click to add points, double-click to finish.');
        renderPolygonDraftPreview();
      } else {
        clearPolygonDraftPreview();
        setScanStatusText('');
      }
      map.getContainer().style.cursor = polygonModeEnabledRef.current ? 'crosshair' : '';
      syncModeButtonStyles();
    };

    const toggleFreeDrawMode = () => {
      freeDrawEnabledRef.current = !freeDrawEnabledRef.current;
      if (freeDrawEnabledRef.current) {
        pointModeEnabledRef.current = false;
        polygonModeEnabledRef.current = false;
        clearPolygonDraftPreview();
        setScanStatusText('Free draw enabled. Click and drag on map to sketch scan shape.');
      } else {
        clearFreeDrawPreview();
        setScanStatusText('');
      }
      map.getContainer().style.cursor = freeDrawEnabledRef.current ? 'crosshair' : '';
      syncModeButtonStyles();
    };

    const modeControl = L.control({ position: 'topleft' });
    modeControl.onAdd = () => {
      const container = L.DomUtil.create('div', 'leaflet-bar geospy-mode-control');
      const pointButton = L.DomUtil.create('a', 'geospy-mode-point', container);
      pointButton.href = '#';
      pointButton.textContent = 'Pt';
      const polygonButton = L.DomUtil.create('a', 'geospy-mode-polygon', container);
      polygonButton.href = '#';
      polygonButton.textContent = 'Pg';
      const freeDrawButton = L.DomUtil.create('a', 'geospy-mode-free', container);
      freeDrawButton.href = '#';
      freeDrawButton.textContent = 'FD';

      pointModeButtonRef.current = pointButton;
      polygonModeButtonRef.current = polygonButton;
      freeDrawButtonRef.current = freeDrawButton;

      L.DomEvent.disableClickPropagation(container);
      L.DomEvent.disableScrollPropagation(container);
      L.DomEvent.on(pointButton, 'click', (evt) => {
        L.DomEvent.stop(evt);
        togglePointMode();
      });
      L.DomEvent.on(polygonButton, 'click', (evt) => {
        L.DomEvent.stop(evt);
        togglePolygonMode();
      });
      L.DomEvent.on(freeDrawButton, 'click', (evt) => {
        L.DomEvent.stop(evt);
        toggleFreeDrawMode();
      });

      return container;
    };
    modeControl.addTo(map);
    syncModeButtonStyles();
    setMapControlTooltips(map);

    map.on(L.Draw.Event.CREATED, (e) => {
      clearFreeDrawPreview();
      clearPolygonDraftPreview();
      drawnItemsRef.current.clearLayers();
      drawnItemsRef.current.addLayer(e.layer);
      applyLayerBounds(e.layer, e.layerType);
    });

    map.on(L.Draw.Event.EDITED, (e) => {
      e.layers.eachLayer((layer) => {
        const type = layer instanceof L.Rectangle ? 'rectangle' : 'polygon';
        applyLayerBounds(layer, type);
      });
    });

    map.on(L.Draw.Event.DELETED, () => {
      clearPolygonDraftPreview();
      setScanShapeType(null);
      setSelectedPolygonCoords([]);
    });

    map.on('mousedown', (e) => {
      if (!freeDrawEnabledRef.current) return;
      if (e.originalEvent?.button !== 0) return;
      freeDrawDrawingRef.current = true;
      freeDrawLatLngsRef.current = [e.latlng];
      map.dragging.disable();
      if (freeDrawPreviewRef.current) {
        map.removeLayer(freeDrawPreviewRef.current);
      }
      freeDrawPreviewRef.current = L.polyline(freeDrawLatLngsRef.current, {
        color: '#2f8cff',
        weight: 3,
        opacity: 0.85,
        dashArray: '6 4'
      }).addTo(map);
    });

    map.on('mousemove', (e) => {
      if (!freeDrawEnabledRef.current || !freeDrawDrawingRef.current) return;
      const points = freeDrawLatLngsRef.current;
      const last = points[points.length - 1];
      if (last && map.distance(last, e.latlng) < 0.75) {
        return;
      }
      points.push(e.latlng);
      if (freeDrawPreviewRef.current) {
        freeDrawPreviewRef.current.setLatLngs(points);
      }
    });

    map.on('mouseup', (e) => {
      finishFreeDrawShape(e?.latlng || null);
    });

    map.on('click', (e) => {
      if (polygonModeEnabledRef.current) {
        const points = polygonDraftLatLngsRef.current;
        const last = points[points.length - 1];
        if (last && map.distance(last, e.latlng) < MIN_POLYGON_POINT_DISTANCE_METERS) {
          setScanStatusText(
            `Point ignored (too close). Move at least ${MIN_POLYGON_POINT_DISTANCE_METERS.toFixed(1)}m before placing next point.`
          );
          renderPolygonDraftPreview(e.latlng);
          return;
        }
        points.push(e.latlng);
        renderPolygonDraftPreview(e.latlng);
        setScanStatusText(`Polygon mode: ${points.length} points. Double-click to finish.`);
        return;
      }
      if (pointModeEnabledRef.current || e.originalEvent?.altKey) {
        setOneShotTarget(e.latlng.lat, e.latlng.lng);
      }
    });

    map.on('mousemove', (e) => {
      if (polygonModeEnabledRef.current && !freeDrawDrawingRef.current) {
        renderPolygonDraftPreview(e.latlng);
      }
    });

    map.on('dblclick', () => {
      if (polygonModeEnabledRef.current) {
        finishPolygonDraft();
      }
    });

    map.on('contextmenu', (e) => {
      if (polygonModeEnabledRef.current) {
        finishPolygonDraft();
        return;
      }
      setOneShotTarget(e.latlng.lat, e.latlng.lng);
    });

    const scheduleViewportRefresh = (delayMs = 220) => {
      if (battlefrontMapModeActiveRef.current) {
        if (markersRef.current) {
          markersRef.current.clearLayers();
        }
        return;
      }
      if (viewportDebounceRef.current) {
        clearTimeout(viewportDebounceRef.current);
      }
      viewportDebounceRef.current = setTimeout(() => {
        loadData({ forceStats: false });
      }, delayMs);
    };
    map.on('moveend', () => scheduleViewportRefresh(120));
    map.on('zoomend', () => scheduleViewportRefresh(120));

    loadData({ forceStats: true });
    autoRefreshRef.current = setInterval(() => {
      loadData({ forceStats: true, refreshViewport: false });
    }, 30000);

    return () => {
      if (scanPollRef.current) clearInterval(scanPollRef.current);
      if (autoRefreshRef.current) clearInterval(autoRefreshRef.current);
      if (viewportDebounceRef.current) clearTimeout(viewportDebounceRef.current);
      if (viewportAbortRef.current) viewportAbortRef.current.abort();
      map.doubleClickZoom.enable();
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    currentPathRef.current = location.pathname;
  }, [location.pathname]);

  const stopLocateBattlefrontAudio = (options = {}) => {
    const { reset = true } = options;
    if (locateBattlefrontAudioStopTimeoutRef.current) {
      window.clearTimeout(locateBattlefrontAudioStopTimeoutRef.current);
      locateBattlefrontAudioStopTimeoutRef.current = null;
    }
    const audio = locateBattlefrontAudioRef.current;
    if (!audio) return;
    audio.pause();
    if (reset) {
      try {
        audio.currentTime = 0;
      } catch (error) {
        console.debug('battlefront audio reset failed', error);
      }
    }
  };

  const playLocateBattlefrontAudio = (playbackMs = 0) => {
    if (typeof window === 'undefined' || typeof Audio === 'undefined') {
      return;
    }
    let audio = locateBattlefrontAudioRef.current;
    if (!audio) {
      audio = new Audio(locateBattlefrontAudioPath);
      audio.preload = 'auto';
      audio.loop = false;
      audio.volume = 0.72;
      locateBattlefrontAudioRef.current = audio;
    }
    if (locateBattlefrontAudioStopTimeoutRef.current) {
      window.clearTimeout(locateBattlefrontAudioStopTimeoutRef.current);
      locateBattlefrontAudioStopTimeoutRef.current = null;
    }
    audio.pause();
    try {
      audio.currentTime = 0;
    } catch (error) {
      console.debug('battlefront audio seek failed', error);
    }
    void audio.play().catch((error) => {
      console.debug('battlefront audio playback blocked', error);
    });
    const normalizedPlaybackMs = Math.max(0, Number(playbackMs || 0));
    if (normalizedPlaybackMs > 0) {
      locateBattlefrontAudioStopTimeoutRef.current = window.setTimeout(() => {
        const activeAudio = locateBattlefrontAudioRef.current;
        if (!activeAudio) return;
        activeAudio.pause();
        try {
          activeAudio.currentTime = 0;
        } catch (error) {
          console.debug('battlefront audio stop failed', error);
        }
        locateBattlefrontAudioStopTimeoutRef.current = null;
      }, normalizedPlaybackMs);
    }
  };

  useEffect(() => {
    if (typeof window === 'undefined' || typeof Audio === 'undefined') {
      return undefined;
    }
    const audio = new Audio(locateBattlefrontAudioPath);
    audio.preload = 'auto';
    audio.loop = false;
    audio.volume = 0.72;
    audio.load();
    locateBattlefrontAudioRef.current = audio;
    return () => {
      if (locateBattlefrontAudioStopTimeoutRef.current) {
        window.clearTimeout(locateBattlefrontAudioStopTimeoutRef.current);
        locateBattlefrontAudioStopTimeoutRef.current = null;
      }
      audio.pause();
      try {
        audio.currentTime = 0;
      } catch (error) {
        console.debug('battlefront audio cleanup failed', error);
      }
      if (locateBattlefrontAudioRef.current === audio) {
        locateBattlefrontAudioRef.current = null;
      }
    };
  }, []);

  const clearLocateBattlefrontSequence = (options = {}) => {
    locateBattlefrontTimeoutsRef.current.forEach((timeoutId) => {
      window.clearTimeout(timeoutId);
    });
    locateBattlefrontTimeoutsRef.current = [];
    if (locateBattlefrontSearchReadyTimeoutRef.current) {
      window.clearTimeout(locateBattlefrontSearchReadyTimeoutRef.current);
      locateBattlefrontSearchReadyTimeoutRef.current = null;
    }
    if (
      mapRef.current &&
      locateBattlefrontSearchMoveEndHandlerRef.current &&
      typeof mapRef.current.off === 'function'
    ) {
      mapRef.current.off('moveend', locateBattlefrontSearchMoveEndHandlerRef.current);
      locateBattlefrontSearchMoveEndHandlerRef.current = null;
    }
    locateBattlefrontSearchReadyPromiseRef.current = Promise.resolve();
    if (options.stopMap && mapRef.current && typeof mapRef.current.stop === 'function') {
      mapRef.current.stop();
    }
    if (!options.keepAudio) {
      stopLocateBattlefrontAudio();
    }
    setLocateBattlefrontSequence(null);
  };

  const queueLocateBattlefrontStep = (delayMs, callback) => {
    const timeoutId = window.setTimeout(callback, delayMs);
    locateBattlefrontTimeoutsRef.current.push(timeoutId);
  };

  const waitForLocateBattlefrontSearchReady = async () => {
    const ready = locateBattlefrontSearchReadyPromiseRef.current;
    if (ready && typeof ready.then === 'function') {
      await ready;
    }
  };

  const beginLocateBattlefrontSearch = () => {
    const map = mapRef.current;
    if (!map) return;
    clearLocateBattlefrontSequence({ stopMap: true });
    locateBattlefrontSearchStartedAtRef.current = performance.now();
    locateBattlefrontSearchReadyPromiseRef.current = new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        if (
          map &&
          locateBattlefrontSearchMoveEndHandlerRef.current &&
          typeof map.off === 'function'
        ) {
          map.off('moveend', locateBattlefrontSearchMoveEndHandlerRef.current);
        }
        locateBattlefrontSearchMoveEndHandlerRef.current = null;
        if (locateBattlefrontSearchReadyTimeoutRef.current) {
          window.clearTimeout(locateBattlefrontSearchReadyTimeoutRef.current);
          locateBattlefrontSearchReadyTimeoutRef.current = null;
        }
        resolve();
      };
      const handleMoveEnd = () => {
        window.setTimeout(finish, 180);
      };
      locateBattlefrontSearchMoveEndHandlerRef.current = handleMoveEnd;
      if (typeof map.once === 'function') {
        map.once('moveend', handleMoveEnd);
      }
      locateBattlefrontSearchReadyTimeoutRef.current = window.setTimeout(
        finish,
        locateBattlefrontSearchZoomDurationMs + 700
      );
    });
    const center = map.getCenter();
    setLocateBattlefrontSequence({
      phase: 'searching',
      title: 'Scanning planetary archive',
      subtitle: 'Preparing a wide-area vector sweep before the landing lock begins.',
      targetLabel: 'pending lock',
      previewPath: '',
      familyScore: 0,
      lat: center.lat,
      lon: center.lng,
      stepIndex: 0,
      lineLeadMs: 0,
      lineDurationMs: 0,
      linePauseMs: 0,
      zoomDurationMs: 0,
      isZooming: false
    });
    map.flyTo([center.lat, center.lng], 2, {
      animate: true,
      duration: 3.9,
      easeLinearity: 0.1
    });
  };

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
  }, [isLocatePage]);

  useEffect(() => {
    if (
      location.pathname !== '/scan' &&
      location.pathname !== '/search' &&
      location.pathname !== '/locate'
    ) {
      navigate('/scan', { replace: true });
    }
  }, [location.pathname, navigate]);

  const setMapControlTooltips = (map) => {
    const mapContainer = map.getContainer();
    const controlTitles = [
      ['.leaflet-control-zoom-in', 'Zoom in'],
      ['.leaflet-control-zoom-out', 'Zoom out'],
      ['.leaflet-draw-edit-edit', 'Edit selected scan shape'],
      ['.leaflet-draw-edit-remove', 'Delete selected scan shape'],
      ['.geospy-mode-point', 'Point mode: click map to place one-shot target'],
      ['.geospy-mode-polygon', 'Polygon mode: click points and double-click to finish'],
      ['.geospy-mode-free', 'Free draw mode: click and drag to sketch scan area']
    ];
    controlTitles.forEach(([selector, title]) => {
      const el = mapContainer.querySelector(selector);
      if (!el) return;
      el.setAttribute('title', title);
      el.setAttribute('aria-label', title);
      el.setAttribute('data-tip', title);
      el.classList.add('geospy-has-tooltip');
    });
  };

  useEffect(() => {
    if (geojsonData) renderMarkers(geojsonData);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorMode]);

  useEffect(() => {
    activeScanIdRef.current = activeScanId;
  }, [activeScanId]);

  useEffect(() => {
    trackedScanIdsRef.current = trackedScanIds;
    if (trackedScanIds.length) {
      window.localStorage.setItem(ACTIVE_SCANS_STORAGE_KEY, JSON.stringify(trackedScanIds));
    } else {
      window.localStorage.removeItem(ACTIVE_SCANS_STORAGE_KEY);
      window.localStorage.removeItem(LEGACY_ACTIVE_SCAN_STORAGE_KEY);
    }
  }, [trackedScanIds]);

  useEffect(() => {
    if (!retrievalFile) {
      setRetrievalFilePreviewUrl('');
      return undefined;
    }
    const objectUrl = URL.createObjectURL(retrievalFile);
    setRetrievalFilePreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [retrievalFile]);

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

  useEffect(() => {
    if (!locateBusy || !locateProgressId) {
      return undefined;
    }
    let cancelled = false;
    const pollProgress = async () => {
      try {
        const res = await fetch(`/api/retrieval/progress/${encodeURIComponent(locateProgressId)}`, {
          cache: 'no-store'
        });
        if (!res.ok) {
          return;
        }
        const body = await res.json();
        if (cancelled) {
          return;
        }
        setLocateLiveProgress(body);
        setLocatePipeline({ stages: buildLocatePipelineStagesFromProgress(body) });
        if (String(body?.message || '').trim()) {
          setLocateStatus(String(body.message).trim());
        }
      } catch (error) {
        console.error('locate progress error', error);
      }
    };
    pollProgress();
    const intervalId = window.setInterval(pollProgress, 700);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [locateBusy, locateProgressId]);

  const setOneShotTarget = (lat, lon) => {
    setOneShotLat(lat.toFixed(7));
    setOneShotLon(lon.toFixed(7));
    if (!mapRef.current) return;

    if (oneShotMarkerRef.current) {
      mapRef.current.removeLayer(oneShotMarkerRef.current);
    }

    oneShotMarkerRef.current = L.circleMarker([lat, lon], {
      radius: 7,
      fillColor: '#ffd166',
      fillOpacity: 0.95,
      color: '#111',
      weight: 2
    }).addTo(mapRef.current);
    oneShotMarkerRef.current.bindTooltip('One-shot target');
  };

  const clearSearchFallbackMarker = () => {
    const map = mapRef.current;
    if (!map || !searchFallbackMarkerRef.current) return;
    map.removeLayer(searchFallbackMarkerRef.current);
    searchFallbackMarkerRef.current = null;
  };

  const buildStreetViewUrl = (result) => {
    const lat = Number(result?.lat);
    const lon = Number(result?.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return '';
    }
    const params = new URLSearchParams();
    params.set('api', '1');
    params.set('map_action', 'pano');
    params.set('viewpoint', `${lat},${lon}`);
    const heading = Number(result?.heading);
    if (Number.isFinite(heading)) {
      params.set('heading', String(heading));
    }
    const panoId = String(result?.pano_id || '').trim();
    if (panoId) {
      params.set('pano', panoId);
    }
    return `https://www.google.com/maps/@?${params.toString()}`;
  };

  const loadData = async (options = {}) => {
    const { forceStats = true, refreshViewport = true } = options;
    try {
      const map = mapRef.current;
      const currentBattlefrontMapModeActive = battlefrontMapModeActiveRef.current;
      const currentBattlefrontMarkersHidden = battlefrontMarkersHiddenRef.current;
      if (refreshViewport && map && !currentBattlefrontMarkersHidden) {
        const bounds = map.getBounds();
        const zoom = Math.max(1, Math.round(map.getZoom()));
        const pointLimit = currentBattlefrontMapModeActive
          ? zoom >= 16
            ? 5000
            : zoom >= 12
              ? 4200
              : 3200
          : zoom >= 18
            ? 5000
            : zoom >= 16
              ? 3200
              : 1800;
        const clusterZoomThreshold = currentBattlefrontMapModeActive ? 0 : 16;
        const params = new URLSearchParams({
          min_lat: String(bounds.getSouth()),
          min_lon: String(bounds.getWest()),
          max_lat: String(bounds.getNorth()),
          max_lon: String(bounds.getEast()),
          zoom: String(zoom),
          limit: String(pointLimit),
          cluster_zoom_threshold: String(clusterZoomThreshold)
        });
        if (viewportAbortRef.current) {
          viewportAbortRef.current.abort();
        }
        const controller = new AbortController();
        viewportAbortRef.current = controller;
        const panoRes = await fetch(`/api/panoramas/bbox?${params.toString()}`, {
          signal: controller.signal
        });
        if (!panoRes.ok) {
          throw new Error(`bbox fetch failed (${panoRes.status})`);
        }
        const geo = await panoRes.json();
        setGeojsonData(geo);
        renderMarkers(geo);
      }

      if (forceStats) {
        const statsRes = await fetch('/api/stats');
        if (!statsRes.ok) {
          throw new Error(`stats fetch failed (${statsRes.status})`);
        }
        const s = await statsRes.json();
        setStats(s);
      if (!hasAutoFittedBoundsRef.current && s?.bounds?.min_lat != null && mapRef.current) {
        mapRef.current.fitBounds(
          [
            [s.bounds.min_lat, s.bounds.min_lon],
            [s.bounds.max_lat, s.bounds.max_lon]
          ],
          { padding: [50, 50] }
        );
        hasAutoFittedBoundsRef.current = true;
        }
      }
    } catch (error) {
      if (error?.name === 'AbortError') return;
      console.error('Failed to load data', error);
    }
  };

  useEffect(() => {
    if (!mapRef.current) return;
    if (battlefrontMarkersHidden && markersRef.current) {
      markersRef.current.clearLayers();
      return;
    }
    loadData({ forceStats: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [battlefrontMapModeActive, battlefrontMarkersHidden]);

  const renderMarkers = (geojson) => {
    if (!markersRef.current) return;
    markersRef.current.clearLayers();
    if (battlefrontMarkersHiddenRef.current) return;
    const features = geojson?.features || [];
    if (!features.length) return;
    const renderer = markerRendererRef.current || undefined;

    const pointFeatures = features.filter((f) => !f?.properties?.cluster);
    const timestamps = pointFeatures.length
      ? pointFeatures.map((f) => new Date(f.properties.timestamp).getTime())
      : [Date.now()];
    const minT = Math.min(...timestamps);
    const maxT = Math.max(...timestamps);
    const rangeT = maxT - minT || 1;

    features.forEach((f) => {
      const [lon, lat] = f.geometry.coordinates;
      const props = f.properties || {};
      if (props.cluster) {
        const count = Number(props.point_count || 0);
        const marker = L.circleMarker([lat, lon], {
          radius: Math.max(8, Math.min(24, 8 + Math.sqrt(count))),
          fillColor: '#3c78d8',
          fillOpacity: 0.78,
          color: '#fff',
          weight: 1,
          renderer
        });
        marker.bindTooltip(`${count} points`);
        marker.on('click', () => {
          const map = mapRef.current;
          if (!map) return;
          map.setView([lat, lon], Math.min(20, map.getZoom() + 2));
        });
        markersRef.current.addLayer(marker);
        return;
      }
      let color = '#e94560';

      if (colorMode === 'recency') {
        const t = new Date(props.timestamp).getTime();
        color = interpolateColor('#0f3460', '#e94560', (t - minT) / rangeT);
      } else if (colorMode === 'heading') {
        const hue = (props.heading / 360) * 300;
        color = `hsl(${hue}, 70%, 50%)`;
      }

      const marker = L.circleMarker([lat, lon], {
        radius: 6,
        fillColor: color,
        fillOpacity: 0.8,
        color: '#fff',
        weight: 1,
        renderer
      });
      marker.on('click', () => showPreview(props));
      markersRef.current.addLayer(marker);
    });
  };

  const showPreview = async (props) => {
    try {
      const res = await fetch(`/api/panorama/${props.id}`);
      const captures = await res.json();
      setPreview({
        title: `${props.id} | pano: ${props.pano_id || 'N/A'}`,
        captures: captures.map((c) => ({
          id: c.id,
          heading: c.heading,
          src: String(c.web_path || '').trim()
        }))
      });
    } catch {
      setPreview({ title: 'Preview error', captures: [] });
    }
  };

  const runOneShot = async () => {
    const lat = Number(oneShotLat);
    const lon = Number(oneShotLon);
    if (Number.isNaN(lat) || Number.isNaN(lon)) {
      setOneShotStatus('Set a target first (Alt+Click or right-click map).');
      return;
    }

    setOneShotStatus(`Capturing at ${lat.toFixed(6)}, ${lon.toFixed(6)}...`);
    try {
      const res = await fetch('/api/capture-once', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lon })
      });
      const body = await res.json();
      if (!res.ok) {
        const detail = body?.detail;
        setOneShotStatus(`Capture failed: ${typeof detail === 'string' ? detail : detail?.message || 'error'}`);
        return;
      }
      setOneShotStatus('Capture complete. Refreshing map...');
      await loadData();
      setOneShotStatus('Capture complete.');
    } catch (error) {
      setOneShotStatus(`Capture failed: ${error.message}`);
    }
  };

  const startScanPolling = () => {
    if (scanPollRef.current) {
      clearInterval(scanPollRef.current);
    }
    scanPollRef.current = setInterval(() => {
      pollScanStatus();
    }, 3000);
  };

  const stopScanPolling = () => {
    if (scanPollRef.current) {
      clearInterval(scanPollRef.current);
      scanPollRef.current = null;
    }
  };

  const isScanTerminalStatus = (status) => {
    const value = String(status || '');
    return value === 'finished' || value === 'stopped' || value.startsWith('failed:');
  };

  const isScanActiveStatus = (status) => {
    const value = String(status || '');
    return value === 'running' || value === 'stopping' || value.startsWith('running');
  };

  const addTrackedScanId = (scanId, makePrimary = false) => {
    if (!scanId) return;
    setTrackedScanIds((prev) => {
      if (prev.includes(scanId)) return prev;
      return [...prev, scanId];
    });
    if (makePrimary || !activeScanIdRef.current) {
      setActiveScanId(scanId);
    }
  };

  const removeTrackedScanId = (scanId) => {
    if (!scanId) return;
    setTrackedScanIds((prev) => prev.filter((id) => id !== scanId));
    if (activeScanIdRef.current === scanId) {
      setActiveScanId((prevActive) => {
        if (prevActive !== scanId) return prevActive;
        const remaining = trackedScanIdsRef.current.filter((id) => id !== scanId);
        return remaining[0] || null;
      });
    }
  };

  const startScan = async () => {
    const payload = {
      min_lat: Number(scanForm.minLat),
      min_lon: Number(scanForm.minLon),
      max_lat: Number(scanForm.maxLat),
      max_lon: Number(scanForm.maxLon),
      polygon_coords: scanShapeType === 'polygon' ? selectedPolygonCoords : null,
      num_workers: Number(scanForm.workers) || 4,
      step_meters: Number(scanForm.stepMeters) || 50,
      dedup_radius: Number(scanForm.dedupRadius) || 25,
      mode: scanForm.mode,
      job_type: scanForm.jobType || 'scan',
      capture_profile: scanForm.captureProfile || 'high_v1',
      fill_gap_meters: Number(scanForm.fillGapMeters) || 40,
      enrich_missing_only: Boolean(scanForm.enrichMissingOnly)
    };

    setScanStatusText('Submitting scan...');
    try {
      const res = await fetch('/api/scan-area', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const body = await res.json();
      if (!res.ok) {
        setScanStatusText(`Scan failed: ${body?.detail || 'unknown error'}`);
        return;
      }

      addTrackedScanId(body.scan_id, true);
      setScanStatusText(
        `${String(body.job_type || payload.job_type).toUpperCase()} ${body.scan_id} started: ${body.land_seeds} targets, ${body.workers_spawned} workers ` +
          `(water removed ${body.water_filtered}, gap removed ${body.gap_filtered || 0}${body.polygon_filtered_out ? `, outside-shape ${body.polygon_filtered_out}` : ''})`
      );

      startScanPolling();
      pollScanStatus(body.scan_id);
    } catch (error) {
      setScanStatusText(`Scan failed: ${error.message}`);
    }
  };

  const stopScan = async (scanIdOverride = null) => {
    try {
      const targetScanId = scanIdOverride || activeScanIdRef.current || trackedScanIdsRef.current[0] || null;
      const payload = targetScanId ? { scan_id: targetScanId } : {};
      const res = await fetch('/api/scan-stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const body = await res.json();
      const localStopped = Number(body?.stopped || 0);
      const modalSignals = Number(body?.modal_stop_signals || 0);
      setScanStatusText(
        `Stop requested${targetScanId ? ` for ${targetScanId}` : ''}. local_signals=${localStopped}, modal_signals=${modalSignals}.`
      );
      if (!scanPollRef.current) {
        startScanPolling();
      }
      pollScanStatus(targetScanId);
    } catch (error) {
      setScanStatusText(`Stop failed: ${error.message}`);
    }
  };

  const pollScanStatus = async (scanIdOverride = null, options = {}) => {
    const allowAutoAttach = Boolean(options.allowAutoAttach);
    try {
      const res = await fetch('/api/scan-status');
      const data = await res.json();
      const q = data.queue || {};
      const scans = data.scans || [];
      const activeScans = scans.filter((item) => isScanActiveStatus(item.status));
      const sortedScans = [...scans].sort((a, b) => {
        const aActive = isScanActiveStatus(a.status) ? 0 : 1;
        const bActive = isScanActiveStatus(b.status) ? 0 : 1;
        if (aActive !== bActive) return aActive - bActive;
        return String(a.scan_id || '').localeCompare(String(b.scan_id || ''));
      });
      setScanJobs(sortedScans.slice(0, 30));

      const existingIds = new Set(scans.map((item) => item.scan_id));
      const nextTracked = trackedScanIdsRef.current.filter((scanId) => existingIds.has(scanId));
      activeScans.forEach((scanItem) => {
        if (!nextTracked.includes(scanItem.scan_id)) {
          nextTracked.push(scanItem.scan_id);
        }
      });
      const trackedChanged =
        nextTracked.length !== trackedScanIdsRef.current.length ||
        nextTracked.some((id, idx) => id !== trackedScanIdsRef.current[idx]);
      if (trackedChanged) {
        setTrackedScanIds(nextTracked);
      }

      let targetScanId =
        scanIdOverride || activeScanIdRef.current || nextTracked[0] || null;
      let scan = targetScanId ? scans.find((item) => item.scan_id === targetScanId) : null;

      if ((!scan || isScanTerminalStatus(scan.status)) && allowAutoAttach) {
        scan = activeScans[0] || null;
        if (scan) {
          targetScanId = scan.scan_id;
          setActiveScanId(scan.scan_id);
          setScanStatusText(`Reattached to scan ${scan.scan_id}.`);
        } else if (targetScanId) {
          setActiveScanId(null);
        }
      }

      if (scan?.mode === 'modal') {
        const p = scan.modal_progress || {};
        const workersTotal = p.workers_total || 0;
        const workersSubmitted = p.workers_submitted || 0;
        const workersCompleted = p.workers_completed || 0;
        const workersFailed = p.workers_failed || 0;
        const workersCancelled = p.workers_cancelled || 0;
        const workersRunning = Math.max(
          0,
          workersSubmitted - workersCompleted - workersFailed - workersCancelled
        );

        setScanProgress({
          pending: Math.max(0, workersTotal - workersSubmitted),
          inProgress: workersRunning,
          done: workersCompleted,
          skipped: p.retries_queued || 0,
          failed: workersFailed,
          workers: workersRunning
        });

        setScanStatusText(
          `Tracking ${scan.scan_id}: modal ${workersCompleted}/${workersTotal} complete, ${workersFailed} failed, ${workersCancelled} cancelled, ${workersRunning} running`
        );

        if (isScanTerminalStatus(scan.status)) {
          removeTrackedScanId(scan.scan_id);
          if (scan.status === 'finished') setScanStatusText(`Scan ${scan.scan_id} complete.`);
          if (scan.status === 'stopped') setScanStatusText(`Scan ${scan.scan_id} stopped.`);
        }
      } else if (scan) {
        const workersAlive = Number(scan.workers_alive || 0);
        let totalAlive = 0;
        scans.forEach((s) => {
          totalAlive += s.workers_alive || 0;
        });
        setScanProgress({
          pending: q.pending || 0,
          inProgress: q.in_progress || 0,
          done: q.done || 0,
          skipped: q.skipped || 0,
          failed: q.failed || 0,
          workers: workersAlive || totalAlive
        });
        setScanStatusText(
          `Tracking ${scan.scan_id}: ${String(scan.job_type || 'scan')} ${String(scan.status || 'running')}, workers=${workersAlive}`
        );
        if (scan && isScanTerminalStatus(scan.status)) {
          removeTrackedScanId(scan.scan_id);
          if (scan.status === 'finished') setScanStatusText(`Scan ${scan.scan_id} complete.`);
          if (scan.status === 'stopped') setScanStatusText(`Scan ${scan.scan_id} stopped.`);
        }
      } else {
        let totalAlive = 0;
        scans.forEach((s) => {
          totalAlive += s.workers_alive || 0;
        });
        setScanProgress({
          pending: q.pending || 0,
          inProgress: q.in_progress || 0,
          done: q.done || 0,
          skipped: q.skipped || 0,
          failed: q.failed || 0,
          workers: totalAlive
        });
      }

      setStats((prev) => ({
        ...prev,
        total_panoramas: data.panoramas ?? prev.total_panoramas,
        total_captures: data.captures ?? prev.total_captures
      }));
      if (activeScans.length === 0 && nextTracked.length === 0) {
        stopScanPolling();
      }
      return scan;
    } catch (error) {
      console.error('scan poll error', error);
      return null;
    }
  };

  const loadRetrievalStats = async () => {
    try {
      const res = await fetch('/api/retrieval/index-stats');
      if (!res.ok) return;
      const body = await res.json();
      const models = Array.isArray(body.models) ? body.models : [];
      const baseToModel = new Map();
      models.forEach((modelStats) => {
        const base = String(modelStats.embedding_base || '').trim().toLowerCase();
        if (!base || baseToModel.has(base)) return;
        baseToModel.set(base, modelStats);
      });
      const normalizedOptions = baseToModel.size ? Array.from(baseToModel.keys()) : ['clip'];
      setRetrievalEmbeddingBaseOptions(
        normalizedOptions.map((value) => ({
          value,
          label: formatRetrievalModelLabel(
            baseToModel.get(value)?.model_name,
            baseToModel.get(value)?.model_version
          ) || value.toUpperCase()
        }))
      );
      const resolveDefaultEmbeddingBase = (prev) => {
        if (normalizedOptions.includes(prev)) return prev;
        if (normalizedOptions.includes('place')) return 'place';
        return normalizedOptions[0];
      };
      setSearchEmbeddingBase((prev) => resolveDefaultEmbeddingBase(prev));
      setLocateEmbeddingBase((prev) => resolveDefaultEmbeddingBase(prev));
      setRetrievalStats({
        model_name: body.model_name || '',
        model_version: body.model_version || '',
        total_captures: Number(body.total_captures || 0),
        embedded_captures: Number(body.embedded_captures || 0),
        pending_captures: Number(body.pending_captures || 0),
        models
      });
    } catch (error) {
      console.error('retrieval stats error', error);
    }
  };

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

  const runImageLocate = async () => {
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

  const startLocateBattlefrontReveal = async (result, options = {}) => {
    const map = mapRef.current;
    const lat = Number(result?.lat ?? result?.family_center_lat);
    const lon = Number(result?.lon ?? result?.family_center_lon);
    if (!map || !Number.isFinite(lat) || !Number.isFinite(lon)) {
      await focusRetrievalResult(result);
      if (typeof options.onComplete === 'function') {
        options.onComplete();
      }
      return;
    }

    clearLocateBattlefrontSequence({ stopMap: true });
    clearSearchFallbackMarker();

    const targetLabel = String(result?.pano_id || '').trim()
      ? `pano ${String(result.pano_id).trim()}`
      : result?.family_id != null
        ? `family ${result.family_id}`
        : 'matched landing zone';
    const previewPath = String(result?.web_path || '').trim();
    const familyScore = Number(result?.family_score || 0);

    locateBattlefrontRevealStages.forEach((stage, index) => {
      const runStage = () => {
        setLocateBattlefrontSequence({
          phase: stage.key,
          title: stage.title,
          subtitle: stage.subtitle,
          targetLabel,
          previewPath,
          familyScore: Number.isFinite(familyScore) ? familyScore : 0,
          lat,
          lon,
          stepIndex: index + 1,
          lineLeadMs: Number(stage.lineLeadMs || 0),
          lineDurationMs: Number(stage.lineDurationMs || 0),
          linePauseMs: Number(stage.pauseAfterLinesMs || 0),
          zoomDurationMs: Math.max(0, Math.round(Number(stage.duration || 0) * 1000)),
          isZooming: false
        });
        queueLocateBattlefrontStep(Number(stage.zoomDelayMs || stage.lineLeadMs || 0), () => {
          playLocateBattlefrontAudio(
            Math.max(
              0,
              Math.round(Number(stage.duration || 0) * 1000) + Number(stage.settleMs || 0)
            )
          );
          setLocateBattlefrontSequence((prev) =>
            prev && prev.phase === stage.key
              ? {
                  ...prev,
                  isZooming: true
                }
              : prev
          );
          if (typeof map.stop === 'function') {
            map.stop();
          }
          map.flyTo([lat, lon], stage.zoom, {
            animate: true,
            duration: stage.duration,
            easeLinearity: 0.12
          });
        });
      };
      if (stage.at <= 0) {
        runStage();
      } else {
        queueLocateBattlefrontStep(stage.at, runStage);
      }
    });

    queueLocateBattlefrontStep(locateBattlefrontLockDelayMs, () => {
      if (typeof map.stop === 'function') {
        map.stop();
      }
      setLocateBattlefrontSequence((prev) =>
        prev
          ? {
              ...prev,
              phase: 'locked',
              title: 'Landing solution confirmed',
              subtitle: 'Locate target committed to the map and preview dock.',
              lineLeadMs: 0,
              lineDurationMs: 0,
              linePauseMs: 0,
              zoomDurationMs: 0,
              isZooming: false
            }
          : null
      );
      void focusRetrievalResult(result, {
        keepBattlefrontReveal: true,
        skipMapMove: true,
        skipPreview: true,
        preferredZoom: Math.max(
          20,
          Number(locateBattlefrontRevealStages[locateBattlefrontRevealStages.length - 1]?.zoom || 20)
        )
      });
      if (typeof options.onComplete === 'function') {
        options.onComplete();
      }
    });

    queueLocateBattlefrontStep(locateBattlefrontClearDelayMs, () => {
      clearLocateBattlefrontSequence();
    });
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

  useEffect(() => {
    let isCancelled = false;
    const hydrateScanAndRetrievalState = async () => {
      await loadRetrievalStats();
      const storedIdsRaw = window.localStorage.getItem(ACTIVE_SCANS_STORAGE_KEY);
      const legacyScanId = window.localStorage.getItem(LEGACY_ACTIVE_SCAN_STORAGE_KEY);
      let storedScanIds = [];
      try {
        storedScanIds = storedIdsRaw ? JSON.parse(storedIdsRaw) : [];
      } catch {
        storedScanIds = [];
      }
      if ((!Array.isArray(storedScanIds) || storedScanIds.length === 0) && legacyScanId) {
        storedScanIds = [legacyScanId];
      }
      const normalizedStored = Array.isArray(storedScanIds)
        ? storedScanIds.filter((scanId) => typeof scanId === 'string' && scanId.trim() !== '')
        : [];
      if (normalizedStored.length) {
        setTrackedScanIds(normalizedStored);
        setActiveScanId(normalizedStored[0]);
      }
      const attachedScan = await pollScanStatus(normalizedStored[0] || null, { allowAutoAttach: true });
      if (isCancelled) {
        return;
      }
      if (attachedScan && isScanActiveStatus(attachedScan.status)) {
        addTrackedScanId(attachedScan.scan_id, true);
        startScanPolling();
      }
    };
    hydrateScanAndRetrievalState();
    return () => {
      isCancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onScanField = (field, value) => {
    setScanForm((prev) => ({ ...prev, [field]: value }));
  };

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
  const fallbackLocateOrbPopupMessage = orbPopupMoments[locateOrbPopupMoment % orbPopupMoments.length];
  const liveLocatePhase = String(liveLocateProgress?.phase || '').trim().toLowerCase();
  const locateOrbPopupMessage =
    locateOrbPopupPhase === 'processing'
      ? String(liveLocateProgress?.message || '').trim() || fallbackLocateOrbPopupMessage
      : fallbackLocateOrbPopupMessage;
  const locateOrbPreviewStage =
    locateOrbPopupPhase === 'processing'
      ? mapLocateProgressPhaseToPreviewStage(liveLocatePhase, liveLocateOrbStats)
      : orbPopupPreviewStages[locateOrbPopupMoment % orbPopupPreviewStages.length];
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
  const locateModelLabel =
    retrievalEmbeddingBaseOptions.find((option) => option.value === locateEmbeddingBase)?.label ||
    String(locateEmbeddingBase || 'active').toUpperCase();
  const locateBattlefrontStepIndex = Math.max(0, Number(locateBattlefrontSequence?.stepIndex || 0));
  const locateBattlefrontSearching = locateBattlefrontSequence?.phase === 'searching';
  const locateBattlefrontHudTitle =
    locateBattlefrontSearching
      ? liveLocatePhase === 'vector_search'
        ? 'Country-scale vector sweep'
        : liveLocatePhase === 'orb_rerank'
          ? 'Local feature verification'
          : liveLocatePhase === 'panorama_rerank'
            ? 'Panorama merge underway'
            : liveLocatePhase === 'family_rank'
              ? 'Locking the landing zone'
              : 'Scanning planetary archive'
      : String(locateBattlefrontSequence?.title || '').trim();
  const locateBattlefrontHudSubtitle =
    locateBattlefrontSearching
      ? String(liveLocateProgress?.message || '').trim() ||
        'Zoomed out while the backend resolves a landing zone from the live locate pipeline.'
      : String(locateBattlefrontSequence?.subtitle || '').trim();
  const locateBattlefrontMetaItems = locateBattlefrontSearching
    ? [
        `model ${locateModelLabel}`,
        `phase ${String(formatLocateProgressStageBadge(liveLocatePhase, locateOrbEnabled)).toLowerCase()}`,
        Number(liveLocateProgress?.vector_candidates || 0) > 0
          ? `raw ${Number(liveLocateProgress?.vector_candidates || 0)}`
          : '',
        locateOrbEnabled && locateOrbCandidateCount > 0
          ? `orb ${locateOrbProcessedCandidates}/${locateOrbCandidateCount}`
          : locateOrbEnabled
            ? 'orb queued'
            : 'orb off',
        Number(liveLocateProgress?.panorama_candidates || 0) > 0
          ? `panos ${Number(liveLocateProgress?.panorama_candidates || 0)}`
          : '',
        Number(liveLocateProgress?.matches || 0) > 0
          ? `families ${Number(liveLocateProgress?.matches || 0)}`
          : ''
      ].filter(Boolean)
    : [
        `target ${String(locateBattlefrontSequence?.targetLabel || '').trim() || 'pending lock'}`,
        `lat ${formatCoordinate(locateBattlefrontSequence?.lat)}`,
        `lon ${formatCoordinate(locateBattlefrontSequence?.lon)}`,
        `score ${formatMetric(locateBattlefrontSequence?.familyScore)}`
      ];
  const locateBattlefrontPreviewLabel = locateBattlefrontSearching ? 'Reference image' : 'Recovered target';
  const locateBattlefrontPreviewPath =
    locateBattlefrontSearching
      ? String(retrievalFilePreviewUrl || '').trim()
      : String(locateBattlefrontSequence?.previewPath || '').trim();
  const locateBattlefrontProgressStages = locateBattlefrontSearching
    ? (locatePipeline.stages || []).map((stage) => ({
        key: stage.key,
        label: stage.title,
        status:
          String(stage.status || '').toLowerCase() === 'completed'
            ? 'done'
            : String(stage.status || '').toLowerCase() === 'running'
              ? 'active'
              : String(stage.status || '').toLowerCase() === 'failed'
                ? 'failed'
                : ''
      }))
    : locateBattlefrontRevealStages.map((stage, index) => ({
        key: stage.key,
        label: stage.label,
        status:
          locateBattlefrontStepIndex > index + 1
            ? 'done'
            : locateBattlefrontStepIndex === index + 1
              ? 'active'
              : ''
      }));
  const locateOrbPopupTitle =
    locateOrbPopupPhase === 'processing'
      ? 'Comparing local feature fingerprints'
      : locateOrbPopupPhase === 'error'
        ? 'Fingerprint pass needs attention'
        : 'Fingerprint pass complete';
  const locateOrbPopupDescription =
    locateOrbPopupPhase === 'processing'
      ? locateOrbPopupMessage
      : locateOrbPopupPhase === 'error'
        ? locateStatus || 'The ORB fingerprint pass did not complete successfully.'
        : hasLocateOrbComparisons
          ? 'The strongest overlays are ready. You can review them here or jump back into the ORB tab.'
          : 'ORB finished, but this run did not produce visualization overlays.';
  const locateOrbPopupStageBadge =
    locateOrbPopupPhase === 'processing'
      ? formatLocateProgressStageBadge(
          liveLocatePhase,
          Boolean(liveLocateOrbStats?.enabled ?? locateOrbEnabled)
        )
      : locateOrbPopupPhase === 'error'
        ? 'Needs attention'
        : 'Completed';
  const locateOrbPopupStageSummary =
    locateOrbPopupPhase === 'processing'
      ? liveLocatePhase === 'vector_search'
        ? 'Running the same raw vector retrieval step as Search before any local-feature rerank is applied.'
        : liveLocatePhase === 'orb_rerank'
          ? locateOrbCandidateCount > 0
            ? `Compared ${locateOrbProcessedCandidates} of ${locateOrbCandidateCount} candidates using up to ${locateOrbResolvedFeatureCount} ORB features per image.`
            : `Extracting up to ${locateOrbResolvedFeatureCount} ORB keypoints per image before matching and RANSAC checks.`
          : liveLocatePhase === 'panorama_rerank'
            ? 'The backend finished ORB scoring and is collapsing capture hits into panorama candidates.'
            : liveLocatePhase === 'family_rank'
              ? 'The backend is clustering panorama candidates into nearby location families.'
              : 'Preparing the query image and ORB pipeline.'
      : locateOrbPopupPhase === 'error'
        ? 'The comparison stage stopped before it could finish.'
        : `Scored ${Number(displayedLocateOrbStats?.candidates_scored || 0)} candidates and kept ${locateOrbComparisons.length} visual overlays for review.`;
  const locateOrbPopupCloseLabel =
    locateOrbPopupPhase === 'processing'
      ? 'Hide'
      : locateOrbPopupPhase === 'error'
        ? 'Dismiss'
        : 'Done';

  return (
    <div className="app">
      <div ref={mapContainerRef} id="map" />
      <aside id="sidebar">
        <h1>Street View Coverage Tracker</h1>
        <div className="sidebarNav">
          <button
            className={isScanPage ? '' : 'ghost'}
            onClick={() => navigate('/scan')}
          >
            Scan
          </button>
          <button
            className={isSearchPage ? '' : 'ghost'}
            onClick={() => navigate('/search')}
          >
            Search
          </button>
          <button
            className={isLocatePage ? '' : 'ghost'}
            onClick={() => navigate('/locate')}
          >
            Locate
          </button>
        </div>

        <section className="card">
          <h2>Stats</h2>
          <div className="row"><span>Panoramas</span><span>{stats.total_panoramas}</span></div>
          <div className="row"><span>Captures</span><span>{stats.total_captures}</span></div>
          <label>
            Color mode
            <select value={colorMode} onChange={(e) => setColorMode(e.target.value)}>
              <option value="recency">Recency</option>
              <option value="heading">Heading</option>
              <option value="fixed">Fixed</option>
            </select>
          </label>
        </section>

        {isScanPage ? (
          <>
        <section className="card">
          <h2>One-shot capture</h2>
          <label>Lat<input value={oneShotLat} onChange={(e) => setOneShotLat(e.target.value)} /></label>
          <label>Lon<input value={oneShotLon} onChange={(e) => setOneShotLon(e.target.value)} /></label>
          <button onClick={runOneShot}>Capture one point</button>
          <p className="status">{oneShotStatus}</p>
        </section>

        <section className="card">
          <h2>Area scan</h2>
              <p className="hint">Use map controls: Pg (polygon), Pt (point), FD (free draw).</p>
              <div className="grid2">
                <label>
                  Job type
                  <select value={scanForm.jobType} onChange={(e) => onScanField('jobType', e.target.value)}>
                    <option value="scan">scan</option>
                    <option value="enrich">enrich</option>
                    <option value="fill">fill</option>
                  </select>
                </label>
                <label>
                  Profile
                  <select value={scanForm.captureProfile} onChange={(e) => onScanField('captureProfile', e.target.value)}>
                    <option value="base">base</option>
                    <option value="high_v1">high_v1</option>
                  </select>
                </label>
              </div>
          <div className="grid2">
            <label>Min lat<input value={scanForm.minLat} onChange={(e) => onScanField('minLat', e.target.value)} /></label>
            <label>Min lon<input value={scanForm.minLon} onChange={(e) => onScanField('minLon', e.target.value)} /></label>
            <label>Max lat<input value={scanForm.maxLat} onChange={(e) => onScanField('maxLat', e.target.value)} /></label>
            <label>Max lon<input value={scanForm.maxLon} onChange={(e) => onScanField('maxLon', e.target.value)} /></label>
          </div>
          <div className="grid2">
            <label>
              Workers
              <input
                value={scanForm.workers}
                min="1"
                max={String(currentWorkerLimit)}
                onChange={(e) => onScanField('workers', e.target.value)}
              />
              <span className="fieldHint">Limit: {currentWorkerLimit} for {scanForm.mode} mode.</span>
            </label>
            <label>Step m<input value={scanForm.stepMeters} onChange={(e) => onScanField('stepMeters', e.target.value)} /></label>
            <label>Dedup m<input value={scanForm.dedupRadius} onChange={(e) => onScanField('dedupRadius', e.target.value)} /></label>
                <label>Fill gap m<input value={scanForm.fillGapMeters} onChange={(e) => onScanField('fillGapMeters', e.target.value)} /></label>
            <label>
              Mode
              <select value={scanForm.mode} onChange={(e) => onScanField('mode', e.target.value)}>
                <option value="modal">modal</option>
                <option value="local">local</option>
              </select>
            </label>
          </div>
              {scanForm.jobType === 'enrich' ? (
                <label className="checkboxLabel">
                  <input
                    type="checkbox"
                    checked={Boolean(scanForm.enrichMissingOnly)}
                    onChange={(e) => onScanField('enrichMissingOnly', e.target.checked)}
                  />
                  Only enrich missing views
                </label>
              ) : null}
          <div className="buttonRow">
            <button onClick={startScan}>Start scan</button>
                <button className="ghost" onClick={() => stopScan()}>Stop selected</button>
          </div>
          <p className="status">{scanStatusText}</p>
          <div className="progressGrid">
            <div>Pending: {scanProgress.pending}</div>
            <div>In progress: {scanProgress.inProgress}</div>
            <div>Done: {scanProgress.done}</div>
            <div>Skipped: {scanProgress.skipped}</div>
            <div>Failed: {scanProgress.failed}</div>
            <div>Workers alive: {scanProgress.workers}</div>
              </div>
              <div className="jobList">
                {(scanJobs || []).map((job) => (
                  <div key={job.scan_id} className={`jobItem ${activeScanId === job.scan_id ? 'active' : ''}`}>
                    <div className="jobMeta">
                      <div className="jobId">{job.scan_id}</div>
                      <div className="jobHint">
                        {String(job.job_type || 'scan')} | {String(job.mode || 'unknown')} | {String(job.status || 'running')}
                      </div>
                    </div>
                    <div className="jobActions">
                      <button
                        className="ghost"
                        onClick={() => addTrackedScanId(job.scan_id, true)}
                      >
                        Track
                      </button>
                      <button
                        className="ghost"
                        onClick={() => stopScan(job.scan_id)}
                        disabled={isScanTerminalStatus(job.status)}
                      >
                        Stop
                      </button>
                    </div>
                  </div>
                ))}
                {scanJobs.length === 0 ? <p className="hint">No active or recent scan jobs yet.</p> : null}
              </div>
            </section>
          </>
        ) : null}

        {isSearchPage ? (
          <>
            <section className="card">
          <h2>Image search</h2>
          <p className="hint">
            Indexed {retrievalStats.embedded_captures}/{retrievalStats.total_captures} captures
            {retrievalStats.model_name ? ` (${retrievalStats.model_name}${retrievalStats.model_version ? ` ${retrievalStats.model_version}` : ''})` : ''}
          </p>
          {Array.isArray(retrievalStats.models) && retrievalStats.models.length > 1 ? (
            <p className="hint">
              Models: {retrievalStats.models.map((m) => `${m.model_name} ${m.model_version} ${m.embedded_captures}/${m.total_captures}`).join(' | ')}
            </p>
          ) : null}
          <label title={retrievalControlHelp.reference_image}>
            Reference image
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setRetrievalFile(e.target.files?.[0] || null)}
            />
          </label>
          <div className="grid2">
            <label title={retrievalControlHelp.embedding_base}>
              Search model
              <select
                value={searchEmbeddingBase}
                onChange={(e) => setSearchEmbeddingBase(e.target.value)}
                disabled={retrievalBusy}
              >
                {retrievalEmbeddingBaseOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label title={retrievalControlHelp.search_top_k}>
              Search Top K
              <input
                type="number"
                min="1"
                max="200"
                value={retrievalSearchTopK}
                onChange={(e) => setRetrievalSearchTopK(e.target.value)}
              />
            </label>
            <label title={retrievalControlHelp.min_similarity}>
              Min similarity
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={retrievalMinSimilarity}
                onChange={(e) => setRetrievalMinSimilarity(e.target.value)}
                placeholder="optional"
              />
            </label>
          </div>
          <div className="buttonRow">
            <button onClick={runImageSearch} disabled={retrievalBusy}>
              {retrievalBusy ? 'Searching...' : 'Search by image'}
            </button>
            <button
              className="ghost"
              onClick={() => {
                setRetrievalResults([]);
                setRetrievalStatus('');
                setTopSearchCaptureId(null);
                clearSearchFallbackMarker();
              }}
              disabled={retrievalBusy}
            >
              Clear
            </button>
          </div>
          <p className="hint">Best match is auto-focused on the map and highlighted below.</p>
          <p className="status">{retrievalStatus}</p>
          <div className="retrievalGrid">
            {retrievalResults.map((result) => (
              <button
                key={result.capture_id}
                className={`retrievalItem ${
                  Number(result.capture_id) === Number(topSearchCaptureId) ? 'retrievalItemBest' : ''
                }`}
                onClick={() => focusRetrievalResult(result)}
                title={`score ${(Number(result.similarity || 0)).toFixed(3)}`}
              >
                {String(result.web_path || '').trim() ? (
                  <img src={result.web_path} alt={`match ${result.capture_id}`} />
                ) : (
                  <div className="retrievalItemPlaceholder">
                    No local image
                    <span>
                      {Number.isFinite(Number(result.lat)) && Number.isFinite(Number(result.lon))
                        ? `${Number(result.lat).toFixed(6)}, ${Number(result.lon).toFixed(6)}`
                        : 'Coordinates unavailable'}
                    </span>
                  </div>
                )}
                <span>score {(Number(result.similarity || 0)).toFixed(3)}</span>
                {Number(result.capture_id) === Number(topSearchCaptureId) ? (
                  <span className="bestMatchTag">Best match</span>
                ) : null}
              </button>
            ))}
          </div>
        </section>

        <section id="preview" className="card">
          <h2>Preview</h2>
          <div className="coords">{preview.title}</div>
          <div className="previewGrid">
            {preview.captures.length ? (
              preview.captures.map((c) => (
                String(c.src || '').trim() ? (
                  <img key={c.id} src={c.src} alt={`heading ${c.heading}`} title={`${c.heading} deg`} />
                ) : (
                  <div key={c.id} className="retrievalItemPlaceholder">
                    No local image
                    <span>heading {Number(c.heading || 0)}</span>
                  </div>
                )
              ))
            ) : (
              <p className="hint">Click a marker to preview captures.</p>
            )}
          </div>
        </section>
          </>
        ) : null}

        {isLocatePage ? (
          <>
            <section className="card">
              <h2>Locate image</h2>
              <p className="hint">
                Uses the selected locate model, optionally reranks top vector hits with ORB, then clusters panorama families.
              </p>
              <label title={retrievalControlHelp.reference_image}>
                Reference image
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setRetrievalFile(e.target.files?.[0] || null)}
                />
              </label>
              <div className="grid2">
                <label title={retrievalControlHelp.embedding_base}>
                  Locate model
                  <select
                    value={locateEmbeddingBase}
                    onChange={(e) => setLocateEmbeddingBase(e.target.value)}
                    disabled={locateBusy}
                  >
                    {retrievalEmbeddingBaseOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label title={retrievalControlHelp.locate_top_k}>
                  Family Top K
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={locateTopK}
                    onChange={(e) => setLocateTopK(e.target.value)}
                  />
                </label>
                <label title={retrievalControlHelp.min_similarity}>
                  Min similarity
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={retrievalMinSimilarity}
                    onChange={(e) => setRetrievalMinSimilarity(e.target.value)}
                    placeholder="optional"
                  />
                </label>
              </div>
              <div className="grid2">
                <label className="checkboxLabel locateToggleCard" title={retrievalControlHelp.locate_battlefront_mode}>
                  <input
                    type="checkbox"
                    checked={locateBattlefrontModeEnabled}
                    onChange={(e) => setLocateBattlefrontModeEnabled(e.target.checked)}
                  />
                  Battlefront reveal mode
                </label>
                <label title={retrievalControlHelp.locate_orb_enabled}>
                  ORB rerank
                  <input
                    type="checkbox"
                    checked={locateOrbEnabled}
                    onChange={(e) => setLocateOrbEnabled(e.target.checked)}
                  />
                </label>
                <label title={retrievalControlHelp.locate_orb_top_n}>
                  ORB Top N
                  <input
                    type="number"
                    min="1"
                    max="5000"
                    value={locateOrbTopN}
                    onChange={(e) => setLocateOrbTopN(e.target.value)}
                  />
                </label>
              </div>
              <div className="grid2">
                <label title={retrievalControlHelp.locate_orb_weight}>
                  ORB Weight
                  <input
                    type="number"
                    min="0"
                    max="5"
                    step="0.05"
                    value={locateOrbWeight}
                    onChange={(e) => setLocateOrbWeight(e.target.value)}
                  />
                </label>
              </div>
              <details className="locateTunePanel">
                <summary>Advanced ORB options</summary>
                <div className="grid2">
                  <label title={retrievalControlHelp.locate_orb_feature_count}>
                    ORB feature count
                    <input
                      type="number"
                      min="100"
                      max="2000"
                      step="50"
                      value={locateOrbFeatureCount}
                      onChange={(e) => setLocateOrbFeatureCount(e.target.value)}
                    />
                  </label>
                  <label title={retrievalControlHelp.locate_orb_ransac_top_k}>
                    RANSAC Top K
                    <input
                      type="number"
                      min="0"
                      max={String(Math.max(1, Number(locateOrbTopN) || 100))}
                      value={locateOrbRansacTopK}
                      onChange={(e) => setLocateOrbRansacTopK(e.target.value)}
                    />
                  </label>
                  <label title={retrievalControlHelp.locate_orb_ignore_bottom_ratio}>
                    Ignore lower frame ratio
                    <input
                      type="number"
                      min="0"
                      max="0.6"
                      step="0.02"
                      value={locateOrbIgnoreBottomRatio}
                      onChange={(e) => setLocateOrbIgnoreBottomRatio(e.target.value)}
                    />
                  </label>
                </div>
                <label className="checkboxLabel locateToggleCard" title={retrievalControlHelp.locate_sam2_mask_cars}>
                  <input
                    type="checkbox"
                    checked={locateSam2MaskCars}
                    onChange={(e) => setLocateSam2MaskCars(e.target.checked)}
                  />
                  Mask cars with local SAM 2
                </label>
                <label className="checkboxLabel locateToggleCard" title={retrievalControlHelp.locate_sam2_mask_trees}>
                  <input
                    type="checkbox"
                    checked={locateSam2MaskTrees}
                    onChange={(e) => setLocateSam2MaskTrees(e.target.checked)}
                  />
                  Mask trees and vegetation
                </label>
                <div className="hint">
                  These settings apply when ORB rerank is enabled. SAM 2 masking affects the query image plus the top ORB review candidates only, so the initial vector search still uses the original image.
                </div>
              </details>
              <div className="grid2">
                <label className="checkboxLabel locateToggleCard" title={retrievalControlHelp.locate_orb_popup}>
                  <input
                    type="checkbox"
                    checked={locateOrbPopupEnabled}
                    onChange={(e) => setLocateOrbPopupEnabled(e.target.checked)}
                  />
                  Show ORB popup
                </label>
                {hasLocateOrbComparisons ? (
                  <button
                    className="ghost compact locateInlineAction"
                    onClick={() => {
                      locateOrbPopupSuppressedRef.current = false;
                      setLocateOrbPopupPhase('results');
                      setLocateOrbPopupOpen(true);
                    }}
                  >
                    Open ORB popup
                  </button>
                ) : (
                  <div className="hint locateInlineHint">
                    Enable ORB to unlock the fingerprint popup and comparison gallery.
                  </div>
                )}
              </div>
              <div className="buttonRow">
                <button onClick={runImageLocate} disabled={locateBusy}>
                  {locateBusy ? 'Locating...' : 'Locate by image'}
                </button>
                <button
                  className="ghost"
                  onClick={() => {
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
                  }}
                  disabled={locateBusy}
                >
                  Clear
                </button>
              </div>
              <div className="locatorPipelineCard">
                <h3>Locator pipeline</h3>
                <div className="pipelineStageList">
                  {(locatePipeline.stages || []).map((stage) => (
                    <div
                      key={stage.key}
                      className={`pipelineStage pipelineStage-${String(stage.status || 'pending').toLowerCase()}`}
                    >
                      <span className="pipelineDot" />
                      <div className="pipelineStageBody">
                        <div>
                          <div className="pipelineStageTitle">{stage.title}</div>
                          {stage.detail ? <div className="hint">{stage.detail}</div> : null}
                        </div>
                        <div className="pipelineStageStatus">{stage.status || 'pending'}</div>
                      </div>
                    </div>
                  ))}
                  {(!locatePipeline.stages || locatePipeline.stages.length === 0) ? (
                    <p className="hint">No locate request yet.</p>
                  ) : null}
                </div>
              </div>
              <p className="status">{locateStatus}</p>
              <div className="locateSubtabBar">
                <button
                  className={locateViewTab === 'families' ? '' : 'ghost'}
                  onClick={() => setLocateViewTab('families')}
                >
                  Families
                </button>
                <button
                  className={locateViewTab === 'orb' ? '' : 'ghost'}
                  onClick={() => setLocateViewTab('orb')}
                  disabled={!locateOrbEnabled && !hasLocateOrbComparisons}
                >
                  ORB fingerprints{hasLocateOrbComparisons ? ` (${locateOrbComparisons.length})` : ''}
                </button>
              </div>
              {locateViewTab === 'families' ? (
                <div className="retrievalGrid">
                  {locateResults.map((result) => (
                    <button
                      key={result.family_id || result.panorama_id}
                      className={`retrievalItem ${
                        String(result.family_id || '') === String(topLocateFamilyId || '')
                          ? 'retrievalItemBest'
                          : ''
                      }`}
                      onClick={() => focusRetrievalResult(result)}
                      title={`family score ${(Number(result.family_score || 0)).toFixed(3)}`}
                    >
                      {String(result.web_path || '').trim() ? (
                        <img src={result.web_path} alt={`family ${result.family_id || result.panorama_id}`} />
                      ) : (
                        <div className="retrievalItemPlaceholder">
                          No local image
                          <span>
                            {Number.isFinite(Number(result.lat ?? result.family_center_lat)) &&
                            Number.isFinite(Number(result.lon ?? result.family_center_lon))
                              ? `${Number(result.lat ?? result.family_center_lat).toFixed(6)}, ${Number(result.lon ?? result.family_center_lon).toFixed(6)}`
                              : 'Coordinates unavailable'}
                          </span>
                        </div>
                      )}
                      <span>family score {(Number(result.family_score || 0)).toFixed(3)}</span>
                      <span>
                        {Number(result.family_panorama_count || 0)} panoramas | {Number(result.family_capture_hits || 0)} capture hits
                      </span>
                      {String(result.family_id || '') === String(topLocateFamilyId || '') ? (
                        <span className="bestMatchTag">Top family</span>
                      ) : null}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="orbLabPanel">
                  <div className="orbLabHeader">
                    <div>
                      <div className="orbLabEyebrow">ORB fingerprint comparisons</div>
                      <div className="hint">
                        Match overlays for the top reranked capture images from this locate run. Click a card to focus that panorama on the map.
                      </div>
                      {locateOrbMaskPercent > 0 ? (
                        <div className="hint orbMaskHint">
                          Lower {locateOrbMaskPercent}% of each frame is ignored during ORB extraction to reduce cars and road clutter.
                        </div>
                      ) : null}
                    </div>
                    {displayedLocateOrbStats ? (
                      <div className="orbChipRow">
                        <span className="orbStatChip">scored {Number(displayedLocateOrbStats.candidates_scored || 0)}</span>
                        <span className="orbStatChip">RANSAC {Number(displayedLocateOrbStats.ransac_checked || 0)}</span>
                        <span className="orbStatChip">query kp {Number(displayedLocateOrbStats.query_keypoints || 0)}</span>
                        <span className="orbStatChip">features {locateOrbResolvedFeatureCount}</span>
                        {locateOrbMaskPercent > 0 ? (
                          <span className="orbStatChip">lower mask {locateOrbMaskPercent}%</span>
                        ) : null}
                        {locateSam2CarsEnabled ? (
                          <span className="orbStatChip">
                            SAM 2 cars {locateSam2VehicleBoxes > 0 ? locateSam2VehicleBoxes : 0}
                          </span>
                        ) : null}
                        {locateSam2TreesEnabled ? (
                          <span className="orbStatChip">
                            trees {locateSam2TreeBoxes > 0 ? locateSam2TreeBoxes : 0}
                          </span>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                  {String(displayedLocateOrbStats?.query_fingerprint_data_url || '').trim() ? (
                    <div className="orbQueryCard">
                      <div>
                        <div className="orbLabEyebrow">Query fingerprint</div>
                        <div className="hint">
                          ORB keypoints extracted from the uploaded reference image.
                        </div>
                      </div>
                      <img
                        src={displayedLocateOrbStats.query_fingerprint_data_url}
                        alt="ORB query fingerprint"
                      />
                    </div>
                  ) : null}
                  {hasLocateOrbComparisons ? (
                    <div className="orbComparisonGrid">
                      {locateOrbComparisons.map((comparison) => (
                        <button
                          key={comparison.capture_id}
                          className="orbComparisonCard"
                          onClick={() => focusRetrievalResult(comparison)}
                          title={`ORB ${formatMetric(comparison.orb_score)}`}
                        >
                          {String(comparison.visualization_data_url || '').trim() ? (
                            <img
                              src={comparison.visualization_data_url}
                              alt={`ORB comparison ${comparison.capture_id}`}
                            />
                          ) : (
                            <div className="retrievalItemPlaceholder">
                              No comparison visualization
                              <span>capture {comparison.capture_id}</span>
                            </div>
                          )}
                          <div className="orbChipRow">
                            <span className="orbStatChip">before #{comparison.rank_before}</span>
                            <span className="orbStatChip">after #{comparison.rank_after}</span>
                            <span className="orbStatChip">ORB {formatMetric(comparison.orb_score)}</span>
                          </div>
                          <span>
                            vector {formatMetric(comparison.score_before)} → {formatMetric(comparison.score_after)}
                          </span>
                          <span>
                            {Number(comparison.orb_good_matches || 0)} good matches | {Number(comparison.orb_inliers || 0)} inliers
                          </span>
                          <span>
                            visualized {Number(comparison.visual_match_count || 0)} match lines
                          </span>
                          <span>
                            mean match distance {formatMetric(comparison.orb_mean_match_distance || 0, 1)}
                          </span>
                          <span>
                            query {Number(comparison.query_keypoints || 0)} kp | candidate {Number(comparison.candidate_keypoints || 0)} kp
                          </span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="orbEmptyState">
                      <div className="orbLabEyebrow">No fingerprint comparisons yet</div>
                      <div className="hint">
                        Turn on ORB rerank and run Locate to open the comparison lab.
                        {displayedLocateOrbStats?.reason ? ` Current state: ${displayedLocateOrbStats.reason}.` : ''}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </section>

            <section id="preview" className="card">
              <h2>Preview</h2>
              <div className="coords">{preview.title}</div>
              <div className="previewGrid">
                {preview.captures.length ? (
                  preview.captures.map((c) => (
                    String(c.src || '').trim() ? (
                      <img key={c.id} src={c.src} alt={`heading ${c.heading}`} title={`${c.heading} deg`} />
                    ) : (
                      <div key={c.id} className="retrievalItemPlaceholder">
                        No local image
                        <span>heading {Number(c.heading || 0)}</span>
                      </div>
                    )
                  ))
                ) : (
                  <p className="hint">Run locate and click a family to preview captures.</p>
                )}
              </div>
            </section>
          </>
        ) : null}
      </aside>
      {isLocatePage && locateOrbPopupOpen ? (
        <div
          className="orbPopupBackdrop"
          onClick={() => {
            if (locateBusy && locateOrbPopupPhase === 'processing') return;
            locateOrbPopupSuppressedRef.current = true;
            setLocateOrbPopupOpen(false);
          }}
        >
          <div
            className={`orbPopup orbPopup-${locateOrbPopupPhase}`}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="orbPopupHeader">
              <div>
                <div className="orbLabEyebrow">ORB Fingerprint Lab</div>
                <h2>{locateOrbPopupTitle}</h2>
                <p className="hint">{locateOrbPopupDescription}</p>
              </div>
              <button
                className="ghost"
                onClick={() => {
                  locateOrbPopupSuppressedRef.current = true;
                  setLocateOrbPopupOpen(false);
                }}
              >
                {locateOrbPopupCloseLabel}
              </button>
            </div>
            <div className="orbPopupHero">
              <div className="orbPopupQuery">
                <div className="orbLabEyebrow">Reference image</div>
                {retrievalFilePreviewUrl ? (
                  locateOrbPopupPhase === 'processing' ? (
                    <div className={`orbPopupLivePreview orbPopupLivePreview-${locateOrbPreviewStage}`}>
                      <img
                        className="orbPopupLivePreviewImage"
                        src={retrievalFilePreviewUrl}
                        alt="Reference query"
                      />
                      <div className="orbPopupLivePreviewShade" />
                      <div className="orbPopupLivePreviewGrid" />
                      <div className="orbPopupLivePreviewSweep" />
                      <div className="orbPopupLivePreviewPulse" />
                      <div className="orbPopupLivePreviewNodes" aria-hidden="true">
                        {orbPopupPreviewPoints.map((point, index) => (
                          <span
                            key={`orb-preview-point-${index}`}
                            style={{
                              '--x': `${point.x}%`,
                              '--y': `${point.y}%`,
                              '--delay': `${point.delay}s`
                            }}
                          />
                        ))}
                      </div>
                      {locateSam2Enabled ? (
                        <>
                          {locateSam2CarsEnabled ? (
                            <>
                              <div className="orbPopupLivePreviewVehicle orbPopupLivePreviewVehicle-a" />
                              <div className="orbPopupLivePreviewVehicle orbPopupLivePreviewVehicle-b" />
                            </>
                          ) : null}
                          <div className="orbPopupLivePreviewBadge">
                            {locateSam2CarsEnabled || locateSam2TreesEnabled
                              ? `SAM 2 ${[
                                  locateSam2CarsEnabled ? `cars ${locateSam2VehicleBoxes}` : '',
                                  locateSam2TreesEnabled ? `trees ${locateSam2TreeBoxes}` : ''
                                ]
                                  .filter(Boolean)
                                  .join(' | ')}`
                              : 'SAM 2 masking queued'}
                          </div>
                        </>
                      ) : null}
                      {locateOrbMaskPercent > 0 ? (
                        <div
                          className="orbPopupLivePreviewMask"
                          style={{ height: `${locateOrbMaskPercent}%` }}
                        >
                          <span>ignore lower {locateOrbMaskPercent}%</span>
                        </div>
                      ) : null}
                      <div className="orbPopupLivePreviewCaption">
                        {locateOrbPopupMessage}
                      </div>
                    </div>
                  ) : (
                    <img src={retrievalFilePreviewUrl} alt="Reference query" />
                  )
                ) : (
                  <div className="retrievalItemPlaceholder">
                    Add a query image to preview it here.
                  </div>
                )}
                {String(displayedLocateOrbStats?.query_fingerprint_data_url || '').trim() ? (
                  <img
                    className="orbPopupFingerprint"
                    src={displayedLocateOrbStats.query_fingerprint_data_url}
                    alt="Query ORB fingerprint"
                  />
                ) : null}
              </div>
              <div className="orbPopupStageCard">
                <div className="orbScannerHalo" />
                <div className="orbScannerGrid" />
                <div className={`orbPopupStageBadge orbPopupStageBadge-${locateOrbPopupPhase}`}>
                  {locateOrbPopupStageBadge}
                </div>
                <div className="orbChipRow">
                  <span className="orbStatChip">
                    model {locateModelLabel}
                  </span>
                  <span className="orbStatChip">
                    ORB top {Math.max(1, Number(displayedLocateOrbStats?.top_n || locateOrbTopN) || 100)}
                  </span>
                  <span className="orbStatChip">
                    weight {formatMetric(displayedLocateOrbStats?.weight || locateOrbWeight, 2)}
                  </span>
                  <span className="orbStatChip">
                    features {locateOrbResolvedFeatureCount}
                  </span>
                  {locateOrbMaskPercent > 0 ? (
                    <span className="orbStatChip">
                      lower mask {locateOrbMaskPercent}%
                    </span>
                  ) : null}
                  {locateOrbPopupPhase === 'processing' && locateOrbCandidateCount > 0 ? (
                    <span className="orbStatChip">
                      done {locateOrbProcessedCandidates}/{locateOrbCandidateCount}
                    </span>
                  ) : null}
                  {locateSam2CarsEnabled ? (
                    <span className="orbStatChip">
                      SAM 2 cars {locateSam2VehicleBoxes > 0 ? locateSam2VehicleBoxes : 0}
                    </span>
                  ) : null}
                  {locateSam2TreesEnabled ? (
                    <span className="orbStatChip">
                      trees {locateSam2TreeBoxes > 0 ? locateSam2TreeBoxes : 0}
                    </span>
                  ) : null}
                </div>
                <div className="orbPopupMoment">
                  {locateOrbPopupPhase === 'processing'
                    ? locateOrbPopupMessage
                    : locateOrbPopupTitle}
                </div>
                <div className="orbPopupStageSummary">{locateOrbPopupStageSummary}</div>
                {locateOrbPopupPhase === 'processing' &&
                liveLocateComparison &&
                String(liveLocateComparison.web_path || '').trim() ? (
                  <button
                    className="orbPopupCurrentCandidate"
                    onClick={() => focusRetrievalResult(liveLocateComparison)}
                  >
                    <div className="orbLabEyebrow">Current candidate</div>
                    <img
                      src={liveLocateComparison.web_path}
                      alt={`Current candidate ${liveLocateComparison.capture_id}`}
                    />
                    <div className="orbChipRow">
                      <span className="orbStatChip">capture {Number(liveLocateComparison.capture_id || 0)}</span>
                      <span className="orbStatChip">before #{Number(liveLocateComparison.rank_before || 0)}</span>
                    </div>
                    <span>
                      vector {formatMetric(liveLocateComparison.score_before)} | similarity {formatMetric(liveLocateComparison.similarity)}
                    </span>
                  </button>
                ) : null}
                <div className="orbPopupTimeline">
                  {orbPopupMoments.map((moment, index) => (
                    <div
                      key={moment}
                      className={`orbPopupTimelineItem ${getOrbPopupTimelineState(
                        index,
                        locateOrbPopupPhase,
                        locateOrbPopupMoment
                      )}`}
                    >
                      <span className="pipelineDot" />
                      <span>{moment}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            {locateOrbPopupPhase === 'processing' && liveLocateComparison ? (
              <div className="orbPopupResults orbPopupLiveResults">
                <div className="orbPopupResultsHeader">
                  <div>
                    <div className="orbLabEyebrow">Live backend comparison</div>
                    <div className="hint">
                      Candidate {Math.max(1, locateOrbProcessedCandidates)} of {Math.max(1, locateOrbCandidateCount || Number(displayedLocateOrbStats?.top_n || locateOrbTopN) || 1)}
                    </div>
                  </div>
                  <div className="orbChipRow">
                    <span className="orbStatChip">before #{Number(liveLocateComparison.rank_before || 0)}</span>
                    <span className="orbStatChip">ORB {formatMetric(liveLocateComparison.orb_score)}</span>
                    <span className="orbStatChip">{Number(liveLocateComparison.orb_good_matches || 0)} matches</span>
                  </div>
                </div>
                <div className="orbComparisonGrid orbComparisonGridPopup orbPopupLiveComparisonGrid">
                  <button
                    key={`live-${liveLocateComparison.capture_id}`}
                    className="orbComparisonCard"
                    onClick={() => focusRetrievalResult(liveLocateComparison)}
                  >
                    {String(liveLocateComparison.visualization_data_url || '').trim() ? (
                      <img
                        src={liveLocateComparison.visualization_data_url}
                        alt={`Live ORB comparison ${liveLocateComparison.capture_id}`}
                      />
                    ) : (
                      <div className="retrievalItemPlaceholder">
                        Live comparison still rendering
                      </div>
                    )}
                    <div className="orbChipRow">
                      <span className="orbStatChip">Match overlay</span>
                      <span className="orbStatChip">after #{Number(liveLocateComparison.rank_after || 0)}</span>
                    </div>
                    <span>
                      {Number(liveLocateComparison.orb_good_matches || 0)} good matches | {Number(liveLocateComparison.orb_inliers || 0)} inliers
                    </span>
                    <span>
                      visualized {Number(liveLocateComparison.visual_match_count || 0)} match lines
                    </span>
                    <span>
                      mean match distance {formatMetric(liveLocateComparison.orb_mean_match_distance || 0, 1)}
                    </span>
                  </button>
                </div>
              </div>
            ) : null}
            {locateOrbPopupPhase === 'results' ? (
              hasLocateOrbComparisons ? (
                <div className="orbPopupResults">
                  <div className="orbPopupResultsHeader">
                    <div className="orbChipRow">
                      <span className="orbStatChip">scored {Number(displayedLocateOrbStats?.candidates_scored || 0)}</span>
                      <span className="orbStatChip">RANSAC {Number(displayedLocateOrbStats?.ransac_checked || 0)}</span>
                      <span className="orbStatChip">query kp {Number(displayedLocateOrbStats?.query_keypoints || 0)}</span>
                      <span className="orbStatChip">features {locateOrbResolvedFeatureCount}</span>
                    </div>
                    <button
                      className="ghost"
                      onClick={() => {
                        locateOrbPopupSuppressedRef.current = true;
                        setLocateViewTab('orb');
                        setLocateOrbPopupOpen(false);
                      }}
                    >
                      Open ORB tab
                    </button>
                  </div>
                  <div className="orbComparisonGrid orbComparisonGridPopup">
                    {locateOrbComparisons.slice(0, 4).map((comparison) => (
                      <button
                        key={`popup-${comparison.capture_id}`}
                        className="orbComparisonCard"
                        onClick={() => focusRetrievalResult(comparison)}
                      >
                        {String(comparison.visualization_data_url || '').trim() ? (
                          <img
                            src={comparison.visualization_data_url}
                            alt={`ORB popup comparison ${comparison.capture_id}`}
                          />
                        ) : (
                          <div className="retrievalItemPlaceholder">
                            No comparison visualization
                          </div>
                        )}
                        <div className="orbChipRow">
                          <span className="orbStatChip">after #{comparison.rank_after}</span>
                          <span className="orbStatChip">ORB {formatMetric(comparison.orb_score)}</span>
                        </div>
                        <span>
                          {Number(comparison.orb_good_matches || 0)} good matches | {Number(comparison.visual_match_count || 0)} lines shown
                        </span>
                        <span>
                          mean match distance {formatMetric(comparison.orb_mean_match_distance || 0, 1)}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="orbEmptyState">
                  <div className="orbLabEyebrow">No ORB visuals this time</div>
                  <div className="hint">
                    {displayedLocateOrbStats?.reason
                      ? `The ORB stage reported: ${displayedLocateOrbStats.reason}.`
                      : 'This locate run did not produce comparison overlays.'}
                  </div>
                </div>
              )
            ) : null}
          </div>
        </div>
      ) : null}
      {isLocatePage && locateBattlefrontSequence ? (
        <div
          className={`battlefrontReveal battlefrontReveal-${locateBattlefrontSequence.phase}${
            locateBattlefrontSequence.isZooming ? ' battlefrontReveal-zooming' : ''
          }`}
        >
          <div className="battlefrontRevealShade" />
          <div className="battlefrontRevealStars" />
          <div
            key={`battlefront-grid-${locateBattlefrontSequence.phase}-${Number(locateBattlefrontSequence.stepIndex || 0)}`}
            className="battlefrontRevealGridField"
            style={{
              '--battlefront-line-duration': `${Math.max(0, Number(locateBattlefrontSequence.lineDurationMs || 0))}ms`,
              '--battlefront-line-horizontal-duration': `${Math.max(
                0,
                Math.round(Number(locateBattlefrontSequence.lineDurationMs || 0) * 0.56)
              )}ms`,
              '--battlefront-line-vertical-delay': `${Math.max(
                0,
                Math.round(Number(locateBattlefrontSequence.lineDurationMs || 0) * 0.56)
              )}ms`,
              '--battlefront-line-vertical-duration': `${Math.max(
                0,
                Number(locateBattlefrontSequence.lineDurationMs || 0) -
                  Math.round(Number(locateBattlefrontSequence.lineDurationMs || 0) * 0.56)
              )}ms`,
              '--battlefront-zoom-duration': `${Math.max(
                0,
                Number(locateBattlefrontSequence.zoomDurationMs || 0)
              )}ms`
            }}
          >
            <div className="battlefrontRevealGridGlow" />
            <div className="battlefrontRevealGridPlane battlefrontRevealGridPlane-major" />
            <div className="battlefrontRevealGridPlane battlefrontRevealGridPlane-minor" />
            <div className="battlefrontRevealGridWindow" />
            <div className="battlefrontRevealGridEdge battlefrontRevealGridEdge-top" />
            <div className="battlefrontRevealGridEdge battlefrontRevealGridEdge-bottom" />
            <div className="battlefrontRevealGridEdge battlefrontRevealGridEdge-left" />
            <div className="battlefrontRevealGridEdge battlefrontRevealGridEdge-right" />
          </div>
          <div className="battlefrontRevealReticle" />
          <div className="battlefrontRevealOrbit battlefrontRevealOrbit-a" />
          <div className="battlefrontRevealOrbit battlefrontRevealOrbit-b" />
          <div className="battlefrontRevealHud">
            <div className="battlefrontRevealEyebrow">Battlefront reveal mode</div>
            <div className="battlefrontRevealTitle">{locateBattlefrontHudTitle}</div>
            <div className="battlefrontRevealSubtitle">{locateBattlefrontHudSubtitle}</div>
            <div className="battlefrontRevealProgress">
              {locateBattlefrontProgressStages.map((stage) => (
                <div
                  key={stage.key}
                  className={`battlefrontRevealProgressItem ${stage.status}`}
                >
                  <span className="battlefrontRevealProgressDot" />
                  <span>{stage.label}</span>
                </div>
              ))}
            </div>
            <div className="battlefrontRevealMeta">
              {locateBattlefrontMetaItems.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          </div>
          <div className="battlefrontRevealPreviewCard">
            <div className="battlefrontRevealPreviewLabel">{locateBattlefrontPreviewLabel}</div>
            {locateBattlefrontPreviewPath ? (
              <img
                src={locateBattlefrontPreviewPath}
                alt={
                  locateBattlefrontSearching
                    ? 'Locate reference image'
                    : `Locate target ${locateBattlefrontSequence.targetLabel}`
                }
              />
            ) : (
              <div className="retrievalItemPlaceholder">
                {locateBattlefrontSearching ? 'Waiting for a landing lock' : 'No local image'}
                <span>
                  {locateBattlefrontSearching
                    ? String(liveLocateProgress?.message || 'Locate pipeline is still resolving the match.')
                    : locateBattlefrontSequence.targetLabel}
                </span>
              </div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
