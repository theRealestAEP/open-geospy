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
const LOCATE_TUNING_STORAGE_KEY = 'geospy.locate_tuning';
const defaultLocateTuning = {
  top_k_per_crop: 220,
  max_candidates: 5000,
  panorama_vote_cap: 3,
  cluster_radius_m: 45,
  verify_top_n: 120,
  min_good_matches: 11,
  min_inlier_ratio: 0.16,
  appearance_penalty_weight: 0.22,
  db_max_top_k: 5000,
  ivfflat_probes: 120
};
const defaultLocateTuningBounds = {
  top_k_per_crop: { min: 5, max: 1200, type: 'int' },
  max_candidates: { min: 30, max: 10000, type: 'int' },
  panorama_vote_cap: { min: 1, max: 8, type: 'int' },
  cluster_radius_m: { min: 5, max: 250, type: 'float' },
  verify_top_n: { min: 5, max: 400, type: 'int' },
  min_good_matches: { min: 4, max: 80, type: 'int' },
  min_inlier_ratio: { min: 0.01, max: 0.95, type: 'float' },
  appearance_penalty_weight: { min: 0, max: 0.95, type: 'float' },
  db_max_top_k: { min: 100, max: 20000, type: 'int' },
  ivfflat_probes: { min: 1, max: 1000, type: 'int' }
};

function normalizeLocateBounds(rawBounds) {
  const next = { ...defaultLocateTuningBounds };
  if (!rawBounds || typeof rawBounds !== 'object') {
    return next;
  }
  Object.keys(next).forEach((key) => {
    const fallback = next[key];
    const value = rawBounds[key];
    if (!value || typeof value !== 'object') {
      return;
    }
    const parsedMin = Number(value.min);
    const parsedMax = Number(value.max);
    next[key] = {
      min: Number.isFinite(parsedMin) ? parsedMin : fallback.min,
      max: Number.isFinite(parsedMax) ? parsedMax : fallback.max,
      type: value.type === 'float' ? 'float' : 'int'
    };
  });
  return next;
}

function normalizeLocateTuning(rawTuning, defaults = defaultLocateTuning, bounds = defaultLocateTuningBounds) {
  const source = rawTuning && typeof rawTuning === 'object' ? rawTuning : {};
  const next = {};
  Object.keys(defaults).forEach((key) => {
    const baseline = Number(defaults[key]);
    const bound = bounds[key] || defaultLocateTuningBounds[key];
    const candidate = Number(source[key]);
    const fallback = Number.isFinite(baseline) ? baseline : Number(defaultLocateTuning[key]);
    let value = Number.isFinite(candidate) ? candidate : fallback;
    value = Math.max(Number(bound.min), Math.min(Number(bound.max), value));
    if (bound.type !== 'float') {
      value = Math.round(value);
    }
    next[key] = value;
  });
  return next;
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
  const locatorSupportLayerRef = useRef(null);
  const drawnItemsRef = useRef(null);
  const oneShotMarkerRef = useRef(null);
  const locatorMarkerRef = useRef(null);
  const locatorRadiusRef = useRef(null);
  const pointModeEnabledRef = useRef(false);
  const freeDrawEnabledRef = useRef(false);
  const freeDrawDrawingRef = useRef(false);
  const freeDrawLatLngsRef = useRef([]);
  const freeDrawPreviewRef = useRef(null);
  const polygonModeEnabledRef = useRef(false);
  const polygonDraftLatLngsRef = useRef([]);
  const polygonDraftPreviewRef = useRef(null);
  const polygonDraftVertexLayerRef = useRef(null);
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
  const [locateTuningDefaults, setLocateTuningDefaults] = useState(defaultLocateTuning);
  const [locateTuningBounds, setLocateTuningBounds] = useState(defaultLocateTuningBounds);
  const [locateTuning, setLocateTuning] = useState(defaultLocateTuning);
  const [retrievalMinSimilarity, setRetrievalMinSimilarity] = useState('');
  const [retrievalStatus, setRetrievalStatus] = useState('');
  const [retrievalBusy, setRetrievalBusy] = useState(false);
  const [retrievalResults, setRetrievalResults] = useState([]);
  const [retrievalIncludeDebug, setRetrievalIncludeDebug] = useState(false);
  const [locatorEstimate, setLocatorEstimate] = useState(null);
  const [retrievalDebug, setRetrievalDebug] = useState(null);
  const [retrievalStats, setRetrievalStats] = useState({
    model_name: '',
    model_version: '',
    total_captures: 0,
    embedded_captures: 0,
    pending_captures: 0,
    models: []
  });
  const isScanPage = location.pathname === '/scan' || location.pathname === '/';
  const isLocatePage = location.pathname === '/locate';

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, { preferCanvas: true }).setView([37.785, -122.43], 14);
    mapRef.current = map;
    markerRendererRef.current = L.canvas({ padding: 0.5 });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    markersRef.current = L.layerGroup().addTo(map);
    locatorSupportLayerRef.current = L.layerGroup().addTo(map);
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
    if (location.pathname !== '/scan' && location.pathname !== '/locate') {
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

  const clearLocatorOverlay = () => {
    const map = mapRef.current;
    if (map && locatorMarkerRef.current) {
      map.removeLayer(locatorMarkerRef.current);
      locatorMarkerRef.current = null;
    }
    if (map && locatorRadiusRef.current) {
      map.removeLayer(locatorRadiusRef.current);
      locatorRadiusRef.current = null;
    }
    if (locatorSupportLayerRef.current) {
      locatorSupportLayerRef.current.clearLayers();
    }
  };

  const renderLocatorOverlay = (body) => {
    const map = mapRef.current;
    if (!map) return;
    clearLocatorOverlay();
    const estimate = body?.best_estimate || null;
    const supports = Array.isArray(body?.supporting_matches) ? body.supporting_matches : [];
    if (!estimate) return;
    const lat = Number(estimate.lat);
    const lon = Number(estimate.lon);
    const radiusM = Number(estimate.radius_m || 0);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

    locatorMarkerRef.current = L.circleMarker([lat, lon], {
      radius: 7,
      fillColor: '#00d4ff',
      fillOpacity: 0.95,
      color: '#002b36',
      weight: 2
    }).addTo(map);
    locatorMarkerRef.current.bindTooltip(`Estimate conf ${(Number(estimate.confidence || 0)).toFixed(2)}`);

    locatorRadiusRef.current = L.circle([lat, lon], {
      radius: Math.max(1, radiusM),
      color: '#00d4ff',
      opacity: 0.8,
      fillColor: '#00d4ff',
      fillOpacity: 0.12,
      weight: 1
    }).addTo(map);

    if (locatorSupportLayerRef.current) {
      supports.slice(0, 15).forEach((item) => {
        const sLat = Number(item.lat);
        const sLon = Number(item.lon);
        if (!Number.isFinite(sLat) || !Number.isFinite(sLon)) return;
        const score = Number(item.score || item.similarity || 0);
        const marker = L.circleMarker([sLat, sLon], {
          radius: 4,
          fillColor: '#8be9fd',
          fillOpacity: 0.8,
          color: '#1d3557',
          weight: 1
        });
        marker.bindTooltip(`score ${score.toFixed(3)}`);
        locatorSupportLayerRef.current.addLayer(marker);
      });
    }

    map.setView([lat, lon], Math.max(16, map.getZoom()));
  };

  const loadData = async (options = {}) => {
    const { forceStats = true, refreshViewport = true } = options;
    try {
      const map = mapRef.current;
      if (refreshViewport && map) {
        const bounds = map.getBounds();
        const zoom = Math.max(1, Math.round(map.getZoom()));
        const pointLimit = zoom >= 18 ? 5000 : zoom >= 16 ? 3200 : 1800;
        const params = new URLSearchParams({
          min_lat: String(bounds.getSouth()),
          min_lon: String(bounds.getWest()),
          max_lat: String(bounds.getNorth()),
          max_lon: String(bounds.getEast()),
          zoom: String(zoom),
          limit: String(pointLimit),
          cluster_zoom_threshold: '16'
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

  const renderMarkers = (geojson) => {
    if (!markersRef.current) return;
    markersRef.current.clearLayers();
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
          src: c.web_path || `/${String(c.filepath || '').replace(/\\/g, '/')}`
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
      setRetrievalStats({
        model_name: body.model_name || '',
        model_version: body.model_version || '',
        total_captures: Number(body.total_captures || 0),
        embedded_captures: Number(body.embedded_captures || 0),
        pending_captures: Number(body.pending_captures || 0),
        models: Array.isArray(body.models) ? body.models : []
      });
    } catch (error) {
      console.error('retrieval stats error', error);
    }
  };

  const loadLocateTuningConfig = async () => {
    let nextDefaults = defaultLocateTuning;
    let nextBounds = defaultLocateTuningBounds;
    try {
      const res = await fetch('/api/retrieval/locate-params');
      if (res.ok) {
        const body = await res.json();
        nextBounds = normalizeLocateBounds(body?.bounds || {});
        nextDefaults = normalizeLocateTuning(body?.defaults || {}, defaultLocateTuning, nextBounds);
      }
    } catch (error) {
      console.error('locate tuning config error', error);
    }

    let stored = {};
    try {
      const raw = window.localStorage.getItem(LOCATE_TUNING_STORAGE_KEY);
      stored = raw ? JSON.parse(raw) : {};
    } catch {
      stored = {};
    }

    const merged = normalizeLocateTuning({ ...nextDefaults, ...(stored || {}) }, nextDefaults, nextBounds);
    setLocateTuningDefaults(nextDefaults);
    setLocateTuningBounds(nextBounds);
    setLocateTuning(merged);
    window.localStorage.setItem(LOCATE_TUNING_STORAGE_KEY, JSON.stringify(merged));
  };

  const onLocateTuningField = (field, value) => {
    setLocateTuning((prev) => {
      const next = normalizeLocateTuning(
        {
          ...prev,
          [field]: value
        },
        locateTuningDefaults,
        locateTuningBounds
      );
      window.localStorage.setItem(LOCATE_TUNING_STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };

  const resetLocateTuning = () => {
    const next = normalizeLocateTuning(locateTuningDefaults, locateTuningDefaults, locateTuningBounds);
    setLocateTuning(next);
    window.localStorage.setItem(LOCATE_TUNING_STORAGE_KEY, JSON.stringify(next));
  };

  const runImageSearch = async () => {
    if (!retrievalFile) {
      setRetrievalStatus('Pick a reference image first.');
      return;
    }
    setRetrievalBusy(true);
    setLocatorEstimate(null);
    setRetrievalDebug(null);
    clearLocatorOverlay();
    setRetrievalStatus('Searching similar captures...');
    try {
      const formData = new FormData();
      formData.append('image', retrievalFile);
      formData.append('top_k', String(Math.max(1, Number(retrievalSearchTopK) || 12)));
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
        return;
      }
      const matches = Array.isArray(body.matches) ? body.matches : [];
      setRetrievalResults(matches);
      setRetrievalStatus(`Found ${matches.length} matches.`);
      await loadRetrievalStats();
    } catch (error) {
      setRetrievalStatus(`Search failed: ${error.message}`);
      setRetrievalResults([]);
    } finally {
      setRetrievalBusy(false);
    }
  };

  const runImageLocate = async () => {
    if (!retrievalFile) {
      setRetrievalStatus('Pick a reference image first.');
      return;
    }
    setRetrievalBusy(true);
    setRetrievalStatus('Locating image...');
    setRetrievalDebug(null);
    try {
      const formData = new FormData();
      formData.append('image', retrievalFile);
      Object.entries(locateTuning).forEach(([field, fieldValue]) => {
        formData.append(field, String(fieldValue));
      });
      const minSimilarityRaw = String(retrievalMinSimilarity ?? '').trim();
      if (minSimilarityRaw !== '') {
        const parsedSimilarity = Number(minSimilarityRaw);
        if (!Number.isNaN(parsedSimilarity)) {
          formData.append('min_similarity', String(parsedSimilarity));
        }
      }
      if (retrievalIncludeDebug) {
        formData.append('include_debug', '1');
      }
      const res = await fetch('/api/retrieval/locate-by-image', {
        method: 'POST',
        body: formData
      });
      const body = await res.json();
      if (!res.ok) {
        setRetrievalStatus(`Locate failed: ${body?.detail || 'unknown error'}`);
        setRetrievalResults([]);
        setLocatorEstimate(null);
        clearLocatorOverlay();
        return;
      }

      const supports = Array.isArray(body.supporting_matches) ? body.supporting_matches : [];
      const estimate = body.best_estimate || null;
      setRetrievalResults(supports);
      setLocatorEstimate(estimate);
      setRetrievalDebug(retrievalIncludeDebug ? body.debug || {} : null);

      if (estimate) {
        setRetrievalStatus(
          `Estimate lat=${Number(estimate.lat).toFixed(6)} lon=${Number(estimate.lon).toFixed(6)} conf=${Number(estimate.confidence || 0).toFixed(2)} radius=${Number(estimate.radius_m || 0).toFixed(1)}m`
        );
      } else {
        const flags = Array.isArray(body.flags) ? body.flags.join(', ') : 'no estimate';
        setRetrievalStatus(`No estimate. flags=${flags}`);
      }

      renderLocatorOverlay(body);
      if (supports[0]?.panorama_id != null) {
        await showPreview({ id: supports[0].panorama_id, pano_id: supports[0].pano_id || '' });
      }
      await loadRetrievalStats();
    } catch (error) {
      setRetrievalStatus(`Locate failed: ${error.message}`);
      setRetrievalResults([]);
      setLocatorEstimate(null);
      clearLocatorOverlay();
    } finally {
      setRetrievalBusy(false);
    }
  };

  const focusRetrievalResult = async (result) => {
    const map = mapRef.current;
    const lat = Number(result?.lat);
    const lon = Number(result?.lon);
    if (map && Number.isFinite(lat) && Number.isFinite(lon)) {
      map.setView([lat, lon], Math.max(16, map.getZoom()));
    }
    if (result?.panorama_id != null) {
      await showPreview({ id: result.panorama_id, pano_id: result.pano_id || '' });
    }
  };

  useEffect(() => {
    let isCancelled = false;
    const hydrateScanAndRetrievalState = async () => {
      await loadRetrievalStats();
      await loadLocateTuningConfig();
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
                <label>Workers<input value={scanForm.workers} onChange={(e) => onScanField('workers', e.target.value)} /></label>
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

        {isLocatePage ? (
          <>
            <section className="card">
          <h2>Image retrieval</h2>
          <p className="hint">
            Indexed {retrievalStats.embedded_captures}/{retrievalStats.total_captures} captures
            {retrievalStats.model_name ? ` (${retrievalStats.model_name}${retrievalStats.model_version ? ` ${retrievalStats.model_version}` : ''})` : ''}
          </p>
          {Array.isArray(retrievalStats.models) && retrievalStats.models.length > 1 ? (
            <p className="hint">
              Models: {retrievalStats.models.map((m) => `${m.model_name} ${m.model_version} ${m.embedded_captures}/${m.total_captures}`).join(' | ')}
            </p>
          ) : null}
          <label>
            Reference image
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setRetrievalFile(e.target.files?.[0] || null)}
            />
          </label>
          <div className="grid2">
            <label>
              Search Top K
              <input
                type="number"
                min="1"
                max="200"
                value={retrievalSearchTopK}
                onChange={(e) => setRetrievalSearchTopK(e.target.value)}
              />
            </label>
            <label>
              Locate Top K / crop
              <input
                type="number"
                min={locateTuningBounds.top_k_per_crop.min}
                max={locateTuningBounds.top_k_per_crop.max}
                value={locateTuning.top_k_per_crop}
                onChange={(e) => onLocateTuningField('top_k_per_crop', e.target.value)}
              />
            </label>
            <label>
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
          <details className="locateTunePanel">
            <summary>Advanced locate tuning</summary>
            <div className="grid2">
              <label>
                Max merged candidates
                <input
                  type="number"
                  min={locateTuningBounds.max_candidates.min}
                  max={locateTuningBounds.max_candidates.max}
                  value={locateTuning.max_candidates}
                  onChange={(e) => onLocateTuningField('max_candidates', e.target.value)}
                />
              </label>
              <label>
                Verify top N
                <input
                  type="number"
                  min={locateTuningBounds.verify_top_n.min}
                  max={locateTuningBounds.verify_top_n.max}
                  value={locateTuning.verify_top_n}
                  onChange={(e) => onLocateTuningField('verify_top_n', e.target.value)}
                />
              </label>
              <label>
                Min good matches
                <input
                  type="number"
                  min={locateTuningBounds.min_good_matches.min}
                  max={locateTuningBounds.min_good_matches.max}
                  value={locateTuning.min_good_matches}
                  onChange={(e) => onLocateTuningField('min_good_matches', e.target.value)}
                />
              </label>
              <label>
                Min inlier ratio
                <input
                  type="number"
                  min={locateTuningBounds.min_inlier_ratio.min}
                  max={locateTuningBounds.min_inlier_ratio.max}
                  step="0.01"
                  value={locateTuning.min_inlier_ratio}
                  onChange={(e) => onLocateTuningField('min_inlier_ratio', e.target.value)}
                />
              </label>
              <label>
                Panorama vote cap
                <input
                  type="number"
                  min={locateTuningBounds.panorama_vote_cap.min}
                  max={locateTuningBounds.panorama_vote_cap.max}
                  value={locateTuning.panorama_vote_cap}
                  onChange={(e) => onLocateTuningField('panorama_vote_cap', e.target.value)}
                />
              </label>
              <label>
                Cluster radius (m)
                <input
                  type="number"
                  min={locateTuningBounds.cluster_radius_m.min}
                  max={locateTuningBounds.cluster_radius_m.max}
                  step="1"
                  value={locateTuning.cluster_radius_m}
                  onChange={(e) => onLocateTuningField('cluster_radius_m', e.target.value)}
                />
              </label>
              <label>
                Appearance penalty
                <input
                  type="number"
                  min={locateTuningBounds.appearance_penalty_weight.min}
                  max={locateTuningBounds.appearance_penalty_weight.max}
                  step="0.01"
                  value={locateTuning.appearance_penalty_weight}
                  onChange={(e) => onLocateTuningField('appearance_penalty_weight', e.target.value)}
                />
              </label>
              <label>
                DB max top K
                <input
                  type="number"
                  min={locateTuningBounds.db_max_top_k.min}
                  max={locateTuningBounds.db_max_top_k.max}
                  value={locateTuning.db_max_top_k}
                  onChange={(e) => onLocateTuningField('db_max_top_k', e.target.value)}
                />
              </label>
              <label>
                ivfflat probes
                <input
                  type="number"
                  min={locateTuningBounds.ivfflat_probes.min}
                  max={locateTuningBounds.ivfflat_probes.max}
                  value={locateTuning.ivfflat_probes}
                  onChange={(e) => onLocateTuningField('ivfflat_probes', e.target.value)}
                />
              </label>
            </div>
            <div className="buttonRow">
              <button className="ghost" type="button" onClick={resetLocateTuning} disabled={retrievalBusy}>
                Reset locate tuning
              </button>
            </div>
            <p className="hint">These values are saved locally and sent with each Locate request.</p>
          </details>
          <div className="buttonRow">
            <button onClick={runImageSearch} disabled={retrievalBusy}>
              {retrievalBusy ? 'Searching...' : 'Search by image'}
            </button>
            <button className="ghost" onClick={runImageLocate} disabled={retrievalBusy}>
              {retrievalBusy ? 'Running...' : 'Locate image'}
            </button>
          </div>
          <label className="checkboxLabel">
            <input
              type="checkbox"
              checked={Boolean(retrievalIncludeDebug)}
              onChange={(e) => setRetrievalIncludeDebug(e.target.checked)}
            />
            Include debug diagnostics
          </label>
          <div className="buttonRow">
            <button
              className="ghost"
              onClick={() => {
                setRetrievalResults([]);
                setRetrievalStatus('');
                setLocatorEstimate(null);
                setRetrievalDebug(null);
                clearLocatorOverlay();
              }}
              disabled={retrievalBusy}
            >
              Clear
            </button>
          </div>
          <p className="status">{retrievalStatus}</p>
          {locatorEstimate ? (
            <p className="status">
              Locate estimate: {Number(locatorEstimate.lat).toFixed(6)}, {Number(locatorEstimate.lon).toFixed(6)}
              {' | '}
              conf {Number(locatorEstimate.confidence || 0).toFixed(2)}
              {' | '}
              radius {Number(locatorEstimate.radius_m || 0).toFixed(1)}m
            </p>
          ) : null}
          {retrievalDebug && Object.keys(retrievalDebug).length ? (
            <pre className="retrievalDebug">{JSON.stringify(retrievalDebug, null, 2)}</pre>
          ) : null}
          <div className="retrievalGrid">
            {retrievalResults.map((result) => (
              <button
                key={result.capture_id}
                className="retrievalItem"
                onClick={() => focusRetrievalResult(result)}
                title={`score ${(Number(result.similarity || 0)).toFixed(3)}`}
              >
                <img src={result.web_path} alt={`match ${result.capture_id}`} />
                <span>score {(Number(result.similarity || 0)).toFixed(3)}</span>
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
                    <img key={c.id} src={c.src} alt={`heading ${c.heading}`} title={`${c.heading} deg`} />
                  ))
                ) : (
                  <p className="hint">Click a marker to preview captures.</p>
                )}
              </div>
            </section>
          </>
        ) : null}
      </aside>
    </div>
  );
}

export default App;
