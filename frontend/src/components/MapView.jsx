import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';
import 'leaflet-draw';

import installMapDrawTools from '../lib/mapDrawTools';
import { interpolateColor } from '../lib/format';

/**
 * Owns the Leaflet map lifecycle: tile layer, marker rendering, viewport
 * refreshes, scan drawing tools, and battlefront map locking. The map instance
 * is exposed through `mapRef` and the data loader through `loadMapDataRef` so
 * sibling features (locate, one-shot capture) can drive the map.
 */
export default function MapView({
  mapRef,
  loadMapDataRef,
  colorMode,
  battlefrontMapModeActive,
  battlefrontMarkersHidden,
  setStats,
  onMarkerPreview,
  onOneShotTarget,
  onBoundsSelected,
  onScanStatus
}) {
  const mapContainerRef = useRef(null);
  const markersRef = useRef(null);
  const markerRendererRef = useRef(null);
  const hasAutoFittedBoundsRef = useRef(false);
  const autoRefreshRef = useRef(null);
  const viewportAbortRef = useRef(null);
  const viewportDebounceRef = useRef(null);
  const battlefrontMapModeActiveRef = useRef(false);
  const battlefrontMarkersHiddenRef = useRef(false);
  const battlefrontMapResizeTimeoutRef = useRef(null);
  const handlersRef = useRef({ onMarkerPreview, onOneShotTarget, onBoundsSelected, onScanStatus });

  const [geojsonData, setGeojsonData] = useState(null);

  useEffect(() => {
    handlersRef.current = { onMarkerPreview, onOneShotTarget, onBoundsSelected, onScanStatus };
  });

  const renderMarkers = (geojson, mode) => {
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

      if (mode === 'recency') {
        const t = new Date(props.timestamp).getTime();
        color = interpolateColor('#0f3460', '#e94560', (t - minT) / rangeT);
      } else if (mode === 'heading') {
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
      marker.on('click', () => handlersRef.current.onMarkerPreview(props));
      markersRef.current.addLayer(marker);
    });
  };

  const loadDataWithMode = async (options = {}, mode) => {
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
        renderMarkers(geo, mode);
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

  const loadData = (options = {}) => loadDataWithMode(options, colorMode);

  useEffect(() => {
    loadMapDataRef.current = loadData;
  });

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
    const map = mapRef.current;
    if (!map) {
      return undefined;
    }
    const container = map.getContainer();
    container.classList.toggle('battlefrontMapLocked', battlefrontMapModeActive);
    if (battlefrontMapModeActive) {
      map.dragging.disable();
      map.scrollWheelZoom.disable();
      map.doubleClickZoom.disable();
      map.boxZoom.disable();
      map.keyboard.disable();
      if (map.touchZoom) {
        map.touchZoom.disable();
      }
      if (map.tap) {
        map.tap.disable();
      }
    } else {
      map.dragging.enable();
      map.scrollWheelZoom.enable();
      map.doubleClickZoom.enable();
      map.boxZoom.enable();
      map.keyboard.enable();
      if (map.touchZoom) {
        map.touchZoom.enable();
      }
      if (map.tap) {
        map.tap.enable();
      }
    }
    window.requestAnimationFrame(() => {
      map.invalidateSize(false);
    });
    if (battlefrontMapResizeTimeoutRef.current) {
      window.clearTimeout(battlefrontMapResizeTimeoutRef.current);
    }
    battlefrontMapResizeTimeoutRef.current = window.setTimeout(() => {
      map.invalidateSize(false);
      battlefrontMapResizeTimeoutRef.current = null;
    }, 280);
    return () => {
      if (battlefrontMapResizeTimeoutRef.current) {
        window.clearTimeout(battlefrontMapResizeTimeoutRef.current);
        battlefrontMapResizeTimeoutRef.current = null;
      }
      container.classList.remove('battlefrontMapLocked');
    };
  }, [battlefrontMapModeActive, mapRef]);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return undefined;

    const map = L.map(mapContainerRef.current, { preferCanvas: true }).setView([37.785, -122.43], 14);
    mapRef.current = map;
    markerRendererRef.current = L.canvas({ padding: 0.5 });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    markersRef.current = L.layerGroup().addTo(map);

    installMapDrawTools(map, {
      onBoundsSelected: (bounds, type, polygonCoords) =>
        handlersRef.current.onBoundsSelected(bounds, type, polygonCoords),
      onStatus: (text) => handlersRef.current.onScanStatus(text),
      onOneShotTarget: (lat, lon) => handlersRef.current.onOneShotTarget(lat, lon)
    });

    // Viewport-driven refreshes render with the color mode captured at map
    // init, matching the pre-refactor closure behavior; the colorMode effect
    // below recolors markers whenever the selector changes.
    const colorModeAtInit = colorMode;
    const loadDataAtInit = (options = {}) => loadDataWithMode(options, colorModeAtInit);

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
        loadDataAtInit({ forceStats: false });
      }, delayMs);
    };
    map.on('moveend', () => scheduleViewportRefresh(120));
    map.on('zoomend', () => scheduleViewportRefresh(120));

    loadDataAtInit({ forceStats: true });
    autoRefreshRef.current = setInterval(() => {
      loadDataAtInit({ forceStats: true, refreshViewport: false });
    }, 30000);

    return () => {
      if (autoRefreshRef.current) clearInterval(autoRefreshRef.current);
      if (viewportDebounceRef.current) clearTimeout(viewportDebounceRef.current);
      if (viewportAbortRef.current) viewportAbortRef.current.abort();
      map.doubleClickZoom.enable();
      map.remove();
      mapRef.current = null;
    };
    // Map is initialized exactly once; handlers reach current values via refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (geojsonData) renderMarkers(geojsonData, colorMode);
    // Only recolor on mode change; data-driven renders happen inside loadData.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorMode]);

  useEffect(() => {
    if (!mapRef.current) return;
    if (battlefrontMarkersHidden && markersRef.current) {
      markersRef.current.clearLayers();
      return;
    }
    loadData({ forceStats: false });
    // Refresh markers only when battlefront visibility flips; loadData is
    // recreated per render and would retrigger this effect every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [battlefrontMapModeActive, battlefrontMarkersHidden]);

  return (
    <div
      ref={mapContainerRef}
      id="map"
      className={battlefrontMapModeActive ? 'battlefrontMapFull' : ''}
    />
  );
}
