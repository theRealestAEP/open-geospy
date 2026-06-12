import L from 'leaflet';

const MIN_POLYGON_POINT_DISTANCE_METERS = 2.0;

function setMapControlTooltips(map) {
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
}

/**
 * Installs the scan drawing tools (rectangle/polygon edit control, point mode,
 * polygon mode, free draw mode) on a Leaflet map. All drawing state is local to
 * this installer; `map.remove()` tears everything down.
 *
 * Handlers:
 * - onBoundsSelected(bounds, shapeType, polygonCoords): a scan shape was committed.
 * - onStatus(text): scan status line updates while drawing.
 * - onOneShotTarget(lat, lon): a one-shot capture target was placed.
 */
export default function installMapDrawTools(map, { onBoundsSelected, onStatus, onOneShotTarget }) {
  const drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);
  const polygonDraftVertexLayer = L.layerGroup().addTo(map);

  let pointModeEnabled = false;
  let freeDrawEnabled = false;
  let freeDrawDrawing = false;
  let freeDrawLatLngs = [];
  let freeDrawPreview = null;
  let polygonModeEnabled = false;
  let polygonDraftLatLngs = [];
  let polygonDraftPreview = null;
  let pointModeButton = null;
  let polygonModeButton = null;
  let freeDrawButton = null;
  let oneShotMarker = null;

  const drawControl = new L.Control.Draw({
    edit: {
      featureGroup: drawnItems,
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

  const setOneShotTarget = (lat, lon) => {
    onOneShotTarget(lat, lon);
    if (oneShotMarker) {
      map.removeLayer(oneShotMarker);
    }
    oneShotMarker = L.circleMarker([lat, lon], {
      radius: 7,
      fillColor: '#ffd166',
      fillOpacity: 0.95,
      color: '#111',
      weight: 2
    }).addTo(map);
    oneShotMarker.bindTooltip('One-shot target');
  };

  const applyLayerBounds = (layer, type) => {
    const bounds = layer.getBounds();
    const nextBounds = {
      minLat: Number(bounds.getSouth().toFixed(6)),
      minLon: Number(bounds.getWest().toFixed(6)),
      maxLat: Number(bounds.getNorth().toFixed(6)),
      maxLon: Number(bounds.getEast().toFixed(6))
    };
    let polygonCoords = [];
    if (type === 'polygon') {
      const latlngs = layer.getLatLngs();
      const ring = Array.isArray(latlngs?.[0]) ? latlngs[0] : [];
      polygonCoords = ring.map((ll) => [Number(ll.lat.toFixed(6)), Number(ll.lng.toFixed(6))]);
    }
    onBoundsSelected(nextBounds, type, polygonCoords);
  };

  const syncModeButtonStyles = () => {
    if (pointModeButton) {
      pointModeButton.classList.toggle('active', pointModeEnabled);
    }
    if (polygonModeButton) {
      polygonModeButton.classList.toggle('active', polygonModeEnabled);
    }
    if (freeDrawButton) {
      freeDrawButton.classList.toggle('active', freeDrawEnabled);
    }
  };

  const clearFreeDrawPreview = () => {
    if (freeDrawPreview) {
      map.removeLayer(freeDrawPreview);
      freeDrawPreview = null;
    }
    freeDrawLatLngs = [];
    freeDrawDrawing = false;
    map.dragging.enable();
  };

  const renderPolygonDraftPreview = (cursorLatLng = null) => {
    if (polygonDraftPreview) {
      map.removeLayer(polygonDraftPreview);
      polygonDraftPreview = null;
    }
    polygonDraftVertexLayer.clearLayers();

    const points = polygonDraftLatLngs;
    points.forEach((point) => {
      L.circleMarker(point, {
        radius: 4,
        color: '#2f8cff',
        weight: 2,
        fillColor: '#ffffff',
        fillOpacity: 1
      }).addTo(polygonDraftVertexLayer);
    });

    const previewPoints = cursorLatLng ? [...points, cursorLatLng] : [...points];
    if (previewPoints.length >= 3) {
      polygonDraftPreview = L.polygon(previewPoints, {
        color: '#2f8cff',
        weight: 2,
        opacity: 0.95,
        fillColor: '#2f8cff',
        fillOpacity: 0.16,
        dashArray: '6 4'
      }).addTo(map);
    } else if (previewPoints.length >= 2) {
      polygonDraftPreview = L.polyline(previewPoints, {
        color: '#2f8cff',
        weight: 3,
        opacity: 0.85,
        dashArray: '6 4'
      }).addTo(map);
    }
  };

  const clearPolygonDraftPreview = () => {
    if (polygonDraftPreview) {
      map.removeLayer(polygonDraftPreview);
      polygonDraftPreview = null;
    }
    polygonDraftVertexLayer.clearLayers();
    polygonDraftLatLngs = [];
    map.doubleClickZoom.enable();
  };

  const finishPolygonDraft = () => {
    const points = polygonDraftLatLngs;
    if (!polygonModeEnabled) return;
    if (points.length < 3) {
      onStatus('Polygon mode needs at least 3 points.');
      return;
    }
    const polygonLayer = L.polygon(points, {
      color: '#e94560',
      weight: 2,
      opacity: 0.95,
      fillColor: '#e94560',
      fillOpacity: 0.18
    });
    drawnItems.clearLayers();
    drawnItems.addLayer(polygonLayer);
    applyLayerBounds(polygonLayer, 'polygon');
    onStatus(`Polygon shape ready (${points.length} points).`);
    polygonModeEnabled = false;
    map.getContainer().style.cursor = '';
    clearPolygonDraftPreview();
    syncModeButtonStyles();
  };

  const finishFreeDrawShape = (releaseLatLng) => {
    if (!freeDrawEnabled || !freeDrawDrawing) return;
    freeDrawDrawing = false;
    map.dragging.enable();
    if (releaseLatLng) {
      const points = freeDrawLatLngs;
      const last = points[points.length - 1];
      if (!last || map.distance(last, releaseLatLng) >= 0.25) {
        points.push(releaseLatLng);
      }
    }
    if (freeDrawPreview) {
      map.removeLayer(freeDrawPreview);
      freeDrawPreview = null;
    }
    const points = freeDrawLatLngs;
    if (points.length < 3) {
      onStatus('Free draw needs a larger gesture (at least 3 points).');
      freeDrawLatLngs = [];
      return;
    }

    const polygonLayer = L.polygon(points, {
      color: '#e94560',
      weight: 2,
      opacity: 0.95,
      fillColor: '#e94560',
      fillOpacity: 0.18
    });
    drawnItems.clearLayers();
    drawnItems.addLayer(polygonLayer);
    applyLayerBounds(polygonLayer, 'polygon');
    onStatus(`Free draw shape ready (${points.length} points).`);
    freeDrawLatLngs = [];
  };

  const togglePointMode = () => {
    pointModeEnabled = !pointModeEnabled;
    if (pointModeEnabled) {
      polygonModeEnabled = false;
      freeDrawEnabled = false;
      clearPolygonDraftPreview();
      clearFreeDrawPreview();
      onStatus('Point mode enabled. Click map to set one-shot target.');
    } else {
      onStatus('');
    }
    map.getContainer().style.cursor = pointModeEnabled ? 'crosshair' : '';
    syncModeButtonStyles();
  };

  const togglePolygonMode = () => {
    polygonModeEnabled = !polygonModeEnabled;
    if (polygonModeEnabled) {
      pointModeEnabled = false;
      freeDrawEnabled = false;
      clearFreeDrawPreview();
      clearPolygonDraftPreview();
      map.doubleClickZoom.disable();
      onStatus('Polygon mode enabled. Click to add points, double-click to finish.');
      renderPolygonDraftPreview();
    } else {
      clearPolygonDraftPreview();
      onStatus('');
    }
    map.getContainer().style.cursor = polygonModeEnabled ? 'crosshair' : '';
    syncModeButtonStyles();
  };

  const toggleFreeDrawMode = () => {
    freeDrawEnabled = !freeDrawEnabled;
    if (freeDrawEnabled) {
      pointModeEnabled = false;
      polygonModeEnabled = false;
      clearPolygonDraftPreview();
      onStatus('Free draw enabled. Click and drag on map to sketch scan shape.');
    } else {
      clearFreeDrawPreview();
      onStatus('');
    }
    map.getContainer().style.cursor = freeDrawEnabled ? 'crosshair' : '';
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
    const freeButton = L.DomUtil.create('a', 'geospy-mode-free', container);
    freeButton.href = '#';
    freeButton.textContent = 'FD';

    pointModeButton = pointButton;
    polygonModeButton = polygonButton;
    freeDrawButton = freeButton;

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
    L.DomEvent.on(freeButton, 'click', (evt) => {
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
    drawnItems.clearLayers();
    drawnItems.addLayer(e.layer);
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
    onBoundsSelected(null, null, []);
  });

  map.on('mousedown', (e) => {
    if (!freeDrawEnabled) return;
    if (e.originalEvent?.button !== 0) return;
    freeDrawDrawing = true;
    freeDrawLatLngs = [e.latlng];
    map.dragging.disable();
    if (freeDrawPreview) {
      map.removeLayer(freeDrawPreview);
    }
    freeDrawPreview = L.polyline(freeDrawLatLngs, {
      color: '#2f8cff',
      weight: 3,
      opacity: 0.85,
      dashArray: '6 4'
    }).addTo(map);
  });

  map.on('mousemove', (e) => {
    if (!freeDrawEnabled || !freeDrawDrawing) return;
    const points = freeDrawLatLngs;
    const last = points[points.length - 1];
    if (last && map.distance(last, e.latlng) < 0.75) {
      return;
    }
    points.push(e.latlng);
    if (freeDrawPreview) {
      freeDrawPreview.setLatLngs(points);
    }
  });

  map.on('mouseup', (e) => {
    finishFreeDrawShape(e?.latlng || null);
  });

  map.on('click', (e) => {
    if (polygonModeEnabled) {
      const points = polygonDraftLatLngs;
      const last = points[points.length - 1];
      if (last && map.distance(last, e.latlng) < MIN_POLYGON_POINT_DISTANCE_METERS) {
        onStatus(
          `Point ignored (too close). Move at least ${MIN_POLYGON_POINT_DISTANCE_METERS.toFixed(1)}m before placing next point.`
        );
        renderPolygonDraftPreview(e.latlng);
        return;
      }
      points.push(e.latlng);
      renderPolygonDraftPreview(e.latlng);
      onStatus(`Polygon mode: ${points.length} points. Double-click to finish.`);
      return;
    }
    if (pointModeEnabled || e.originalEvent?.altKey) {
      setOneShotTarget(e.latlng.lat, e.latlng.lng);
    }
  });

  map.on('mousemove', (e) => {
    if (polygonModeEnabled && !freeDrawDrawing) {
      renderPolygonDraftPreview(e.latlng);
    }
  });

  map.on('dblclick', () => {
    if (polygonModeEnabled) {
      finishPolygonDraft();
    }
  });

  map.on('contextmenu', (e) => {
    if (polygonModeEnabled) {
      finishPolygonDraft();
      return;
    }
    setOneShotTarget(e.latlng.lat, e.latlng.lng);
  });
}
