import { useCallback, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import BattlefrontOverlay from './components/BattlefrontOverlay';
import EvalPage from './components/EvalPage';
import ImageSearchPanel from './components/ImageSearchPanel';
import LocatePage from './components/LocatePage';
import MapView from './components/MapView';
import OrbPopup from './components/OrbPopup';
import PreviewCard from './components/PreviewCard';
import ScanPage from './components/ScanPage';
import useImageSearch from './hooks/useImageSearch';
import useLocate from './hooks/useLocate';
import useScanJobs from './hooks/useScanJobs';
import { formatRetrievalModelLabel } from './lib/format';

/**
 * App shell: routing, the shared Leaflet map, and the state that genuinely
 * crosses pages (stats, capture preview, the uploaded reference image, and
 * embedding-base options). Page-specific state lives in the page hooks.
 */
function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const mapRef = useRef(null);
  const loadMapDataRef = useRef(null);
  const currentPathRef = useRef(location.pathname);

  const [stats, setStats] = useState({ total_panoramas: 0, total_captures: 0, bounds: null });
  const [colorMode, setColorMode] = useState('recency');
  const [preview, setPreview] = useState({ title: 'Click a marker to preview captures', captures: [] });

  const [retrievalFile, setRetrievalFile] = useState(null);
  const [retrievalFilePreviewUrl, setRetrievalFilePreviewUrl] = useState('');
  const [retrievalMinSimilarity, setRetrievalMinSimilarity] = useState('');
  const [searchEmbeddingBase, setSearchEmbeddingBase] = useState('');
  const [locateEmbeddingBase, setLocateEmbeddingBase] = useState('');
  const [retrievalEmbeddingBaseOptions, setRetrievalEmbeddingBaseOptions] = useState([
    { value: 'clip', label: 'CLIP' }
  ]);
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
  const isEvalPage = location.pathname === '/eval';

  const loadRetrievalStats = useCallback(async () => {
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
  }, []);

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

  const scan = useScanJobs({ setStats, loadMapDataRef });

  const locate = useLocate({
    mapRef,
    isLocatePage,
    currentPathRef,
    retrievalFile,
    retrievalMinSimilarity,
    locateEmbeddingBase,
    showPreview,
    loadRetrievalStats
  });

  const search = useImageSearch({
    retrievalFile,
    searchEmbeddingBase,
    retrievalMinSimilarity,
    clearSearchFallbackMarker: locate.clearSearchFallbackMarker,
    focusRetrievalResult: locate.focusRetrievalResult,
    loadRetrievalStats
  });

  useEffect(() => {
    currentPathRef.current = location.pathname;
  }, [location.pathname]);

  useEffect(() => {
    if (
      location.pathname !== '/scan' &&
      location.pathname !== '/search' &&
      location.pathname !== '/locate' &&
      location.pathname !== '/eval'
    ) {
      navigate('/scan', { replace: true });
    }
  }, [location.pathname, navigate]);

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
    loadRetrievalStats();
  }, [loadRetrievalStats]);

  const locateModelLabel =
    retrievalEmbeddingBaseOptions.find((option) => option.value === locateEmbeddingBase)?.label ||
    String(locateEmbeddingBase || 'active').toUpperCase();
  const locateEvalSettings = {
    locateTopK: locate.locateTopK,
    locateEmbeddingBase,
    locateOrbEnabled: locate.locateOrbEnabled,
    locateOrbTopN: locate.locateOrbTopN,
    locateOrbWeight: locate.locateOrbWeight,
    locateOrbFeatureCount: locate.locateOrbFeatureCount,
    locateOrbRansacTopK: locate.locateOrbRansacTopK,
    locateOrbIgnoreBottomRatio: locate.locateOrbIgnoreBottomRatio,
    locateSam2MaskCars: locate.locateSam2MaskCars,
    locateSam2MaskTrees: locate.locateSam2MaskTrees,
    retrievalMinSimilarity
  };
  const locateEvalBoundary = {
    minLat: Number(scan.scanForm.minLat),
    minLon: Number(scan.scanForm.minLon),
    maxLat: Number(scan.scanForm.maxLat),
    maxLon: Number(scan.scanForm.maxLon),
    shapeType: scan.scanShapeType || 'bbox',
    polygonCoords: scan.scanShapeType === 'polygon' ? scan.selectedPolygonCoords : []
  };

  return (
    <div className="app">
      <MapView
        mapRef={mapRef}
        loadMapDataRef={loadMapDataRef}
        colorMode={colorMode}
        battlefrontMapModeActive={locate.battlefrontMapModeActive}
        battlefrontMarkersHidden={locate.battlefrontMarkersHidden}
        setStats={setStats}
        onMarkerPreview={showPreview}
        onOneShotTarget={scan.setOneShotTarget}
        onBoundsSelected={scan.applyDrawnBounds}
        onScanStatus={scan.setScanStatusText}
      />
      <aside id="sidebar" className={locate.battlefrontMapModeActive ? 'sidebarHidden' : ''}>
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
          <button
            className={isEvalPage ? '' : 'ghost'}
            onClick={() => navigate('/eval')}
          >
            Eval
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

        {isScanPage ? <ScanPage scan={scan} /> : null}

        {isSearchPage ? (
          <>
            <ImageSearchPanel
              search={search}
              retrievalStats={retrievalStats}
              onRetrievalFileChange={setRetrievalFile}
              searchEmbeddingBase={searchEmbeddingBase}
              setSearchEmbeddingBase={setSearchEmbeddingBase}
              retrievalEmbeddingBaseOptions={retrievalEmbeddingBaseOptions}
              retrievalMinSimilarity={retrievalMinSimilarity}
              setRetrievalMinSimilarity={setRetrievalMinSimilarity}
              focusRetrievalResult={locate.focusRetrievalResult}
            />
            <PreviewCard preview={preview} emptyHint="Click a marker to preview captures." />
          </>
        ) : null}

        {isLocatePage ? (
          <>
            <LocatePage
              locate={locate}
              onRetrievalFileChange={setRetrievalFile}
              retrievalMinSimilarity={retrievalMinSimilarity}
              setRetrievalMinSimilarity={setRetrievalMinSimilarity}
              locateEmbeddingBase={locateEmbeddingBase}
              setLocateEmbeddingBase={setLocateEmbeddingBase}
              retrievalEmbeddingBaseOptions={retrievalEmbeddingBaseOptions}
            />
            <PreviewCard preview={preview} emptyHint="Run locate and click a family to preview captures." />
          </>
        ) : null}

        {isEvalPage ? (
          <EvalPage
            locateSettings={locateEvalSettings}
            boundary={locateEvalBoundary}
          />
        ) : null}
      </aside>
      {isLocatePage && locate.locateOrbPopupOpen ? (
        <OrbPopup
          locate={locate}
          retrievalFilePreviewUrl={retrievalFilePreviewUrl}
          locateModelLabel={locateModelLabel}
        />
      ) : null}
      {isLocatePage && locate.locateBattlefrontSequence ? (
        <BattlefrontOverlay
          locate={locate}
          retrievalFilePreviewUrl={retrievalFilePreviewUrl}
          locateModelLabel={locateModelLabel}
        />
      ) : null}
    </div>
  );
}

export default App;
