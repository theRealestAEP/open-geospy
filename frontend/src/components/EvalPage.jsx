import { useEffect, useRef, useState } from 'react';

import {
  getLocatorEvalDatasetStatus,
  getLocatorEvalStatus,
  startLocatorEvalDatasetScrape,
  startLocatorEval,
  stopLocatorEvalDataset,
  stopLocatorEval
} from '../api';

const defaultEvalSettingsJson = JSON.stringify(
  [
    {
      id: 'clip-baseline',
      top_k: 8,
      embedding_base: 'clip',
      orb_enabled: false
    }
  ],
  null,
  2
);

function buildLocateSettingsJson(locateSettings) {
  const minSimilarityRaw = String(locateSettings.retrievalMinSimilarity ?? '').trim();
  const parsedMinSimilarity =
    minSimilarityRaw === '' ? NaN : Number(minSimilarityRaw);
  const settings = {
    id: locateSettings.locateOrbEnabled ? 'current-orb' : 'current-baseline',
    top_k: Math.max(1, Number(locateSettings.locateTopK) || 8),
    embedding_base: locateSettings.locateEmbeddingBase || 'clip',
    orb_enabled: Boolean(locateSettings.locateOrbEnabled)
  };
  if (!Number.isNaN(parsedMinSimilarity)) {
    settings.min_similarity = parsedMinSimilarity;
  }
  if (locateSettings.locateOrbEnabled) {
    settings.orb_top_n = Math.max(1, Number(locateSettings.locateOrbTopN) || 100);
    settings.orb_weight = Number(locateSettings.locateOrbWeight) || 0.75;
    settings.orb_feature_count = Math.max(
      100,
      Math.min(2000, Number(locateSettings.locateOrbFeatureCount) || 500)
    );
    settings.orb_ransac_top_k = Math.max(
      0,
      Number(locateSettings.locateOrbRansacTopK) || 0
    );
    settings.orb_ignore_bottom_ratio = Math.max(
      0,
      Math.min(0.6, Number(locateSettings.locateOrbIgnoreBottomRatio) || 0)
    );
    settings.sam2_mask_cars = Boolean(locateSettings.locateSam2MaskCars);
    settings.sam2_mask_trees = Boolean(locateSettings.locateSam2MaskTrees);
  }
  return JSON.stringify([settings], null, 2);
}

export default function EvalPage({ locateSettings, boundary }) {
  const [casesPath, setCasesPath] = useState('eval/datasets/locator_cases.csv');
  const [endpoint, setEndpoint] = useState(
    'http://127.0.0.1:8000/api/retrieval/locate-by-image'
  );
  const [outputDir, setOutputDir] = useState('');
  const [limit, setLimit] = useState('');
  const [concurrency, setConcurrency] = useState(1);
  const [settingsJson, setSettingsJson] = useState(defaultEvalSettingsJson);
  const [job, setJob] = useState(null);
  const [statusText, setStatusText] = useState('');
  const [scrapeCount, setScrapeCount] = useState(50);
  const [scrapeViewsPerPanorama, setScrapeViewsPerPanorama] = useState(1);
  const [scrapeOutputDir, setScrapeOutputDir] = useState('');
  const [scrapeJob, setScrapeJob] = useState(null);
  const [scrapeStatusText, setScrapeStatusText] = useState('');
  const pollRef = useRef(null);
  const scrapePollRef = useRef(null);

  const stopPolling = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const stopScrapePolling = () => {
    if (scrapePollRef.current) {
      window.clearInterval(scrapePollRef.current);
      scrapePollRef.current = null;
    }
  };

  const pollEval = async (evalId) => {
    if (!evalId) return null;
    try {
      const body = await getLocatorEvalStatus(evalId);
      setJob(body);
      const status = String(body.status || 'unknown');
      const summaries = Array.isArray(body?.summary?.summaries)
        ? body.summary.summaries
        : body?.summary
          ? [body.summary]
          : [];
      setStatusText(
        summaries.length
          ? `Eval ${status}. ${summaries.length} settings summarized.`
          : `Eval ${status}.`
      );
      if (['completed', 'failed', 'stopped'].includes(status)) {
        stopPolling();
      }
      return body;
    } catch (error) {
      setStatusText(`Eval status failed: ${error.message}`);
      return null;
    }
  };

  const startPolling = (evalId) => {
    stopPolling();
    pollRef.current = window.setInterval(() => {
      void pollEval(evalId);
    }, 2000);
  };

  const pollScrape = async (datasetId) => {
    if (!datasetId) return null;
    try {
      const body = await getLocatorEvalDatasetStatus(datasetId);
      setScrapeJob(body);
      const status = String(body.status || 'unknown');
      const writtenCases = Number(body?.metadata?.summary?.written_cases || 0);
      setScrapeStatusText(
        writtenCases > 0
          ? `Dataset ${status}. ${writtenCases} cases written.`
          : `Dataset ${status}.`
      );
      if (status === 'completed' && body.manifest_path_relative) {
        setCasesPath(body.manifest_path_relative);
      }
      if (['completed', 'failed', 'stopped'].includes(status)) {
        stopScrapePolling();
      }
      return body;
    } catch (error) {
      setScrapeStatusText(`Dataset status failed: ${error.message}`);
      return null;
    }
  };

  const startScrapePolling = (datasetId) => {
    stopScrapePolling();
    scrapePollRef.current = window.setInterval(() => {
      void pollScrape(datasetId);
    }, 2000);
  };

  useEffect(
    () => () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      if (scrapePollRef.current) {
        window.clearInterval(scrapePollRef.current);
        scrapePollRef.current = null;
      }
    },
    []
  );

  const scrapeDataset = async () => {
    const minLat = Number(boundary?.minLat);
    const minLon = Number(boundary?.minLon);
    const maxLat = Number(boundary?.maxLat);
    const maxLon = Number(boundary?.maxLon);
    if (![minLat, minLon, maxLat, maxLon].every(Number.isFinite)) {
      setScrapeStatusText('Draw or select a valid eval zone first.');
      return;
    }
    stopScrapePolling();
    setScrapeStatusText('Starting dataset scrape...');
    try {
      const body = await startLocatorEvalDatasetScrape({
        min_lat: minLat,
        min_lon: minLon,
        max_lat: maxLat,
        max_lon: maxLon,
        polygon_coords: boundary?.shapeType === 'polygon' ? boundary?.polygonCoords || [] : [],
        count: Math.max(1, Math.min(500, Number(scrapeCount) || 50)),
        output_dir: scrapeOutputDir,
        views_per_panorama: Math.max(1, Math.min(8, Number(scrapeViewsPerPanorama) || 1)),
        no_db_ids: true
      });
      setScrapeJob(body);
      setScrapeStatusText(`Dataset scrape running: ${body.dataset_id}`);
      startScrapePolling(body.dataset_id);
    } catch (error) {
      setScrapeStatusText(`Dataset scrape failed to start: ${error.message}`);
    }
  };

  const stopScrape = async () => {
    const datasetId = String(scrapeJob?.dataset_id || '').trim();
    if (!datasetId) return;
    try {
      const body = await stopLocatorEvalDataset(datasetId);
      setScrapeJob(body);
      setScrapeStatusText(`Dataset ${String(body.status || 'stopped')}.`);
      stopScrapePolling();
    } catch (error) {
      setScrapeStatusText(`Dataset stop failed: ${error.message}`);
    }
  };

  const runEval = async () => {
    try {
      if (String(settingsJson || '').trim()) {
        JSON.parse(settingsJson);
      }
    } catch (error) {
      setStatusText(`Settings JSON is invalid: ${error.message}`);
      return;
    }
    stopPolling();
    setStatusText('Starting locator eval...');
    try {
      const body = await startLocatorEval({
        cases: casesPath,
        endpoint,
        output_dir: outputDir,
        settings_json: settingsJson,
        limit: Math.max(0, Number(limit) || 0),
        concurrency: Math.max(1, Math.min(8, Number(concurrency) || 1))
      });
      setJob(body);
      setStatusText(`Eval running: ${body.eval_id}`);
      startPolling(body.eval_id);
    } catch (error) {
      setStatusText(`Eval failed to start: ${error.message}`);
    }
  };

  const stopEval = async () => {
    const evalId = String(job?.eval_id || '').trim();
    if (!evalId) return;
    try {
      const body = await stopLocatorEval(evalId);
      setJob(body);
      setStatusText(`Eval ${String(body.status || 'stopped')}.`);
      stopPolling();
    } catch (error) {
      setStatusText(`Eval stop failed: ${error.message}`);
    }
  };

  const jobStatus = String(job?.status || '').trim().toLowerCase();
  const isRunning = jobStatus === 'running';
  const scrapeJobStatus = String(scrapeJob?.status || '').trim().toLowerCase();
  const scrapeIsRunning = scrapeJobStatus === 'running';
  const boundaryLabel =
    Number.isFinite(Number(boundary?.minLat)) &&
    Number.isFinite(Number(boundary?.minLon)) &&
    Number.isFinite(Number(boundary?.maxLat)) &&
    Number.isFinite(Number(boundary?.maxLon))
      ? `${Number(boundary.minLat).toFixed(5)}, ${Number(boundary.minLon).toFixed(5)} -> ${Number(boundary.maxLat).toFixed(5)}, ${Number(boundary.maxLon).toFixed(5)}`
      : 'No eval zone selected';
  const summaryRows = Array.isArray(job?.summary?.summaries)
    ? job.summary.summaries
    : job?.summary
      ? [job.summary]
      : [];
  const bestSummary = summaryRows.length
    ? [...summaryRows].sort(
        (a, b) =>
          Number(b.within_50m || 0) - Number(a.within_50m || 0) ||
          Number(a.median_error_m || 1e18) - Number(b.median_error_m || 1e18)
      )[0]
    : null;

  return (
    <>
      <section className="card">
        <h2>Eval dataset</h2>
        <div className="evalBoundaryReadout">
          <span>{boundary?.shapeType === 'polygon' ? 'Polygon' : 'BBox'}</span>
          <span>{boundaryLabel}</span>
        </div>
        <div className="grid2">
          <label>
            Query images
            <input
              type="number"
              min="1"
              max="500"
              value={scrapeCount}
              onChange={(event) => setScrapeCount(event.target.value)}
              disabled={scrapeIsRunning}
            />
          </label>
          <label>
            Views per pano
            <input
              type="number"
              min="1"
              max="8"
              value={scrapeViewsPerPanorama}
              onChange={(event) => setScrapeViewsPerPanorama(event.target.value)}
              disabled={scrapeIsRunning}
            />
          </label>
        </div>
        <label>
          Dataset output dir
          <input
            value={scrapeOutputDir}
            onChange={(event) => setScrapeOutputDir(event.target.value)}
            placeholder="auto"
            disabled={scrapeIsRunning}
          />
        </label>
        <div className="buttonRow">
          <button onClick={scrapeDataset} disabled={scrapeIsRunning}>
            {scrapeIsRunning ? 'Scraping...' : 'Scrape eval zone'}
          </button>
          <button className="ghost" onClick={stopScrape} disabled={!scrapeIsRunning}>
            Stop scrape
          </button>
          <button
            className="ghost"
            onClick={() => pollScrape(scrapeJob?.dataset_id)}
            disabled={!scrapeJob?.dataset_id}
          >
            Refresh scrape
          </button>
        </div>
        <p className="status">{scrapeStatusText}</p>
        {scrapeJob ? (
          <div className="evalJobPanel">
            <div className="row"><span>Dataset ID</span><span>{scrapeJob.dataset_id}</span></div>
            <div className="row"><span>Status</span><span>{scrapeJob.status}</span></div>
            <div className="row"><span>Cases CSV</span><span>{scrapeJob.manifest_path_relative || 'pending'}</span></div>
          </div>
        ) : null}
        {String(scrapeJob?.stdout_tail || '').trim() ? (
          <details className="evalLogPanel">
            <summary>dataset stdout</summary>
            <pre>{scrapeJob.stdout_tail}</pre>
          </details>
        ) : null}
        {String(scrapeJob?.stderr_tail || '').trim() ? (
          <details className="evalLogPanel">
            <summary>dataset stderr</summary>
            <pre>{scrapeJob.stderr_tail}</pre>
          </details>
        ) : null}
      </section>

      <section className="card">
        <h2>Locator eval</h2>
        <div className="grid2">
          <label>
            Cases CSV
            <input
              value={casesPath}
              onChange={(event) => setCasesPath(event.target.value)}
              disabled={isRunning}
            />
          </label>
          <label>
            Output dir
            <input
              value={outputDir}
              onChange={(event) => setOutputDir(event.target.value)}
              placeholder="auto"
              disabled={isRunning}
            />
          </label>
          <label>
            Limit
            <input
              type="number"
              min="0"
              value={limit}
              onChange={(event) => setLimit(event.target.value)}
              placeholder="all"
              disabled={isRunning}
            />
          </label>
          <label>
            Concurrency
            <input
              type="number"
              min="1"
              max="8"
              value={concurrency}
              onChange={(event) => setConcurrency(event.target.value)}
              disabled={isRunning}
            />
          </label>
        </div>
        <label>
          Endpoint
          <input
            value={endpoint}
            onChange={(event) => setEndpoint(event.target.value)}
            disabled={isRunning}
          />
        </label>
        <label>
          Settings JSON
          <textarea
            className="evalSettingsInput"
            value={settingsJson}
            onChange={(event) => setSettingsJson(event.target.value)}
            rows={12}
            spellCheck="false"
            disabled={isRunning}
          />
        </label>
        <div className="buttonRow">
          <button onClick={runEval} disabled={isRunning}>
            {isRunning ? 'Eval running...' : 'Run eval'}
          </button>
          <button
            className="ghost"
            onClick={() => setSettingsJson(buildLocateSettingsJson(locateSettings))}
            disabled={isRunning}
          >
            Use Locate settings
          </button>
          <button className="ghost" onClick={stopEval} disabled={!isRunning}>
            Stop
          </button>
          <button
            className="ghost"
            onClick={() => pollEval(job?.eval_id)}
            disabled={!job?.eval_id}
          >
            Refresh
          </button>
        </div>
        <p className="status">{statusText}</p>
        {job ? (
          <div className="evalJobPanel">
            <div className="row"><span>Eval ID</span><span>{job.eval_id}</span></div>
            <div className="row"><span>Status</span><span>{job.status}</span></div>
            <div className="row"><span>Output</span><span>{job.output_dir || 'pending'}</span></div>
            {bestSummary ? (
              <div className="evalBestBar">
                <span>best {bestSummary.settings_id}</span>
                <span>{Number(bestSummary.within_50m || 0).toFixed(2)}% @50m</span>
                <span>{Number(bestSummary.median_error_m || 0).toFixed(1)}m median</span>
              </div>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="card">
        <h2>Summaries</h2>
        {summaryRows.length ? (
          <div className="evalSummaryGrid">
            {summaryRows.map((summary) => (
              <div key={summary.settings_id || summary.setting_settings_id} className="evalSummaryCard">
                <div className="evalSummaryTitle">
                  {summary.settings_id || summary.setting_settings_id || 'settings'}
                </div>
                <div className="progressGrid">
                  <div>Cases: {Number(summary.total_cases || 0)}</div>
                  <div>OK: {Number(summary.ok_rate || 0).toFixed(1)}%</div>
                  <div>25m: {Number(summary.within_25m || 0).toFixed(1)}%</div>
                  <div>50m: {Number(summary.within_50m || 0).toFixed(1)}%</div>
                  <div>100m: {Number(summary.within_100m || 0).toFixed(1)}%</div>
                  <div>Median: {Number(summary.median_error_m || 0).toFixed(1)}m</div>
                  <div>Pano top1: {Number(summary.panorama_top1 || 0).toFixed(1)}%</div>
                  <div>Reject: {Number(summary.reject_rate || 0).toFixed(1)}%</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="hint">No eval summary yet.</p>
        )}
        {String(job?.stdout_tail || '').trim() ? (
          <details className="evalLogPanel">
            <summary>stdout</summary>
            <pre>{job.stdout_tail}</pre>
          </details>
        ) : null}
        {String(job?.stderr_tail || '').trim() ? (
          <details className="evalLogPanel">
            <summary>stderr</summary>
            <pre>{job.stderr_tail}</pre>
          </details>
        ) : null}
      </section>
    </>
  );
}
