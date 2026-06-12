import { isScanTerminalStatus } from '../hooks/useScanJobs';

/**
 * Scan page panels: one-shot capture plus the area-scan form and job tracker.
 * All state lives in useScanJobs (passed as `scan`) so it survives navigation.
 */
export default function ScanPage({ scan }) {
  const {
    scanForm,
    onScanField,
    scanStatusText,
    activeScanId,
    scanJobs,
    scanProgress,
    startScan,
    stopScan,
    addTrackedScanId,
    oneShotLat,
    setOneShotLat,
    oneShotLon,
    setOneShotLon,
    oneShotStatus,
    runOneShot
  } = scan;
  const currentWorkerLimit = scanForm.mode === 'modal' ? 100 : 32;

  return (
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
  );
}
