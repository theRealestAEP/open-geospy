import { useCallback, useEffect, useRef, useState } from 'react';

const ACTIVE_SCANS_STORAGE_KEY = 'geospy.active_scan_ids';
const LEGACY_ACTIVE_SCAN_STORAGE_KEY = 'geospy.active_scan_id';

export const defaultScanForm = {
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

export function isScanTerminalStatus(status) {
  const value = String(status || '');
  return value === 'finished' || value === 'stopped' || value.startsWith('failed:');
}

export function isScanActiveStatus(status) {
  const value = String(status || '');
  return value === 'running' || value === 'stopping' || value.startsWith('running');
}

/**
 * Scan-job state machine: area-scan form, drawn shape, one-shot capture,
 * job tracking with localStorage persistence, and status polling.
 */
export default function useScanJobs({ setStats, loadMapDataRef }) {
  const scanPollRef = useRef(null);
  const activeScanIdRef = useRef(null);
  const trackedScanIdsRef = useRef([]);

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
  const [oneShotLat, setOneShotLat] = useState('');
  const [oneShotLon, setOneShotLon] = useState('');
  const [oneShotStatus, setOneShotStatus] = useState('');

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

  const stopScanPolling = useCallback(() => {
    if (scanPollRef.current) {
      clearInterval(scanPollRef.current);
      scanPollRef.current = null;
    }
  }, []);

  const addTrackedScanId = useCallback((scanId, makePrimary = false) => {
    if (!scanId) return;
    setTrackedScanIds((prev) => {
      if (prev.includes(scanId)) return prev;
      return [...prev, scanId];
    });
    if (makePrimary || !activeScanIdRef.current) {
      setActiveScanId(scanId);
    }
  }, []);

  const removeTrackedScanId = useCallback((scanId) => {
    if (!scanId) return;
    setTrackedScanIds((prev) => prev.filter((id) => id !== scanId));
    if (activeScanIdRef.current === scanId) {
      setActiveScanId((prevActive) => {
        if (prevActive !== scanId) return prevActive;
        const remaining = trackedScanIdsRef.current.filter((id) => id !== scanId);
        return remaining[0] || null;
      });
    }
  }, []);

  const pollScanStatus = useCallback(
    async (scanIdOverride = null, options = {}) => {
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
    },
    [removeTrackedScanId, setStats, stopScanPolling]
  );

  const startScanPolling = useCallback(() => {
    if (scanPollRef.current) {
      clearInterval(scanPollRef.current);
    }
    scanPollRef.current = setInterval(() => {
      pollScanStatus();
    }, 3000);
  }, [pollScanStatus]);

  useEffect(() => {
    let isCancelled = false;
    const hydrateScanState = async () => {
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
    hydrateScanState();
    return () => {
      isCancelled = true;
    };
  }, [addTrackedScanId, pollScanStatus, startScanPolling]);

  useEffect(() => stopScanPolling, [stopScanPolling]);

  const onScanField = (field, value) => {
    setScanForm((prev) => ({ ...prev, [field]: value }));
  };

  const applyDrawnBounds = useCallback((bounds, type, polygonCoords) => {
    if (!bounds) {
      // The drawn shape was deleted from the map.
      setScanShapeType(null);
      setSelectedPolygonCoords([]);
      return;
    }
    // The pre-refactor map handlers closed over the initial scan form, so a
    // drawn shape always merged its bounds into the default form values.
    // Preserve that behavior.
    setScanForm({ ...defaultScanForm, ...bounds });
    setScanShapeType(type);
    setSelectedPolygonCoords(type === 'polygon' ? polygonCoords : []);
  }, []);

  const setOneShotTarget = useCallback((lat, lon) => {
    setOneShotLat(lat.toFixed(7));
    setOneShotLon(lon.toFixed(7));
  }, []);

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
      await loadMapDataRef.current?.();
      setOneShotStatus('Capture complete.');
    } catch (error) {
      setOneShotStatus(`Capture failed: ${error.message}`);
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

  return {
    scanForm,
    onScanField,
    scanShapeType,
    selectedPolygonCoords,
    applyDrawnBounds,
    scanStatusText,
    setScanStatusText,
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
    runOneShot,
    setOneShotTarget
  };
}
