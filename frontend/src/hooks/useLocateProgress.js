import { useEffect, useState } from 'react';

export default function useLocateProgress({
  locateBusy,
  locateProgressId,
  buildStagesFromProgress,
  onMessage
}) {
  const [liveProgress, setLiveProgress] = useState(null);
  const [pipeline, setPipeline] = useState({ stages: [] });

  useEffect(() => {
    if (!locateBusy || !locateProgressId) {
      return undefined;
    }
    let cancelled = false;
    const pollProgress = async () => {
      try {
        const response = await fetch(
          `/api/retrieval/progress/${encodeURIComponent(locateProgressId)}`,
          { cache: 'no-store' }
        );
        if (!response.ok) {
          return;
        }
        const body = await response.json();
        if (cancelled) {
          return;
        }
        setLiveProgress(body);
        setPipeline({ stages: buildStagesFromProgress(body) });
        const message = String(body?.message || '').trim();
        if (message) {
          onMessage(message);
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
  }, [buildStagesFromProgress, locateBusy, locateProgressId, onMessage]);

  return {
    liveProgress,
    pipeline,
    setLiveProgress,
    setPipeline
  };
}
