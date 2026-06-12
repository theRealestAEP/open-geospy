import { useCallback, useEffect, useRef, useState } from 'react';

import {
  locateBattlefrontAudioPath,
  locateBattlefrontClearDelayMs,
  locateBattlefrontLockDelayMs,
  locateBattlefrontRevealStages,
  locateBattlefrontSearchZoomDurationMs
} from '../lib/battlefront';

/**
 * Owns the cinematic "battlefront" locate reveal: the wide-area search zoom,
 * the staged reveal timeline (timeouts + map flyTo), and the audio cue.
 */
export default function useBattlefrontSequence({ mapRef }) {
  const timeoutsRef = useRef([]);
  const audioRef = useRef(null);
  const searchReadyPromiseRef = useRef(Promise.resolve());
  const searchReadyTimeoutRef = useRef(null);
  const searchMoveEndHandlerRef = useRef(null);

  const [sequence, setSequence] = useState(null);

  const stopAudio = useCallback((options = {}) => {
    const { reset = true } = options;
    const audio = audioRef.current;
    if (!audio) return;
    audio.pause();
    if (reset) {
      try {
        audio.currentTime = 0;
      } catch (error) {
        console.debug('battlefront audio reset failed', error);
      }
    }
  }, []);

  const playAudio = useCallback(() => {
    if (typeof window === 'undefined' || typeof Audio === 'undefined') {
      return;
    }
    let audio = audioRef.current;
    if (!audio) {
      audio = new Audio(locateBattlefrontAudioPath);
      audio.preload = 'auto';
      audio.loop = false;
      audio.volume = 0.72;
      audioRef.current = audio;
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
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined' || typeof Audio === 'undefined') {
      return undefined;
    }
    const audio = new Audio(locateBattlefrontAudioPath);
    audio.preload = 'auto';
    audio.loop = false;
    audio.volume = 0.72;
    audio.load();
    audioRef.current = audio;
    return () => {
      audio.pause();
      try {
        audio.currentTime = 0;
      } catch (error) {
        console.debug('battlefront audio cleanup failed', error);
      }
      if (audioRef.current === audio) {
        audioRef.current = null;
      }
    };
  }, []);

  const clearSequence = useCallback(
    (options = {}) => {
      timeoutsRef.current.forEach((timeoutId) => {
        window.clearTimeout(timeoutId);
      });
      timeoutsRef.current = [];
      if (searchReadyTimeoutRef.current) {
        window.clearTimeout(searchReadyTimeoutRef.current);
        searchReadyTimeoutRef.current = null;
      }
      if (
        mapRef.current &&
        searchMoveEndHandlerRef.current &&
        typeof mapRef.current.off === 'function'
      ) {
        mapRef.current.off('moveend', searchMoveEndHandlerRef.current);
        searchMoveEndHandlerRef.current = null;
      }
      searchReadyPromiseRef.current = Promise.resolve();
      if (options.stopMap && mapRef.current && typeof mapRef.current.stop === 'function') {
        mapRef.current.stop();
      }
      if (!options.keepAudio) {
        stopAudio();
      }
      setSequence(null);
    },
    [mapRef, stopAudio]
  );

  const queueStep = useCallback((delayMs, callback) => {
    const timeoutId = window.setTimeout(callback, delayMs);
    timeoutsRef.current.push(timeoutId);
  }, []);

  const waitForSearchReady = useCallback(async () => {
    const ready = searchReadyPromiseRef.current;
    if (ready && typeof ready.then === 'function') {
      await ready;
    }
  }, []);

  const beginSearch = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    clearSequence({ stopMap: true });
    searchReadyPromiseRef.current = new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        if (
          map &&
          searchMoveEndHandlerRef.current &&
          typeof map.off === 'function'
        ) {
          map.off('moveend', searchMoveEndHandlerRef.current);
        }
        searchMoveEndHandlerRef.current = null;
        if (searchReadyTimeoutRef.current) {
          window.clearTimeout(searchReadyTimeoutRef.current);
          searchReadyTimeoutRef.current = null;
        }
        resolve();
      };
      const handleMoveEnd = () => {
        window.setTimeout(finish, 180);
      };
      searchMoveEndHandlerRef.current = handleMoveEnd;
      if (typeof map.once === 'function') {
        map.once('moveend', handleMoveEnd);
      }
      searchReadyTimeoutRef.current = window.setTimeout(
        finish,
        locateBattlefrontSearchZoomDurationMs + 700
      );
    });
    const center = map.getCenter();
    setSequence({
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
  }, [clearSequence, mapRef]);

  const startReveal = useCallback(
    async (result, options = {}) => {
      const { onComplete, focusResult, clearSearchFallbackMarker } = options;
      const map = mapRef.current;
      const lat = Number(result?.lat ?? result?.family_center_lat);
      const lon = Number(result?.lon ?? result?.family_center_lon);
      if (!map || !Number.isFinite(lat) || !Number.isFinite(lon)) {
        await focusResult(result);
        if (typeof onComplete === 'function') {
          onComplete();
        }
        return;
      }

      clearSequence({ stopMap: true });
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
          playAudio();
          setSequence({
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
          queueStep(Number(stage.zoomDelayMs || stage.lineLeadMs || 0), () => {
            setSequence((prev) =>
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
          queueStep(stage.at, runStage);
        }
      });

      queueStep(locateBattlefrontLockDelayMs, () => {
        if (typeof map.stop === 'function') {
          map.stop();
        }
        setSequence((prev) =>
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
        void focusResult(result, {
          keepBattlefrontReveal: true,
          skipMapMove: true,
          skipPreview: true,
          preferredZoom: Math.max(
            20,
            Number(locateBattlefrontRevealStages[locateBattlefrontRevealStages.length - 1]?.zoom || 20)
          )
        });
        if (typeof onComplete === 'function') {
          onComplete();
        }
      });

      queueStep(locateBattlefrontClearDelayMs, () => {
        clearSequence();
      });
    },
    [clearSequence, mapRef, playAudio, queueStep]
  );

  return {
    sequence,
    clearSequence,
    beginSearch,
    waitForSearchReady,
    startReveal
  };
}
