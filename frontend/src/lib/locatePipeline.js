export const orbPopupMoments = [
  'Extracting keypoints from the query frame.',
  'Sweeping candidate facades for matching fingerprints.',
  'Linking feature pairs and pruning weak matches.',
  'Testing the strongest alignments for geometric consistency.'
];

export const orbPopupPreviewStages = ['keypoints', 'scan', 'link', 'geometry'];

export const orbPopupPreviewPoints = [
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

export function buildLocatePipelineStages(activeKey = 'vector_search', status = 'running') {
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

export function mapLocateProgressPhaseToPreviewStage(phase, orbProgress) {
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

export function formatLocateProgressStageBadge(phase, orbEnabled) {
  const normalizedPhase = String(phase || '').trim().toLowerCase();
  if (normalizedPhase === 'vector_search') return 'Vector search';
  if (normalizedPhase === 'orb_rerank') return 'ORB rerank';
  if (normalizedPhase === 'panorama_rerank') return orbEnabled ? 'Panorama merge' : 'Panorama aggregation';
  if (normalizedPhase === 'family_rank') return 'Family ranking';
  if (normalizedPhase === 'completed') return 'Wrapping up';
  return orbEnabled ? 'Preparing ORB' : 'Preparing locate';
}

export function buildLocatePipelineStagesFromProgress(progress) {
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

export function getOrbPopupTimelineState(index, phase, activeMoment) {
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
