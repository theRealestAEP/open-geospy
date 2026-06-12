export const locateBattlefrontAudioPath = '/assets/loading-4s.mp3';
export const locateBattlefrontRevealStartDelayMs = 1000;
export const locateBattlefrontSearchZoomDurationMs = 4400;

const locateBattlefrontStageLineDurationMs = 3000;
const locateBattlefrontStagePauseMs = 700;
const locateBattlefrontStageZoomDurationSeconds = 1.9;
const locateBattlefrontStageSettleMs = 260;

const locateBattlefrontRevealStageBlueprints = [
  {
    key: 'acquiring',
    label: 'Country lock',
    title: 'Country lock acquired',
    subtitle: 'Resolving the winning family inside a national-scale tactical grid.',
    zoom: 5,
    lineDurationMs: locateBattlefrontStageLineDurationMs,
    pauseAfterLinesMs: locateBattlefrontStagePauseMs,
    zoomDurationSeconds: locateBattlefrontStageZoomDurationSeconds,
    settleMs: locateBattlefrontStageSettleMs
  },
  {
    key: 'approach',
    label: 'State corridor',
    title: 'State corridor resolved',
    subtitle: 'Compressing the search volume into a state-level ingress lane.',
    zoom: 8,
    lineDurationMs: locateBattlefrontStageLineDurationMs,
    pauseAfterLinesMs: locateBattlefrontStagePauseMs,
    zoomDurationSeconds: locateBattlefrontStageZoomDurationSeconds,
    settleMs: locateBattlefrontStageSettleMs
  },
  {
    key: 'descent',
    label: 'City / county',
    title: 'City / county descent',
    subtitle: 'Tightening the corridor onto the metro and county grid.',
    zoom: 12,
    lineDurationMs: locateBattlefrontStageLineDurationMs,
    pauseAfterLinesMs: locateBattlefrontStagePauseMs,
    zoomDurationSeconds: locateBattlefrontStageZoomDurationSeconds,
    settleMs: locateBattlefrontStageSettleMs
  },
  {
    key: 'terminal',
    label: 'Location lock',
    title: 'Location lock',
    subtitle: 'Finalizing the recovered panorama and committing the street position.',
    zoom: 21,
    lineDurationMs: locateBattlefrontStageLineDurationMs,
    pauseAfterLinesMs: locateBattlefrontStagePauseMs,
    zoomDurationSeconds: locateBattlefrontStageZoomDurationSeconds,
    settleMs: locateBattlefrontStageSettleMs
  }
];

export const locateBattlefrontRevealStages = [];
let locateBattlefrontRevealCursorMs = 0;
locateBattlefrontRevealStageBlueprints.forEach((stage) => {
  const zoomDelayMs = Number(stage.lineDurationMs || 0) + Number(stage.pauseAfterLinesMs || 0);
  const stageDurationMs =
    zoomDelayMs + Math.round(Number(stage.zoomDurationSeconds || 0) * 1000) + Number(stage.settleMs || 0);
  locateBattlefrontRevealStages.push({
    ...stage,
    at: locateBattlefrontRevealCursorMs,
    lineLeadMs: zoomDelayMs,
    zoomDelayMs,
    duration: Number(stage.zoomDurationSeconds || 0),
    stageDurationMs
  });
  locateBattlefrontRevealCursorMs += stageDurationMs;
});

export const locateBattlefrontLockDelayMs = locateBattlefrontRevealCursorMs;
export const locateBattlefrontClearDelayMs = locateBattlefrontLockDelayMs + 1800;
