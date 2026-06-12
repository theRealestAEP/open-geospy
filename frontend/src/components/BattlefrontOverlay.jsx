import { locateBattlefrontRevealStages } from '../lib/battlefront';
import { formatCoordinate, formatMetric } from '../lib/format';
import { formatLocateProgressStageBadge } from '../lib/locatePipeline';

/**
 * Full-screen battlefront reveal overlay: tactical grid, HUD with stage
 * progress, and the recovered-target preview card.
 */
export default function BattlefrontOverlay({ locate, retrievalFilePreviewUrl, locateModelLabel }) {
  const {
    locateBattlefrontSequence,
    locatePipeline,
    locateOrbEnabled,
    liveLocateProgress,
    liveLocatePhase,
    locateOrbProcessedCandidates,
    locateOrbCandidateCount
  } = locate;

  const locateBattlefrontStepIndex = Math.max(0, Number(locateBattlefrontSequence?.stepIndex || 0));
  const locateBattlefrontSearching = locateBattlefrontSequence?.phase === 'searching';
  const locateBattlefrontHudTitle =
    locateBattlefrontSearching
      ? liveLocatePhase === 'vector_search'
        ? 'Country-scale vector sweep'
        : liveLocatePhase === 'orb_rerank'
          ? 'Local feature verification'
          : liveLocatePhase === 'panorama_rerank'
            ? 'Panorama merge underway'
            : liveLocatePhase === 'family_rank'
              ? 'Locking the landing zone'
              : 'Scanning planetary archive'
      : String(locateBattlefrontSequence?.title || '').trim();
  const locateBattlefrontHudSubtitle =
    locateBattlefrontSearching
      ? String(liveLocateProgress?.message || '').trim() ||
        'Zoomed out while the backend resolves a landing zone from the live locate pipeline.'
      : String(locateBattlefrontSequence?.subtitle || '').trim();
  const locateBattlefrontMetaItems = locateBattlefrontSearching
    ? [
        `model ${locateModelLabel}`,
        `phase ${String(formatLocateProgressStageBadge(liveLocatePhase, locateOrbEnabled)).toLowerCase()}`,
        Number(liveLocateProgress?.vector_candidates || 0) > 0
          ? `raw ${Number(liveLocateProgress?.vector_candidates || 0)}`
          : '',
        locateOrbEnabled && locateOrbCandidateCount > 0
          ? `orb ${locateOrbProcessedCandidates}/${locateOrbCandidateCount}`
          : locateOrbEnabled
            ? 'orb queued'
            : 'orb off',
        Number(liveLocateProgress?.panorama_candidates || 0) > 0
          ? `panos ${Number(liveLocateProgress?.panorama_candidates || 0)}`
          : '',
        Number(liveLocateProgress?.matches || 0) > 0
          ? `families ${Number(liveLocateProgress?.matches || 0)}`
          : ''
      ].filter(Boolean)
    : [
        `target ${String(locateBattlefrontSequence?.targetLabel || '').trim() || 'pending lock'}`,
        `lat ${formatCoordinate(locateBattlefrontSequence?.lat)}`,
        `lon ${formatCoordinate(locateBattlefrontSequence?.lon)}`,
        `score ${formatMetric(locateBattlefrontSequence?.familyScore)}`
      ];
  const locateBattlefrontPreviewLabel = locateBattlefrontSearching ? 'Reference image' : 'Recovered target';
  const locateBattlefrontPreviewPath =
    locateBattlefrontSearching
      ? String(retrievalFilePreviewUrl || '').trim()
      : String(locateBattlefrontSequence?.previewPath || '').trim();
  const locateBattlefrontProgressStages = locateBattlefrontSearching
    ? (locatePipeline.stages || []).map((stage) => ({
        key: stage.key,
        label: stage.title,
        status:
          String(stage.status || '').toLowerCase() === 'completed'
            ? 'done'
            : String(stage.status || '').toLowerCase() === 'running'
              ? 'active'
              : String(stage.status || '').toLowerCase() === 'failed'
                ? 'failed'
                : ''
      }))
    : locateBattlefrontRevealStages.map((stage, index) => ({
        key: stage.key,
        label: stage.label,
        status:
          locateBattlefrontStepIndex > index + 1
            ? 'done'
            : locateBattlefrontStepIndex === index + 1
              ? 'active'
              : ''
      }));

  return (
    <div
      className={`battlefrontReveal battlefrontReveal-${locateBattlefrontSequence.phase}${
        locateBattlefrontSequence.isZooming ? ' battlefrontReveal-zooming' : ''
      }`}
    >
      <div className="battlefrontRevealShade" />
      <div className="battlefrontRevealStars" />
      <div
        key={`battlefront-grid-${locateBattlefrontSequence.phase}-${Number(locateBattlefrontSequence.stepIndex || 0)}`}
        className="battlefrontRevealGridField"
        style={{
          '--battlefront-line-duration': `${Math.max(0, Number(locateBattlefrontSequence.lineDurationMs || 0))}ms`,
          '--battlefront-line-horizontal-duration': `${Math.max(
            0,
            Math.round(Number(locateBattlefrontSequence.lineDurationMs || 0) * 0.56)
          )}ms`,
          '--battlefront-line-vertical-delay': `${Math.max(
            0,
            Math.round(Number(locateBattlefrontSequence.lineDurationMs || 0) * 0.56)
          )}ms`,
          '--battlefront-line-vertical-duration': `${Math.max(
            0,
            Number(locateBattlefrontSequence.lineDurationMs || 0) -
              Math.round(Number(locateBattlefrontSequence.lineDurationMs || 0) * 0.56)
          )}ms`,
          '--battlefront-zoom-duration': `${Math.max(
            0,
            Number(locateBattlefrontSequence.zoomDurationMs || 0)
          )}ms`
        }}
      >
        <div className="battlefrontRevealGridGlow" />
        <div className="battlefrontRevealGridPlane battlefrontRevealGridPlane-major" />
        <div className="battlefrontRevealGridPlane battlefrontRevealGridPlane-minor" />
        <div className="battlefrontRevealGridWindow" />
        <div className="battlefrontRevealGridEdge battlefrontRevealGridEdge-top" />
        <div className="battlefrontRevealGridEdge battlefrontRevealGridEdge-bottom" />
        <div className="battlefrontRevealGridEdge battlefrontRevealGridEdge-left" />
        <div className="battlefrontRevealGridEdge battlefrontRevealGridEdge-right" />
      </div>
      <div className="battlefrontRevealReticle" />
      <div className="battlefrontRevealOrbit battlefrontRevealOrbit-a" />
      <div className="battlefrontRevealOrbit battlefrontRevealOrbit-b" />
      <div className="battlefrontRevealHud">
        <div className="battlefrontRevealTitle">{locateBattlefrontHudTitle}</div>
        <div className="battlefrontRevealSubtitle">{locateBattlefrontHudSubtitle}</div>
        <div className="battlefrontRevealProgress">
          {locateBattlefrontProgressStages.map((stage) => (
            <div
              key={stage.key}
              className={`battlefrontRevealProgressItem ${stage.status}`}
            >
              <span className="battlefrontRevealProgressDot" />
              <span>{stage.label}</span>
            </div>
          ))}
        </div>
        <div className="battlefrontRevealMeta">
          {locateBattlefrontMetaItems.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      </div>
      <div className="battlefrontRevealPreviewCard">
        <div className="battlefrontRevealPreviewLabel">{locateBattlefrontPreviewLabel}</div>
        {locateBattlefrontPreviewPath ? (
          <img
            src={locateBattlefrontPreviewPath}
            alt={
              locateBattlefrontSearching
                ? 'Locate reference image'
                : `Locate target ${locateBattlefrontSequence.targetLabel}`
            }
          />
        ) : (
          <div className="retrievalItemPlaceholder">
            {locateBattlefrontSearching ? 'Waiting for a landing lock' : 'No local image'}
            <span>
              {locateBattlefrontSearching
                ? String(liveLocateProgress?.message || 'Locate pipeline is still resolving the match.')
                : locateBattlefrontSequence.targetLabel}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
