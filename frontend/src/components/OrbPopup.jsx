import { formatMetric } from '../lib/format';
import {
  formatLocateProgressStageBadge,
  getOrbPopupTimelineState,
  mapLocateProgressPhaseToPreviewStage,
  orbPopupMoments,
  orbPopupPreviewPoints,
  orbPopupPreviewStages
} from '../lib/locatePipeline';

/**
 * The ORB Fingerprint Lab popup: live processing preview over the reference
 * image, the stage card with timeline phases, and the results/error states.
 */
export default function OrbPopup({ locate, retrievalFilePreviewUrl, locateModelLabel }) {
  const {
    locateBusy,
    locateStatus,
    locateOrbEnabled,
    locateOrbTopN,
    locateOrbWeight,
    locateOrbPopupPhase,
    locateOrbPopupMoment,
    dismissOrbPopup,
    openOrbTabFromPopup,
    focusRetrievalResult,
    locateOrbComparisons,
    hasLocateOrbComparisons,
    displayedLocateOrbStats,
    liveLocateProgress,
    liveLocateOrbStats,
    liveLocateComparison,
    liveLocatePhase,
    locateOrbMaskPercent,
    locateSam2Enabled,
    locateSam2CarsEnabled,
    locateSam2TreesEnabled,
    locateSam2VehicleBoxes,
    locateSam2TreeBoxes,
    locateOrbProcessedCandidates,
    locateOrbCandidateCount,
    locateOrbResolvedFeatureCount
  } = locate;

  const fallbackLocateOrbPopupMessage = orbPopupMoments[locateOrbPopupMoment % orbPopupMoments.length];
  const locateOrbPopupMessage =
    locateOrbPopupPhase === 'processing'
      ? String(liveLocateProgress?.message || '').trim() || fallbackLocateOrbPopupMessage
      : fallbackLocateOrbPopupMessage;
  const locateOrbPreviewStage =
    locateOrbPopupPhase === 'processing'
      ? mapLocateProgressPhaseToPreviewStage(liveLocatePhase, liveLocateOrbStats)
      : orbPopupPreviewStages[locateOrbPopupMoment % orbPopupPreviewStages.length];
  const locateOrbPopupTitle =
    locateOrbPopupPhase === 'processing'
      ? 'Comparing local feature fingerprints'
      : locateOrbPopupPhase === 'error'
        ? 'Fingerprint pass needs attention'
        : 'Fingerprint pass complete';
  const locateOrbPopupDescription =
    locateOrbPopupPhase === 'processing'
      ? locateOrbPopupMessage
      : locateOrbPopupPhase === 'error'
        ? locateStatus || 'The ORB fingerprint pass did not complete successfully.'
        : hasLocateOrbComparisons
          ? 'The strongest overlays are ready. You can review them here or jump back into the ORB tab.'
          : 'ORB finished, but this run did not produce visualization overlays.';
  const locateOrbPopupStageBadge =
    locateOrbPopupPhase === 'processing'
      ? formatLocateProgressStageBadge(
          liveLocatePhase,
          Boolean(liveLocateOrbStats?.enabled ?? locateOrbEnabled)
        )
      : locateOrbPopupPhase === 'error'
        ? 'Needs attention'
        : 'Completed';
  const locateOrbPopupStageSummary =
    locateOrbPopupPhase === 'processing'
      ? liveLocatePhase === 'vector_search'
        ? 'Running the same raw vector retrieval step as Search before any local-feature rerank is applied.'
        : liveLocatePhase === 'orb_rerank'
          ? locateOrbCandidateCount > 0
            ? `Compared ${locateOrbProcessedCandidates} of ${locateOrbCandidateCount} candidates using up to ${locateOrbResolvedFeatureCount} ORB features per image.`
            : `Extracting up to ${locateOrbResolvedFeatureCount} ORB keypoints per image before matching and RANSAC checks.`
          : liveLocatePhase === 'panorama_rerank'
            ? 'The backend finished ORB scoring and is collapsing capture hits into panorama candidates.'
            : liveLocatePhase === 'family_rank'
              ? 'The backend is clustering panorama candidates into nearby location families.'
              : 'Preparing the query image and ORB pipeline.'
      : locateOrbPopupPhase === 'error'
        ? 'The comparison stage stopped before it could finish.'
        : `Scored ${Number(displayedLocateOrbStats?.candidates_scored || 0)} candidates and kept ${locateOrbComparisons.length} visual overlays for review.`;
  const locateOrbPopupCloseLabel =
    locateOrbPopupPhase === 'processing'
      ? 'Hide'
      : locateOrbPopupPhase === 'error'
        ? 'Dismiss'
        : 'Done';

  return (
    <div
      className="orbPopupBackdrop"
      onClick={() => {
        if (locateBusy && locateOrbPopupPhase === 'processing') return;
        dismissOrbPopup();
      }}
    >
      <div
        className={`orbPopup orbPopup-${locateOrbPopupPhase}`}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="orbPopupHeader">
          <div>
            <div className="orbLabEyebrow">ORB Fingerprint Lab</div>
            <h2>{locateOrbPopupTitle}</h2>
            <p className="hint">{locateOrbPopupDescription}</p>
          </div>
          <button className="ghost" onClick={dismissOrbPopup}>
            {locateOrbPopupCloseLabel}
          </button>
        </div>
        <div className="orbPopupHero">
          <div className="orbPopupQuery">
            <div className="orbLabEyebrow">Reference image</div>
            {retrievalFilePreviewUrl ? (
              locateOrbPopupPhase === 'processing' ? (
                <div className={`orbPopupLivePreview orbPopupLivePreview-${locateOrbPreviewStage}`}>
                  <img
                    className="orbPopupLivePreviewImage"
                    src={retrievalFilePreviewUrl}
                    alt="Reference query"
                  />
                  <div className="orbPopupLivePreviewShade" />
                  <div className="orbPopupLivePreviewGrid" />
                  <div className="orbPopupLivePreviewSweep" />
                  <div className="orbPopupLivePreviewPulse" />
                  <div className="orbPopupLivePreviewNodes" aria-hidden="true">
                    {orbPopupPreviewPoints.map((point, index) => (
                      <span
                        key={`orb-preview-point-${index}`}
                        style={{
                          '--x': `${point.x}%`,
                          '--y': `${point.y}%`,
                          '--delay': `${point.delay}s`
                        }}
                      />
                    ))}
                  </div>
                  {locateSam2Enabled ? (
                    <>
                      {locateSam2CarsEnabled ? (
                        <>
                          <div className="orbPopupLivePreviewVehicle orbPopupLivePreviewVehicle-a" />
                          <div className="orbPopupLivePreviewVehicle orbPopupLivePreviewVehicle-b" />
                        </>
                      ) : null}
                      <div className="orbPopupLivePreviewBadge">
                        {locateSam2CarsEnabled || locateSam2TreesEnabled
                          ? `SAM 2 ${[
                              locateSam2CarsEnabled ? `cars ${locateSam2VehicleBoxes}` : '',
                              locateSam2TreesEnabled ? `trees ${locateSam2TreeBoxes}` : ''
                            ]
                              .filter(Boolean)
                              .join(' | ')}`
                          : 'SAM 2 masking queued'}
                      </div>
                    </>
                  ) : null}
                  {locateOrbMaskPercent > 0 ? (
                    <div
                      className="orbPopupLivePreviewMask"
                      style={{ height: `${locateOrbMaskPercent}%` }}
                    >
                      <span>ignore lower {locateOrbMaskPercent}%</span>
                    </div>
                  ) : null}
                  <div className="orbPopupLivePreviewCaption">
                    {locateOrbPopupMessage}
                  </div>
                </div>
              ) : (
                <img src={retrievalFilePreviewUrl} alt="Reference query" />
              )
            ) : (
              <div className="retrievalItemPlaceholder">
                Add a query image to preview it here.
              </div>
            )}
            {String(displayedLocateOrbStats?.query_fingerprint_data_url || '').trim() ? (
              <img
                className="orbPopupFingerprint"
                src={displayedLocateOrbStats.query_fingerprint_data_url}
                alt="Query ORB fingerprint"
              />
            ) : null}
          </div>
          <div className="orbPopupStageCard">
            <div className="orbScannerHalo" />
            <div className="orbScannerGrid" />
            <div className={`orbPopupStageBadge orbPopupStageBadge-${locateOrbPopupPhase}`}>
              {locateOrbPopupStageBadge}
            </div>
            <div className="orbChipRow">
              <span className="orbStatChip">
                model {locateModelLabel}
              </span>
              <span className="orbStatChip">
                ORB top {Math.max(1, Number(displayedLocateOrbStats?.top_n || locateOrbTopN) || 100)}
              </span>
              <span className="orbStatChip">
                weight {formatMetric(displayedLocateOrbStats?.weight || locateOrbWeight, 2)}
              </span>
              <span className="orbStatChip">
                features {locateOrbResolvedFeatureCount}
              </span>
              {locateOrbMaskPercent > 0 ? (
                <span className="orbStatChip">
                  lower mask {locateOrbMaskPercent}%
                </span>
              ) : null}
              {locateOrbPopupPhase === 'processing' && locateOrbCandidateCount > 0 ? (
                <span className="orbStatChip">
                  done {locateOrbProcessedCandidates}/{locateOrbCandidateCount}
                </span>
              ) : null}
              {locateSam2CarsEnabled ? (
                <span className="orbStatChip">
                  SAM 2 cars {locateSam2VehicleBoxes > 0 ? locateSam2VehicleBoxes : 0}
                </span>
              ) : null}
              {locateSam2TreesEnabled ? (
                <span className="orbStatChip">
                  trees {locateSam2TreeBoxes > 0 ? locateSam2TreeBoxes : 0}
                </span>
              ) : null}
            </div>
            <div className="orbPopupMoment">
              {locateOrbPopupPhase === 'processing'
                ? locateOrbPopupMessage
                : locateOrbPopupTitle}
            </div>
            <div className="orbPopupStageSummary">{locateOrbPopupStageSummary}</div>
            {locateOrbPopupPhase === 'processing' &&
            liveLocateComparison &&
            String(liveLocateComparison.web_path || '').trim() ? (
              <button
                className="orbPopupCurrentCandidate"
                onClick={() => focusRetrievalResult(liveLocateComparison)}
              >
                <div className="orbLabEyebrow">Current candidate</div>
                <img
                  src={liveLocateComparison.web_path}
                  alt={`Current candidate ${liveLocateComparison.capture_id}`}
                />
                <div className="orbChipRow">
                  <span className="orbStatChip">capture {Number(liveLocateComparison.capture_id || 0)}</span>
                  <span className="orbStatChip">before #{Number(liveLocateComparison.rank_before || 0)}</span>
                </div>
                <span>
                  vector {formatMetric(liveLocateComparison.score_before)} | similarity {formatMetric(liveLocateComparison.similarity)}
                </span>
              </button>
            ) : null}
            <div className="orbPopupTimeline">
              {orbPopupMoments.map((moment, index) => (
                <div
                  key={moment}
                  className={`orbPopupTimelineItem ${getOrbPopupTimelineState(
                    index,
                    locateOrbPopupPhase,
                    locateOrbPopupMoment
                  )}`}
                >
                  <span className="pipelineDot" />
                  <span>{moment}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        {locateOrbPopupPhase === 'processing' && liveLocateComparison ? (
          <div className="orbPopupResults orbPopupLiveResults">
            <div className="orbPopupResultsHeader">
              <div>
                <div className="orbLabEyebrow">Live backend comparison</div>
                <div className="hint">
                  Candidate {Math.max(1, locateOrbProcessedCandidates)} of {Math.max(1, locateOrbCandidateCount || Number(displayedLocateOrbStats?.top_n || locateOrbTopN) || 1)}
                </div>
              </div>
              <div className="orbChipRow">
                <span className="orbStatChip">before #{Number(liveLocateComparison.rank_before || 0)}</span>
                <span className="orbStatChip">ORB {formatMetric(liveLocateComparison.orb_score)}</span>
                <span className="orbStatChip">{Number(liveLocateComparison.orb_good_matches || 0)} matches</span>
              </div>
            </div>
            <div className="orbComparisonGrid orbComparisonGridPopup orbPopupLiveComparisonGrid">
              <button
                key={`live-${liveLocateComparison.capture_id}`}
                className="orbComparisonCard"
                onClick={() => focusRetrievalResult(liveLocateComparison)}
              >
                {String(liveLocateComparison.visualization_data_url || '').trim() ? (
                  <img
                    src={liveLocateComparison.visualization_data_url}
                    alt={`Live ORB comparison ${liveLocateComparison.capture_id}`}
                  />
                ) : (
                  <div className="retrievalItemPlaceholder">
                    Live comparison still rendering
                  </div>
                )}
                <div className="orbChipRow">
                  <span className="orbStatChip">Match overlay</span>
                  <span className="orbStatChip">after #{Number(liveLocateComparison.rank_after || 0)}</span>
                </div>
                <span>
                  {Number(liveLocateComparison.orb_good_matches || 0)} good matches | {Number(liveLocateComparison.orb_inliers || 0)} inliers
                </span>
                <span>
                  visualized {Number(liveLocateComparison.visual_match_count || 0)} match lines
                </span>
                <span>
                  mean match distance {formatMetric(liveLocateComparison.orb_mean_match_distance || 0, 1)}
                </span>
              </button>
            </div>
          </div>
        ) : null}
        {locateOrbPopupPhase === 'results' ? (
          hasLocateOrbComparisons ? (
            <div className="orbPopupResults">
              <div className="orbPopupResultsHeader">
                <div className="orbChipRow">
                  <span className="orbStatChip">scored {Number(displayedLocateOrbStats?.candidates_scored || 0)}</span>
                  <span className="orbStatChip">RANSAC {Number(displayedLocateOrbStats?.ransac_checked || 0)}</span>
                  <span className="orbStatChip">query kp {Number(displayedLocateOrbStats?.query_keypoints || 0)}</span>
                  <span className="orbStatChip">features {locateOrbResolvedFeatureCount}</span>
                </div>
                <button className="ghost" onClick={openOrbTabFromPopup}>
                  Open ORB tab
                </button>
              </div>
              <div className="orbComparisonGrid orbComparisonGridPopup">
                {locateOrbComparisons.slice(0, 4).map((comparison) => (
                  <button
                    key={`popup-${comparison.capture_id}`}
                    className="orbComparisonCard"
                    onClick={() => focusRetrievalResult(comparison)}
                  >
                    {String(comparison.visualization_data_url || '').trim() ? (
                      <img
                        src={comparison.visualization_data_url}
                        alt={`ORB popup comparison ${comparison.capture_id}`}
                      />
                    ) : (
                      <div className="retrievalItemPlaceholder">
                        No comparison visualization
                      </div>
                    )}
                    <div className="orbChipRow">
                      <span className="orbStatChip">after #{comparison.rank_after}</span>
                      <span className="orbStatChip">ORB {formatMetric(comparison.orb_score)}</span>
                    </div>
                    <span>
                      {Number(comparison.orb_good_matches || 0)} good matches | {Number(comparison.visual_match_count || 0)} lines shown
                    </span>
                    <span>
                      mean match distance {formatMetric(comparison.orb_mean_match_distance || 0, 1)}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="orbEmptyState">
              <div className="orbLabEyebrow">No ORB visuals this time</div>
              <div className="hint">
                {displayedLocateOrbStats?.reason
                  ? `The ORB stage reported: ${displayedLocateOrbStats.reason}.`
                  : 'This locate run did not produce comparison overlays.'}
              </div>
            </div>
          )
        ) : null}
      </div>
    </div>
  );
}
