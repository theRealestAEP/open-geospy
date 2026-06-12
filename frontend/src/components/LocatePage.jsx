import { formatMetric } from '../lib/format';
import { retrievalControlHelp } from '../lib/retrievalControls';

/**
 * Locate page card: reference image + basic actions, the collapsed
 * "Advanced settings" tuning panel, the locator pipeline card, and the
 * Families / ORB fingerprints result tabs.
 */
export default function LocatePage({
  locate,
  onRetrievalFileChange,
  retrievalMinSimilarity,
  setRetrievalMinSimilarity,
  locateEmbeddingBase,
  setLocateEmbeddingBase,
  retrievalEmbeddingBaseOptions
}) {
  const {
    locateTopK,
    setLocateTopK,
    locateBattlefrontModeEnabled,
    setLocateBattlefrontModeEnabled,
    locateOrbEnabled,
    setLocateOrbEnabled,
    locateOrbTopN,
    setLocateOrbTopN,
    locateOrbWeight,
    setLocateOrbWeight,
    locateOrbFeatureCount,
    setLocateOrbFeatureCount,
    locateOrbRansacTopK,
    setLocateOrbRansacTopK,
    locateOrbIgnoreBottomRatio,
    setLocateOrbIgnoreBottomRatio,
    locateSam2MaskCars,
    setLocateSam2MaskCars,
    locateSam2MaskTrees,
    setLocateSam2MaskTrees,
    locateOrbPopupEnabled,
    setLocateOrbPopupEnabled,
    locateStatus,
    locateBusy,
    locateResults,
    topLocateFamilyId,
    locateViewTab,
    setLocateViewTab,
    locatePipeline,
    locateOrbComparisons,
    openOrbPopup,
    locateBattlefrontSequenceActive,
    locateActionLocked,
    runImageLocate,
    clearLocate,
    focusRetrievalResult,
    displayedLocateOrbStats,
    hasLocateOrbComparisons,
    locateOrbMaskPercent,
    locateSam2CarsEnabled,
    locateSam2TreesEnabled,
    locateSam2VehicleBoxes,
    locateSam2TreeBoxes,
    locateOrbResolvedFeatureCount
  } = locate;

  return (
    <section className="card">
      <h2>Locate image</h2>
      <p className="hint">
        Uses the selected locate model, optionally reranks top vector hits with ORB, then clusters panorama families.
      </p>
      <label title={retrievalControlHelp.reference_image}>
        Reference image
        <input
          type="file"
          accept="image/*"
          onChange={(e) => onRetrievalFileChange(e.target.files?.[0] || null)}
        />
      </label>
      <details className="locateTunePanel">
        <summary>Advanced settings</summary>
        <div className="grid2">
          <label title={retrievalControlHelp.embedding_base}>
            Locate model
            <select
              value={locateEmbeddingBase}
              onChange={(e) => setLocateEmbeddingBase(e.target.value)}
              disabled={locateBusy}
            >
              {retrievalEmbeddingBaseOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label title={retrievalControlHelp.locate_top_k}>
            Family Top K
            <input
              type="number"
              min="1"
              max="50"
              value={locateTopK}
              onChange={(e) => setLocateTopK(e.target.value)}
            />
          </label>
          <label title={retrievalControlHelp.min_similarity}>
            Min similarity
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={retrievalMinSimilarity}
              onChange={(e) => setRetrievalMinSimilarity(e.target.value)}
              placeholder="optional"
            />
          </label>
        </div>
        <div className="grid2">
          <label title={retrievalControlHelp.locate_orb_enabled}>
            ORB rerank
            <input
              type="checkbox"
              checked={locateOrbEnabled}
              onChange={(e) => setLocateOrbEnabled(e.target.checked)}
            />
          </label>
          <label title={retrievalControlHelp.locate_orb_top_n}>
            ORB Top N
            <input
              type="number"
              min="1"
              max="5000"
              value={locateOrbTopN}
              onChange={(e) => setLocateOrbTopN(e.target.value)}
            />
          </label>
          <label title={retrievalControlHelp.locate_orb_weight}>
            ORB Weight
            <input
              type="number"
              min="0"
              max="5"
              step="0.05"
              value={locateOrbWeight}
              onChange={(e) => setLocateOrbWeight(e.target.value)}
            />
          </label>
          <label title={retrievalControlHelp.locate_orb_feature_count}>
            ORB feature count
            <input
              type="number"
              min="100"
              max="2000"
              step="50"
              value={locateOrbFeatureCount}
              onChange={(e) => setLocateOrbFeatureCount(e.target.value)}
            />
          </label>
          <label title={retrievalControlHelp.locate_orb_ransac_top_k}>
            RANSAC Top K
            <input
              type="number"
              min="0"
              max={String(Math.max(1, Number(locateOrbTopN) || 100))}
              value={locateOrbRansacTopK}
              onChange={(e) => setLocateOrbRansacTopK(e.target.value)}
            />
          </label>
          <label title={retrievalControlHelp.locate_orb_ignore_bottom_ratio}>
            Ignore lower frame ratio
            <input
              type="number"
              min="0"
              max="0.6"
              step="0.02"
              value={locateOrbIgnoreBottomRatio}
              onChange={(e) => setLocateOrbIgnoreBottomRatio(e.target.value)}
            />
          </label>
        </div>
        <label className="checkboxLabel locateToggleCard" title={retrievalControlHelp.locate_sam2_mask_cars}>
          <input
            type="checkbox"
            checked={locateSam2MaskCars}
            onChange={(e) => setLocateSam2MaskCars(e.target.checked)}
          />
          Mask cars with local SAM 2
        </label>
        <label className="checkboxLabel locateToggleCard" title={retrievalControlHelp.locate_sam2_mask_trees}>
          <input
            type="checkbox"
            checked={locateSam2MaskTrees}
            onChange={(e) => setLocateSam2MaskTrees(e.target.checked)}
          />
          Mask trees and vegetation
        </label>
        <div className="hint">
          ORB settings apply when ORB rerank is enabled. SAM 2 masking affects the query image plus the top ORB review candidates only, so the initial vector search still uses the original image.
        </div>
        <div className="grid2">
          <label className="checkboxLabel locateToggleCard" title={retrievalControlHelp.locate_battlefront_mode}>
            <input
              type="checkbox"
              checked={locateBattlefrontModeEnabled}
              onChange={(e) => setLocateBattlefrontModeEnabled(e.target.checked)}
            />
            Battlefront reveal mode
          </label>
          <label className="checkboxLabel locateToggleCard" title={retrievalControlHelp.locate_orb_popup}>
            <input
              type="checkbox"
              checked={locateOrbPopupEnabled}
              onChange={(e) => setLocateOrbPopupEnabled(e.target.checked)}
            />
            Show ORB popup
          </label>
          {hasLocateOrbComparisons ? (
            <button
              className="ghost compact locateInlineAction"
              onClick={openOrbPopup}
            >
              Open ORB popup
            </button>
          ) : null}
        </div>
      </details>
      <div className="buttonRow">
        <button onClick={runImageLocate} disabled={locateActionLocked}>
          {locateBusy ? 'Locating...' : locateBattlefrontSequenceActive ? 'Battlefront running...' : 'Locate by image'}
        </button>
        <button
          className="ghost"
          onClick={clearLocate}
          disabled={locateBusy}
        >
          Clear
        </button>
      </div>
      <div className="locatorPipelineCard">
        <h3>Locator pipeline</h3>
        <div className="pipelineStageList">
          {(locatePipeline.stages || []).map((stage) => (
            <div
              key={stage.key}
              className={`pipelineStage pipelineStage-${String(stage.status || 'pending').toLowerCase()}`}
            >
              <span className="pipelineDot" />
              <div className="pipelineStageBody">
                <div>
                  <div className="pipelineStageTitle">{stage.title}</div>
                  {stage.detail ? <div className="hint">{stage.detail}</div> : null}
                </div>
                <div className="pipelineStageStatus">{stage.status || 'pending'}</div>
              </div>
            </div>
          ))}
          {(!locatePipeline.stages || locatePipeline.stages.length === 0) ? (
            <p className="hint">No locate request yet.</p>
          ) : null}
        </div>
      </div>
      <p className="status">{locateStatus}</p>
      <div className="locateSubtabBar">
        <button
          className={locateViewTab === 'families' ? '' : 'ghost'}
          onClick={() => setLocateViewTab('families')}
        >
          Families
        </button>
        <button
          className={locateViewTab === 'orb' ? '' : 'ghost'}
          onClick={() => setLocateViewTab('orb')}
          disabled={!locateOrbEnabled && !hasLocateOrbComparisons}
        >
          ORB fingerprints{hasLocateOrbComparisons ? ` (${locateOrbComparisons.length})` : ''}
        </button>
      </div>
      {locateViewTab === 'families' ? (
        <div className="retrievalGrid">
          {locateResults.map((result) => (
            <button
              key={result.family_id || result.panorama_id}
              className={`retrievalItem ${
                String(result.family_id || '') === String(topLocateFamilyId || '')
                  ? 'retrievalItemBest'
                  : ''
              }`}
              onClick={() => focusRetrievalResult(result)}
              title={`family score ${(Number(result.family_score || 0)).toFixed(3)}`}
            >
              {String(result.web_path || '').trim() ? (
                <img src={result.web_path} alt={`family ${result.family_id || result.panorama_id}`} />
              ) : (
                <div className="retrievalItemPlaceholder">
                  No local image
                  <span>
                    {Number.isFinite(Number(result.lat ?? result.family_center_lat)) &&
                    Number.isFinite(Number(result.lon ?? result.family_center_lon))
                      ? `${Number(result.lat ?? result.family_center_lat).toFixed(6)}, ${Number(result.lon ?? result.family_center_lon).toFixed(6)}`
                      : 'Coordinates unavailable'}
                  </span>
                </div>
              )}
              <span>family score {(Number(result.family_score || 0)).toFixed(3)}</span>
              <span>
                {Number(result.family_panorama_count || 0)} panoramas | {Number(result.family_capture_hits || 0)} capture hits
              </span>
              {String(result.family_id || '') === String(topLocateFamilyId || '') ? (
                <span className="bestMatchTag">Top family</span>
              ) : null}
            </button>
          ))}
        </div>
      ) : (
        <div className="orbLabPanel">
          <div className="orbLabHeader">
            <div>
              <div className="orbLabEyebrow">ORB fingerprint comparisons</div>
              <div className="hint">
                Match overlays for the top reranked capture images from this locate run. Click a card to focus that panorama on the map.
              </div>
              {locateOrbMaskPercent > 0 ? (
                <div className="hint orbMaskHint">
                  Lower {locateOrbMaskPercent}% of each frame is ignored during ORB extraction to reduce cars and road clutter.
                </div>
              ) : null}
            </div>
            {displayedLocateOrbStats ? (
              <div className="orbChipRow">
                <span className="orbStatChip">scored {Number(displayedLocateOrbStats.candidates_scored || 0)}</span>
                <span className="orbStatChip">RANSAC {Number(displayedLocateOrbStats.ransac_checked || 0)}</span>
                <span className="orbStatChip">query kp {Number(displayedLocateOrbStats.query_keypoints || 0)}</span>
                <span className="orbStatChip">features {locateOrbResolvedFeatureCount}</span>
                {locateOrbMaskPercent > 0 ? (
                  <span className="orbStatChip">lower mask {locateOrbMaskPercent}%</span>
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
            ) : null}
          </div>
          {String(displayedLocateOrbStats?.query_fingerprint_data_url || '').trim() ? (
            <div className="orbQueryCard">
              <div>
                <div className="orbLabEyebrow">Query fingerprint</div>
                <div className="hint">
                  ORB keypoints extracted from the uploaded reference image.
                </div>
              </div>
              <img
                src={displayedLocateOrbStats.query_fingerprint_data_url}
                alt="ORB query fingerprint"
              />
            </div>
          ) : null}
          {hasLocateOrbComparisons ? (
            <div className="orbComparisonGrid">
              {locateOrbComparisons.map((comparison) => (
                <button
                  key={comparison.capture_id}
                  className="orbComparisonCard"
                  onClick={() => focusRetrievalResult(comparison)}
                  title={`ORB ${formatMetric(comparison.orb_score)}`}
                >
                  {String(comparison.visualization_data_url || '').trim() ? (
                    <img
                      src={comparison.visualization_data_url}
                      alt={`ORB comparison ${comparison.capture_id}`}
                    />
                  ) : (
                    <div className="retrievalItemPlaceholder">
                      No comparison visualization
                      <span>capture {comparison.capture_id}</span>
                    </div>
                  )}
                  <div className="orbChipRow">
                    <span className="orbStatChip">before #{comparison.rank_before}</span>
                    <span className="orbStatChip">after #{comparison.rank_after}</span>
                    <span className="orbStatChip">ORB {formatMetric(comparison.orb_score)}</span>
                  </div>
                  <span>
                    vector {formatMetric(comparison.score_before)} → {formatMetric(comparison.score_after)}
                  </span>
                  <span>
                    {Number(comparison.orb_good_matches || 0)} good matches | {Number(comparison.orb_inliers || 0)} inliers
                  </span>
                  <span>
                    visualized {Number(comparison.visual_match_count || 0)} match lines
                  </span>
                  <span>
                    mean match distance {formatMetric(comparison.orb_mean_match_distance || 0, 1)}
                  </span>
                  <span>
                    query {Number(comparison.query_keypoints || 0)} kp | candidate {Number(comparison.candidate_keypoints || 0)} kp
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="orbEmptyState">
              <div className="orbLabEyebrow">No fingerprint comparisons yet</div>
              <div className="hint">
                Turn on ORB rerank and run Locate to open the comparison lab.
                {displayedLocateOrbStats?.reason ? ` Current state: ${displayedLocateOrbStats.reason}.` : ''}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
