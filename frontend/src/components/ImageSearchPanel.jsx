import { retrievalControlHelp } from '../lib/retrievalControls';

/**
 * Search page card: reference image upload, search controls, and the
 * similarity-ranked results grid.
 */
export default function ImageSearchPanel({
  search,
  retrievalStats,
  onRetrievalFileChange,
  searchEmbeddingBase,
  setSearchEmbeddingBase,
  retrievalEmbeddingBaseOptions,
  retrievalMinSimilarity,
  setRetrievalMinSimilarity,
  focusRetrievalResult
}) {
  const {
    retrievalSearchTopK,
    setRetrievalSearchTopK,
    retrievalStatus,
    retrievalBusy,
    retrievalResults,
    topSearchCaptureId,
    runImageSearch,
    clearSearch
  } = search;

  return (
    <section className="card">
      <h2>Image search</h2>
      <p className="hint">
        Indexed {retrievalStats.embedded_captures}/{retrievalStats.total_captures} captures
        {retrievalStats.model_name ? ` (${retrievalStats.model_name}${retrievalStats.model_version ? ` ${retrievalStats.model_version}` : ''})` : ''}
      </p>
      {Array.isArray(retrievalStats.models) && retrievalStats.models.length > 1 ? (
        <p className="hint">
          Models: {retrievalStats.models.map((m) => `${m.model_name} ${m.model_version} ${m.embedded_captures}/${m.total_captures}`).join(' | ')}
        </p>
      ) : null}
      <label title={retrievalControlHelp.reference_image}>
        Reference image
        <input
          type="file"
          accept="image/*"
          onChange={(e) => onRetrievalFileChange(e.target.files?.[0] || null)}
        />
      </label>
      <div className="grid2">
        <label title={retrievalControlHelp.embedding_base}>
          Search model
          <select
            value={searchEmbeddingBase}
            onChange={(e) => setSearchEmbeddingBase(e.target.value)}
            disabled={retrievalBusy}
          >
            {retrievalEmbeddingBaseOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label title={retrievalControlHelp.search_top_k}>
          Search Top K
          <input
            type="number"
            min="1"
            max="200"
            value={retrievalSearchTopK}
            onChange={(e) => setRetrievalSearchTopK(e.target.value)}
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
      <div className="buttonRow">
        <button onClick={runImageSearch} disabled={retrievalBusy}>
          {retrievalBusy ? 'Searching...' : 'Search by image'}
        </button>
        <button className="ghost" onClick={clearSearch} disabled={retrievalBusy}>
          Clear
        </button>
      </div>
      <p className="hint">Best match is auto-focused on the map and highlighted below.</p>
      <p className="status">{retrievalStatus}</p>
      <div className="retrievalGrid">
        {retrievalResults.map((result) => (
          <button
            key={result.capture_id}
            className={`retrievalItem ${
              Number(result.capture_id) === Number(topSearchCaptureId) ? 'retrievalItemBest' : ''
            }`}
            onClick={() => focusRetrievalResult(result)}
            title={`score ${(Number(result.similarity || 0)).toFixed(3)}`}
          >
            {String(result.web_path || '').trim() ? (
              <img src={result.web_path} alt={`match ${result.capture_id}`} />
            ) : (
              <div className="retrievalItemPlaceholder">
                No local image
                <span>
                  {Number.isFinite(Number(result.lat)) && Number.isFinite(Number(result.lon))
                    ? `${Number(result.lat).toFixed(6)}, ${Number(result.lon).toFixed(6)}`
                    : 'Coordinates unavailable'}
                </span>
              </div>
            )}
            <span>score {(Number(result.similarity || 0)).toFixed(3)}</span>
            {Number(result.capture_id) === Number(topSearchCaptureId) ? (
              <span className="bestMatchTag">Best match</span>
            ) : null}
          </button>
        ))}
      </div>
    </section>
  );
}
