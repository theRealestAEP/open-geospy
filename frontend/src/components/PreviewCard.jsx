export default function PreviewCard({ preview, emptyHint }) {
  return (
    <section id="preview" className="card">
      <h2>Preview</h2>
      <div className="coords">{preview.title}</div>
      <div className="previewGrid">
        {preview.captures.length ? (
          preview.captures.map((c) => (
            String(c.src || '').trim() ? (
              <img key={c.id} src={c.src} alt={`heading ${c.heading}`} title={`${c.heading} deg`} />
            ) : (
              <div key={c.id} className="retrievalItemPlaceholder">
                No local image
                <span>heading {Number(c.heading || 0)}</span>
              </div>
            )
          ))
        ) : (
          <p className="hint">{emptyHint}</p>
        )}
      </div>
    </section>
  );
}
