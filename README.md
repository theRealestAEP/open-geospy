# Open Geospy

Open Geospy is an open-source street-view collection + image search stack. Inspired by geospy.ai

Current public dataset release is San Francisco-focused and includes multi-million capture embeddings for search.

## Quick Start

### 1) Clone and install

```bash
git clone https://github.com/theRealestAEP/open-geospy.git
cd open-geospy

cp env.example .env
pip install -r requirements.txt
playwright install chromium
```

### 2) Start Postgres

```bash
docker compose up -d postgres
```

### 3) Load the pre-indexed DB snapshot (recommended)

`data-snapshot-v1` is uploaded as multipart assets plus a `.parts.txt` manifest.

```bash
./scripts/install_from_pgvector_snapshot.sh \
  --parts-base-url https://github.com/theRealestAEP/open-geospy/releases/download/data-snapshot-v1 \
  --parts-manifest-url https://github.com/theRealestAEP/open-geospy/releases/download/data-snapshot-v1/<snapshot-file>.parts.txt
```

Find the exact manifest name:

```bash
gh release view data-snapshot-v1 --repo theRealestAEP/open-geospy --json assets --jq '.assets[].name' | grep '\.parts\.txt$'
```

### 4) Run backend

```bash
python -m backend.app.main
```

Backend: `http://localhost:8000`

### 5) Run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://127.0.0.1:5173`

## Optional: Run Retrieval on Local LanceDB

This keeps Postgres for operational metadata (`panoramas`, `captures`, scan state) and
uses LanceDB for vector search.

1) Sync vectors from pgvector into a local Lance table:

```bash
python scripts/sync_pgvector_to_lancedb.py \
  --lance-uri .lancedb \
  --table capture_embeddings \
  --mode overwrite \
  --create-index
```

2) Switch backend to LanceDB:

```bash
export GEOSPY_VECTOR_BACKEND=lancedb
export GEOSPY_LANCEDB_URI=.lancedb
export GEOSPY_LANCEDB_TABLE=capture_embeddings
python -m backend.app.main
```

Publish local LanceDB snapshot to a GitHub release:

```bash
./scripts/export_lancedb_snapshot.sh --release-tag data-snapshot-v1-lancedb
```

Notes:
- Lance mode is currently search-only for embeddings (`/api/retrieval/index-missing` disabled).
- Switch back at any time with `GEOSPY_VECTOR_BACKEND=postgres`.

## Search Flow

1. Open the `Search` page.
2. Upload a query image.
3. Click `Search by image`.
4. Review ranked matches; best match is auto-focused on the map.
5. If a result has no local image, click it to drop a map dot and open Street View.

## What Is In The Snapshot DB

Snapshot: `data-snapshot-v1` (profile measured on 2026-03-04)

- Region: San Francisco captures.
- `captures`: `4,353,633` rows.
- `panoramas`: `27,161` rows.
- `capture_embeddings`: `8,707,266` rows.
- Embedding models:
  - `ViT-B-32:open_clip` (`512d`) -> `4,353,633` vectors.
  - `ViT-B-16:open_clip_place` (`512d`) -> `4,353,633` vectors.
- Capture angle pattern:
  - headings: `0..345` in 15-degree steps (`24` slices)
  - pitch levels: `45`, `60`, `75`, `90`, `105`
- Captures per panorama:
  - average: `160.29`
  - min/max: `4` / `1456`

Not included in snapshot SQL:

- local image files under `./captures`

## Snapshot Commands

Create and upload a snapshot release asset:

```bash
./scripts/export_pgvector_snapshot.sh --release-tag data-snapshot-v1
```

Restore from a single-file snapshot URL:

```bash
./scripts/install_from_pgvector_snapshot.sh \
  --snapshot-url https://github.com/<org>/<repo>/releases/download/data-snapshot-v1/<snapshot-file>.sql.gz
```

Restore from a local file:

```bash
./scripts/install_from_pgvector_snapshot.sh --snapshot-file backups/<snapshot-file>.sql.gz
```

Fast DB rebuild helper:

```bash
./scripts/rebuild_db.sh
```

## Optional: Add More Data

After startup, use the Scan UI (`scan` / `enrich` / `fill`) to collect more captures.

Index embeddings for new captures:

```bash
python -m utils.index_capture_embeddings --batch-size 64
```

For higher-throughput indexing via Modal:

```bash
python -m utils.index_capture_embeddings \
  --mode modal \
  --batch-size 4096 \
  --modal-workers 64 \
  --modal-worker-batch-size 64
```

## Key Scripts

- `scripts/rebuild_db.sh` - reset/rebuild DB.
- `scripts/export_pgvector_snapshot.sh` - export/upload snapshot.
- `scripts/export_lancedb_snapshot.sh` - archive/upload `.lancedb` snapshot.
- `scripts/install_from_pgvector_snapshot.sh` - restore snapshot (single file or multipart).
- `scripts/sync_pgvector_to_lancedb.py` - copy `capture_embeddings` into LanceDB.
- `utils/index_capture_embeddings.py` - (re)index embeddings.
