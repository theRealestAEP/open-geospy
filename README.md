# Open Geospy

Open Geo Spy is an open-source street-view collection + image search stack. Inspired by geospy.ai

Current public dataset release is San Francisco-focused and includes multi-million capture embeddings for search.

## Demo

[Watch the demo video](./demo.mp4)

## Quick Start

### 1) Clone and install

```bash
git clone https://github.com/theRealestAEP/open-geospy.git
cd open-geospy

cp env.example .env
pip install -r requirements.txt
playwright install chromium
```

Optional Modal pinning for this repo:

```bash
cp .modal.toml.example .modal.toml
# fill in the Modal token values you want this repo to use
# then set MODAL_PROFILE in .env to the matching section name
```

### 2) Start Postgres (metadata store)

```bash
docker compose up -d postgres
```

### 3) Load LanceDB snapshot (recommended)

Preferred path: restore `.lancedb` directly from the LanceDB release (`data-snapshot-v1-lancedb`).

Find the latest LanceDB manifest:

```bash
gh release view data-snapshot-v1-lancedb --repo theRealestAEP/open-geospy --json assets --jq '.assets[].name' | grep '\.parts\.txt$'
```

Restore the current published snapshot (example manifest from latest upload):

```bash
BASE_URL="https://github.com/theRealestAEP/open-geospy/releases/download/data-snapshot-v1-lancedb"
MANIFEST="geospy_lancedb_snapshot_20260304_205728.tar.gz.parts.txt"
curl -L --fail -o /tmp/lance.parts.txt "${BASE_URL}/${MANIFEST}"
rm -f /tmp/geospy_lancedb_snapshot.tar.gz
while IFS= read -r part; do
  [[ -z "$part" ]] && continue
  curl -L --fail -o "/tmp/${part}" "${BASE_URL}/${part}"
  cat "/tmp/${part}" >> /tmp/geospy_lancedb_snapshot.tar.gz
  rm -f "/tmp/${part}"
done < /tmp/lance.parts.txt
tar -xzf /tmp/geospy_lancedb_snapshot.tar.gz
```

If you do not have a LanceDB snapshot yet, fallback path:

1) Restore pgvector snapshot into Postgres:

```bash
./scripts/install_from_pgvector_snapshot.sh \
  --parts-base-url https://github.com/theRealestAEP/open-geospy/releases/download/data-snapshot-v1 \
  --parts-manifest-url https://github.com/theRealestAEP/open-geospy/releases/download/data-snapshot-v1/<snapshot-file>.parts.txt
```

2) Sync pgvector embeddings into local LanceDB:

```bash
python scripts/sync_pgvector_to_lancedb.py \
  --lance-uri .lancedb \
  --table capture_embeddings \
  --mode overwrite \
  --create-index
```

3) Optional cleanup: remove pgvector rows after sync (keeps metadata tables only):

```bash
docker exec -i geospy-postgres psql -U geospy -d geospy -v ON_ERROR_STOP=1 <<'SQL'
TRUNCATE TABLE capture_embeddings;
VACUUM (ANALYZE) capture_embeddings;
SQL
```

### 4) Run backend with LanceDB vector search

```bash
export GEOSPY_VECTOR_BACKEND=lancedb
export GEOSPY_LANCEDB_URI=.lancedb
export GEOSPY_LANCEDB_TABLE=capture_embeddings
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

## Modal Config

GeoSpy now auto-loads the project `.env` and will automatically use a repo-local
`.modal.toml` when present by setting `MODAL_CONFIG_PATH` for you.

That gives you two good ways to keep this repo pinned to the right Modal team/environment:

1. Set `MODAL_PROFILE` and `MODAL_ENVIRONMENT` in `.env` if your `~/.modal.toml` already has the right profile.
2. Copy `.modal.toml.example` to `.modal.toml`, paste the token for the exact Modal workspace you want, and set `MODAL_PROFILE` in `.env` to that section name.

For repo-scoped CLI commands that should honor the same settings, use:

```bash
./scripts/modal.sh app list
./scripts/modal.sh run worker/modal_worker.py --help
```

## LanceDB Notes

This setup keeps Postgres for operational metadata (`panoramas`, `captures`, scan state)
and uses LanceDB for retrieval.

Notes:
- New scan/enrich/fill captures now write embeddings through the configured vector backend on ingest.
- Lance mode still keeps Postgres for operational metadata and does not yet support the Postgres-style missing-embedding backfill flow (`/api/retrieval/index-missing` remains disabled).
- Switch back at any time with `GEOSPY_VECTOR_BACKEND=postgres`.

## Alternative: pgvector-Only Retrieval

If you want to run retrieval directly from pgvector without LanceDB:

```bash
export GEOSPY_VECTOR_BACKEND=postgres
python -m backend.app.main
```

## Search Flow

1. Open the `Search` page.
2. Upload a query image.
3. Click `Search by image`.
4. Review ranked matches; best match is auto-focused on the map.
5. If a result has no local image, click it to drop a map dot and open Street View.

## Locate Flow

1. Open the `Locate` page.
2. Upload a query image.
3. Click `Locate by image`.
4. Backend runs vector search across all active embedding families and then reranks by panorama-family clusters.
5. Review the top location families and the pipeline state returned by the backend.

## What Is In The Snapshot DB

Snapshot: `data-snapshot-v1` (profile measured on 2026-03-04)

- Region: San Francisco captures.
- `captures`: `4,353,633` rows.
- `panoramas`: `27,161` rows.
- `capture_embeddings`: `8,707,266` rows.
- Embedding models:
  - `ViT-B-32:open_clip` (`512d`) -> `4,353,633` vectors.
  - `ViT-B-16:open_clip_place` (`512d`, place family) -> `4,353,633` vectors.
- Capture angle pattern:
  - headings: `0..345` in 15-degree steps (`24` slices)
  - pitch levels: `45`, `60`, `75`, `90`, `105`
- Captures per panorama:
  - average: `160.29`
  - min/max: `4` / `1456`

Not included in snapshot SQL:

- local image files under `./captures`

## Snapshot Restore Commands

Restore from a single-file pgvector snapshot URL:

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
- `scripts/sync_pgvector_to_lancedb.py` - copy historical `capture_embeddings` rows from pgvector into LanceDB.
- `utils/index_capture_embeddings.py` - (re)index embeddings.

## Evaluation

Automated locator eval tooling now lives under `eval/`.

- `python -m eval.build_locator_dataset` - build a starter manifest from local captures.
- `python -m eval.run_locator` - run `locate-by-image` evals and compute `within_25m/50m/100m`.

See [eval/README.md](/Users/alexpickett/Desktop/Projects/geoSpy/eval/README.md) for the manifest format and negative-case workflow.
