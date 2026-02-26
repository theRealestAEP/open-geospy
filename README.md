# Street View Crawler & Coverage Tracker

A Playwright-based web automation that navigates Google Street View, captures panoramic screenshots at each location, stores metadata (lat/lon/heading/pitch), deduplicates by proximity, and displays coverage on an interactive map. Supports parallel crawling via local workers or Modal.com cloud containers.

## Architecture

```
┌─────────────────────────────────────────────────┐
│               Coverage Viewer                    │
│                                                  │
│  FastAPI + Leaflet map at http://localhost:8000   │
│  - Draw a polygon/free-draw area to scan         │
│  - Configure workers, step size, run mode        │
│  - Real-time scan progress monitoring            │
│  - Click dots to preview captured images         │
│  - One-shot capture via Alt+Click                │
└───────────────┬─────────────────────────────────┘
                │  POST /api/scan-area
                ▼
┌─────────────────────────────────────────────────┐
│             Scan Pipeline                        │
│                                                  │
│  1. Generate seed grid from bounding box         │
│  2. Filter water points (global_land_mask)       │
│  3. Insert land seeds into seed_tasks queue      │
│  4. Dispatch N workers (local or Modal)          │
└───────────────┬─────────────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌──────────────┐ ┌──────────────┐
│ Local Worker │ │ Modal Worker │
│  (subprocess │ │  (container  │
│   headless)  │ │   headless)  │
└──────┬───────┘ └──────┬───────┘
       │                │
       ▼                ▼
┌─────────────────────────────────────────────────┐
│              Batch Crawler Loop                  │
│                                                  │
│  1. Claim next seed from DB queue                │
│  2. Check water filter (skip if water)           │
│  3. Check dedup (skip if within radius)          │
│  4. Navigate to Street View in headless browser  │
│  5. Parse snapped lat/lon/heading from URL       │
│  6. Fetch thumbnail images at 4 headings         │
│  7. Save images + metadata to disk/Postgres      │
│  8. Mark task done → repeat                      │
└───────────────┬─────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────┐
│              Storage Layer                       │
│                                                  │
│  Postgres: geospy                                │
│  ├── panoramas (id, lat, lon, heading, pitch,    │
│  │              timestamp, pano_id, source_url)  │
│  ├── captures  (id, panorama_id, heading,        │
│  │              filepath, width, height)         │
│  └── seed_tasks (id, lat, lon, status,           │
│                   claimed_by, attempts)          │
│                                                  │
│  Disk: ./captures/{pano_id}/h{heading}.jpg       │
└─────────────────────────────────────────────────┘
```

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

### Local Postgres (Docker Compose)

```bash
# optional: copy and edit defaults
cp env.example .env

# start Postgres
docker compose up -d postgres
```

The compose stack uses `pgvector/pgvector:pg16`, so vector search is available in the same Postgres instance.

### Dependencies

- `playwright` -- headless browser automation
- `fastapi` + `uvicorn` -- web server and coverage viewer
- `global-land-mask` -- offline land/water detection (~1km resolution)
- `modal` -- cloud container orchestration (optional, for parallelization)
- `opencv-python-headless` -- geometric verification (ORB + RANSAC) for image locator reranking

## Usage

### Interactive area scan (recommended)

```bash
# Start the server
python -m backend.app.main
# Open http://localhost:8000

# On the map:
#   1. Use Pg (polygon) or FD (free draw) on the map controls
#   2. Choose job type: scan / enrich / fill
#   3. Configure workers, profile, step size, and run mode in the sidebar
#   4. Click "Start Scan" -- progress updates live in the sidebar
```

### SQLite -> Postgres migration

```bash
# Preview migration counts
python -m utils.migrate_sqlite_to_postgres --dry-run

# Run migration (requires empty target DB or --truncate-target)
python -m utils.migrate_sqlite_to_postgres --apply
```

### CLI workflow

```bash
# 1) Generate seed points for an area
python -m utils.seed_grid --bbox 37.70,-122.52,37.82,-122.35 --step 50 --output seeds.csv

# 2) (Optional) Pre-filter seeds near roads to remove water/empty areas
python -m utils.seed_filter_roads --input seeds.csv --output land_seeds.csv --near-road 90

# 3) Start one or more resumable batch workers
python -m worker.batch_crawler --seeds seeds.csv --max 2000 --headless --worker-id worker-1 --reset-queue
# Second worker against same DB queue:
python -m worker.batch_crawler --seeds seeds.csv --max 2000 --headless --worker-id worker-2

# 4) View coverage
python -m backend.app.main
# Open http://localhost:8000
```

### Modal.com deployment

```bash
# Dispatch N parallel workers in Modal containers
modal run worker/modal_worker.py --seeds seeds.csv --num-workers 8 --max-captures 2000
```

Or trigger Modal workers from the web UI by selecting "Modal" in the run mode dropdown before starting a scan.

### Headless mode testing

```bash
# Verify headless captures work before deploying to Modal
python -m utils.test_headless
```

### Cleanup utility

```bash
# Preview panoramas that have no captures (dry run)
python -m utils.prune_empty_locations --dry-run

# Delete panoramas that have no captures
python -m utils.prune_empty_locations --apply
```

### Postgres backup

```bash
# Create a timestamped compressed dump in ./backups
python -m utils.backup_postgres

# Restore from a dump
gunzip -c backups/<dump-file>.sql.gz | docker exec -i geospy-postgres psql -U geospy -d geospy
```

### Retrieval indexing (multi-model)

```bash
# Build embeddings for existing captures into capture_embeddings
python -m utils.index_capture_embeddings --batch-size 64

# Offload embedding inference to Modal for higher parallelism
# (example uses 64 concurrent Modal workers)
python -m utils.index_capture_embeddings \
  --mode modal \
  --batch-size 4096 \
  --modal-workers 64 \
  --modal-worker-batch-size 64

# Migrate/reindex capture_embeddings schema for multi-model keys
python -m utils.migrate_capture_embeddings_schema --reindex
```

Then use the "Image retrieval" panel in the web UI to upload a reference image and query nearest matches.
The backend also runs a low-impact auto-indexer by default (small batches) so new scan/enrich captures
get embedded continuously. Control with:
- `GEOSPY_AUTO_INDEX_ENABLED`
- `GEOSPY_AUTO_INDEX_INTERVAL_SECONDS`
- `GEOSPY_AUTO_INDEX_BATCH_SIZE`
- `GEOSPY_MODAL_EMBED_ENVIRONMENT`
- `GEOSPY_MODAL_EMBED_MAX_WORKERS`
- `GEOSPY_MODAL_EMBED_BATCH_SIZE`
- `GEOSPY_MODAL_EMBED_MAX_RETRIES`

The retrieval panel also supports a two-step locator flow:
- `Search by image` -- standard vector nearest-neighbor matches
- `Locate image` -- multi-crop, multi-model retrieval + vote cap + geo clustering + geometric verification + appearance penalties

Locate tuning is now primarily frontend-driven from the `Locate` page:
- basic knob: `Locate Top K / crop`
- advanced knobs: `max_candidates`, `verify_top_n`, `min_good_matches`, `min_inlier_ratio`, `panorama_vote_cap`, `cluster_radius_m`, `appearance_penalty_weight`, `db_max_top_k`, `ivfflat_probes`
- values persist in browser local storage and are sent on each `/api/retrieval/locate-by-image` request
- backend publishes default values and valid ranges at `GET /api/retrieval/locate-params`

Detailed retrieval architecture write-up (with flowchart):
- `docs/retrieval_pipeline.md`

Locator env vars now act as startup defaults/fallbacks (useful for API-only clients and baseline policy):
- `GEOSPY_RETRIEVAL_INCLUDE_DEBUG_DEFAULT`
- `GEOSPY_LOCATOR_TOP_K_PER_CROP`
- `GEOSPY_LOCATOR_MAX_MERGED_CANDIDATES`
- `GEOSPY_LOCATOR_PANORAMA_VOTE_CAP`
- `GEOSPY_LOCATOR_CLUSTER_RADIUS_M`
- `GEOSPY_LOCATOR_VERIFY_TOP_N`
- `GEOSPY_LOCATOR_MIN_GOOD_MATCHES`
- `GEOSPY_LOCATOR_MIN_INLIER_RATIO`
- `GEOSPY_LOCATOR_APPEARANCE_PENALTY_WEIGHT`
- `GEOSPY_PLACE_MODEL_ENABLED`
- `GEOSPY_PLACE_MODEL`
- `GEOSPY_PLACE_PRETRAINED`
- `GEOSPY_PLACE_VERSION`
- `GEOSPY_RETRIEVAL_MIN_MODEL_COVERAGE` (skip under-covered secondary models during query-time fusion)
- `GEOSPY_RETRIEVAL_PRIMARY_WEIGHT`
- `GEOSPY_RETRIEVAL_PLACE_WEIGHT`
- `GEOSPY_RETRIEVAL_DILIGENT_MODE` (default `1`; accuracy-first broader candidate search)
- `GEOSPY_RETRIEVAL_SEARCH_CANDIDATE_MULTIPLIER`
- `GEOSPY_RETRIEVAL_SEARCH_MAX_CANDIDATES`
- `GEOSPY_RETRIEVAL_IVFFLAT_PROBES` (higher probes improves ANN recall at latency cost)
- `GEOSPY_RETRIEVAL_QUERY_ADAPTER_PATH`
- `GEOSPY_RETRIEVAL_QUERY_ADAPTER_CLIP_PATH`
- `GEOSPY_RETRIEVAL_QUERY_ADAPTER_PLACE_PATH`
- `GEOSPY_LOCATOR_DB_MAX_TOP_K`
- `GEOSPY_LOCATOR_MODAL_RERANK_ENABLED`
- `GEOSPY_LOCATOR_MODAL_RERANK_REQUIRE_SUCCESS`
- `GEOSPY_LOCATOR_MODAL_RERANK_TOP_N`
- `GEOSPY_LOCATOR_MODAL_RERANK_WEIGHT`
- `GEOSPY_MODAL_RETRIEVAL_ENVIRONMENT`
- `GEOSPY_MODAL_RERANK_MODEL`
- `GEOSPY_MODAL_RERANK_PRETRAINED`
- `GEOSPY_MODAL_RERANK_MAX_WORKERS`
- `GEOSPY_MODAL_RERANK_BATCH_SIZE`
- `GEOSPY_MODAL_RERANK_MAX_RETRIES`
- `GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_ENABLED` (SuperPoint + LightGlue verification stage in Modal)
- `GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_REQUIRE_SUCCESS`
- `GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_TOP_N`
- `GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_MAX_WORKERS`
- `GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_BATCH_SIZE`
- `GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_MAX_RETRIES`
- `GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_MAX_KEYPOINTS`
- `GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_RESIZE_LONG_EDGE`
- `GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_MIN_MATCHES`

### Locator evaluation utility

```bash
# CSV columns: id,image_path,lat,lon
python -m utils.eval_retrieval_locator --queries-csv queries.csv
```

### Partial-match evaluation harness (accuracy tuning / fine-tune prep)

```bash
# Samples existing captures with embeddings, generates partial crops,
# and reports exact/same-panorama hit rates for search + locate modes.
python -m utils.eval_retrieval_partials \
  --sample-size 400 \
  --mode both \
  --top-k 20 \
  --variants center80,center60,center40,left,right,top,bottom,q1,q2,q3,q4 \
  --db-max-top-k 5000 \
  --ivfflat-probes 120 \
  --locate-top-k-per-crop 220 \
  --locate-max-candidates 5000 \
  --locate-verify-top-n 120 \
  --out-csv partial_eval_results.csv \
  --hard-negatives-csv partial_eval_hard_negatives.csv
```

Use `partial_eval_hard_negatives.csv` as your hard-negative set for fusion/rerank fine-tuning.

### Overnight hard-negative fine-tune loop

```bash
# One command: mine hard negatives with high-recall retrieval, then train query adapter.
python -m utils.run_hard_negative_finetune_loop \
  --iterations 1 \
  --sample-size 1500 \
  --db-max-top-k 5000 \
  --ivfflat-probes 120 \
  --locate-top-k-per-crop 220 \
  --locate-max-candidates 5000 \
  --locate-verify-top-n 120 \
  --train-model-id clip \
  --train-max-triplets 80000 \
  --train-epochs 4 \
  --train-batch-size 128
```

The loop prints `latest_adapter=<path>.pt`. Enable it for runtime queries:

```bash
export GEOSPY_RETRIEVAL_QUERY_ADAPTER_CLIP_PATH=/absolute/path/to/query_adapter_clip.pt
python -m backend.app.main
```

### Enrichment and fill validation helpers

```bash
# Missing-view coverage for selected area/profile
python -m utils.check_enrichment_missing \
  --min-lat 37.784 --min-lon -122.438 --max-lat 37.801 --max-lon -122.422 \
  --profile high_v1

# Fill candidate estimate before launching a fill job
python -m utils.check_fill_candidates \
  --min-lat 37.784 --min-lon -122.438 --max-lat 37.801 --max-lon -122.422 \
  --step-meters 50 --gap-meters 40
```

## Worker Queue Model

- Seed CSV points are inserted into `seed_tasks` in Postgres.
- Each worker claims tasks atomically (`pending` -> `in_progress`) with a lease timeout.
- If a worker crashes, stale `in_progress` tasks are reclaimed automatically.
- Dedupe happens by `pano_id` when available, with distance-based fallback.
- Water points are skipped at both seed generation and crawl time.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/scan-area` | POST | Draw-to-scan: generate seeds, filter water, dispatch workers |
| `/api/scan-status` | GET | Queue stats, active worker counts, per-scan progress |
| `/api/scan-stop` | POST | Kill active scan workers |
| `/api/capture-once` | POST | One-shot capture at a single coordinate |
| `/api/panoramas` | GET | All panoramas as GeoJSON (legacy/full export) |
| `/api/panoramas/bbox` | GET | Viewport query; returns clusters at low zoom and raw points at high zoom |
| `/api/stats` | GET | Summary statistics |
| `/api/panorama/{id}` | GET | Captures for a specific panorama |
| `/api/queue` | GET | Seed task queue breakdown |
| `/api/retrieval/index-stats` | GET | Embedding coverage stats (primary + per-model) |
| `/api/retrieval/locate-params` | GET | Locate tuning defaults + bounds for frontend/API clients |
| `/api/retrieval/search-by-image` | POST | Upload reference image and return top-K similar captures |
| `/api/retrieval/locate-by-image` | POST | Two-step geolocation with request-level tuning controls (including `db_max_top_k` + `ivfflat_probes`) |
| `/api/retrieval/index-missing` | POST | Backfill embeddings for captures missing any configured retrieval model |
| `/api/scan-area` | POST | Supports `job_type=scan|enrich|fill` with optional capture profile |

## Configuration

Edit `config.py` to adjust:
- `DATABASE_URL` -- Postgres connection string
- `DEDUP_RADIUS_METERS` -- minimum distance between captures (default 25m)
- `HEADINGS` -- which directions to capture at each point (default: 0, 90, 180, 270)
- `CAPTURE_DELAY` -- seconds to wait for Street View to render
- `MAX_CAPTURES` -- stop after N panoramas
- `NAV_STRATEGY` -- 'bfs', 'dfs', or 'random'

## Project Structure

```
backend/app/main.py  Web UI + API (FastAPI)
backend/app/clip_embeddings.py Multi-model embedding loader/encoder (primary + place branch)
backend/app/services/retrieval_locator.py Two-step locator pipeline + structured retrieval logging
frontend/            React + Vite frontend
worker/crawler.py    Single-seed Playwright crawler (BFS/DFS/random)
worker/batch_crawler.py Batch crawler with CSV seeds and worker task claiming
worker/modal_worker.py Modal.com app for cloud-parallel crawling
worker/modal_retrieval_worker.py Modal app for high-recall retrieval second-pass reranking
config.py            Crawler configuration dataclass
db/postgres_database.py Postgres adapter implementation
seeds/               Generated scan seed CSVs
utils/seed_grid.py   Bounding box to seed grid generator
utils/seed_filter_roads.py OSM road proximity filter (pre-filter for water/empty)
utils/backup_postgres.py Local Postgres backup helper
utils/index_capture_embeddings.py CLIP embedding backfill script
utils/migrate_capture_embeddings_schema.py Multi-model embedding schema migration/reindex helper
utils/eval_retrieval_locator.py Batch evaluator for locator distance error from labeled query CSV
utils/eval_retrieval_partials.py Synthetic partial-crop evaluator for exact/same-panorama retrieval hit-rate
utils/train_retrieval_query_adapter.py Hard-negative trainer for query-side linear adapter
utils/run_hard_negative_finetune_loop.py End-to-end overnight mining + adapter training loop
utils/check_enrichment_missing.py Missing-view audit helper for enrichment profile
utils/check_fill_candidates.py Fill-candidate estimator for selected area
worker/water_filter.py Fast offline land/water checks (global_land_mask)
utils/test_headless.py Headless mode smoke test
```
