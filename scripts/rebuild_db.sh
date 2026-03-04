#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-geospy-postgres}"
DB_NAME="${DB_NAME:-geospy}"
DB_USER="${DB_USER:-geospy}"
SNAPSHOT_URL=""
SNAPSHOT_FILE=""
PARTS_BASE_URL=""
PARTS_MANIFEST_URL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshot-url)
      SNAPSHOT_URL="$2"
      shift 2
      ;;
    --snapshot-file)
      SNAPSHOT_FILE="$2"
      shift 2
      ;;
    --parts-base-url)
      PARTS_BASE_URL="$2"
      shift 2
      ;;
    --parts-manifest-url)
      PARTS_MANIFEST_URL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "Starting postgres via docker compose"
docker compose up -d postgres

echo "Waiting for postgres in container ${CONTAINER_NAME}"
for _ in $(seq 1 40); do
  if docker exec -i "$CONTAINER_NAME" pg_isready -U "$DB_USER" -d postgres >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if [[ -n "$SNAPSHOT_URL" || -n "$SNAPSHOT_FILE" || -n "$PARTS_MANIFEST_URL" ]]; then
  echo "Rebuilding DB from snapshot"
  cmd=(./scripts/install_from_pgvector_snapshot.sh)
  if [[ -n "$SNAPSHOT_URL" ]]; then
    cmd+=(--snapshot-url "$SNAPSHOT_URL")
  fi
  if [[ -n "$SNAPSHOT_FILE" ]]; then
    cmd+=(--snapshot-file "$SNAPSHOT_FILE")
  fi
  if [[ -n "$PARTS_MANIFEST_URL" ]]; then
    cmd+=(--parts-manifest-url "$PARTS_MANIFEST_URL" --parts-base-url "$PARTS_BASE_URL")
  fi
  cmd+=(--no-compose-up)
  "${cmd[@]}"
  echo "DB rebuild complete from snapshot."
  exit 0
fi

echo "Rebuilding empty schema in ${DB_NAME}"
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d postgres <<SQL
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${DB_NAME}';
DROP DATABASE IF EXISTS ${DB_NAME};
CREATE DATABASE ${DB_NAME};
SQL

docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

python - <<'PY'
from config import CrawlerConfig
from db.postgres_database import Database

cfg = CrawlerConfig()
db = Database(cfg.DATABASE_URL)
db.close()
print("Schema initialized.")
PY

echo "Empty DB rebuild complete."
