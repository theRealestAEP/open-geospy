#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-geospy-postgres}"
DB_NAME="${DB_NAME:-geospy}"
DB_USER="${DB_USER:-geospy}"
SNAPSHOT_URL=""
SNAPSHOT_FILE=""
COMPOSE_UP="1"
SKIP_DROP="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --db)
      DB_NAME="$2"
      shift 2
      ;;
    --user)
      DB_USER="$2"
      shift 2
      ;;
    --snapshot-url)
      SNAPSHOT_URL="$2"
      shift 2
      ;;
    --snapshot-file)
      SNAPSHOT_FILE="$2"
      shift 2
      ;;
    --no-compose-up)
      COMPOSE_UP="0"
      shift
      ;;
    --skip-drop)
      SKIP_DROP="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SNAPSHOT_URL" && -z "$SNAPSHOT_FILE" ]]; then
  echo "Provide --snapshot-url or --snapshot-file" >&2
  exit 1
fi
if [[ -n "$SNAPSHOT_URL" && -n "$SNAPSHOT_FILE" ]]; then
  echo "Use either --snapshot-url or --snapshot-file, not both." >&2
  exit 1
fi

if [[ "$COMPOSE_UP" == "1" ]]; then
  echo "Starting postgres container via docker compose"
  docker compose up -d postgres
fi

echo "Waiting for postgres readiness in ${CONTAINER_NAME}"
for _ in $(seq 1 40); do
  if docker exec -i "$CONTAINER_NAME" pg_isready -U "$DB_USER" -d postgres >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

WORK_FILE=""
if [[ -n "$SNAPSHOT_URL" ]]; then
  WORK_FILE="$(mktemp /tmp/geospy_snapshot_XXXXXX.sql.gz)"
  echo "Downloading snapshot: ${SNAPSHOT_URL}"
  curl -L --fail --output "$WORK_FILE" "$SNAPSHOT_URL"
else
  WORK_FILE="$SNAPSHOT_FILE"
fi

if [[ ! -f "$WORK_FILE" ]]; then
  echo "Snapshot file not found: $WORK_FILE" >&2
  exit 1
fi

if [[ "$SKIP_DROP" != "1" ]]; then
  echo "Dropping and recreating database ${DB_NAME}"
  docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d postgres <<SQL
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${DB_NAME}';
DROP DATABASE IF EXISTS ${DB_NAME};
CREATE DATABASE ${DB_NAME};
SQL
fi

echo "Ensuring vector extension exists"
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "Restoring snapshot into ${DB_NAME}"
if [[ "$WORK_FILE" == *.gz ]]; then
  gunzip -c "$WORK_FILE" | docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME"
else
  cat "$WORK_FILE" | docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME"
fi

echo "Running ANALYZE"
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" -c "ANALYZE;"

if [[ -n "$SNAPSHOT_URL" ]]; then
  rm -f "$WORK_FILE"
fi

echo "Restore complete."
echo "Next steps:"
echo "  cp env.example .env"
echo "  python -m backend.app.main"
