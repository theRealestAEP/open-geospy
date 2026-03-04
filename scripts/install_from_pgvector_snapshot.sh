#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-geospy-postgres}"
DB_NAME="${DB_NAME:-geospy}"
DB_USER="${DB_USER:-geospy}"
SNAPSHOT_URL=""
SNAPSHOT_FILE=""
PARTS_BASE_URL=""
PARTS_MANIFEST_URL=""
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
    --parts-base-url)
      PARTS_BASE_URL="$2"
      shift 2
      ;;
    --parts-manifest-url)
      PARTS_MANIFEST_URL="$2"
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

if [[ -z "$SNAPSHOT_URL" && -z "$SNAPSHOT_FILE" && -z "$PARTS_MANIFEST_URL" ]]; then
  echo "Provide --snapshot-url, --snapshot-file, or --parts-manifest-url with --parts-base-url" >&2
  exit 1
fi
mode_count=0
[[ -n "$SNAPSHOT_URL" ]] && mode_count=$((mode_count + 1))
[[ -n "$SNAPSHOT_FILE" ]] && mode_count=$((mode_count + 1))
[[ -n "$PARTS_MANIFEST_URL" ]] && mode_count=$((mode_count + 1))
if [[ "$mode_count" -gt 1 ]]; then
  echo "Use one install mode: single URL, single file, or multi-part manifest." >&2
  exit 1
fi
if [[ -n "$PARTS_MANIFEST_URL" && -z "$PARTS_BASE_URL" ]]; then
  echo "--parts-base-url is required with --parts-manifest-url" >&2
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
elif [[ -n "$PARTS_MANIFEST_URL" ]]; then
  manifest_file="$(mktemp /tmp/geospy_snapshot_parts_XXXXXX.txt)"
  WORK_FILE="$(mktemp /tmp/geospy_snapshot_merged_XXXXXX.sql.gz)"
  echo "Downloading parts manifest: ${PARTS_MANIFEST_URL}"
  curl -L --fail --output "$manifest_file" "$PARTS_MANIFEST_URL"
  : > "$WORK_FILE"
  while IFS= read -r part_name; do
    [[ -z "$part_name" ]] && continue
    part_url="${PARTS_BASE_URL%/}/${part_name}"
    part_tmp="$(mktemp /tmp/geospy_snapshot_part_XXXXXX.bin)"
    echo "Downloading part: ${part_url}"
    curl -L --fail --output "$part_tmp" "$part_url"
    cat "$part_tmp" >> "$WORK_FILE"
    rm -f "$part_tmp"
  done < "$manifest_file"
  rm -f "$manifest_file"
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

if [[ -n "$SNAPSHOT_URL" || -n "$PARTS_MANIFEST_URL" ]]; then
  rm -f "$WORK_FILE"
fi

echo "Restore complete."
echo "Next steps:"
echo "  cp env.example .env"
echo "  python -m backend.app.main"
