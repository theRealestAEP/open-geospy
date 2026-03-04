#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-geospy-postgres}"
DB_NAME="${DB_NAME:-geospy}"
DB_USER="${DB_USER:-geospy}"
OUTPUT_DIR="${OUTPUT_DIR:-backups}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-geospy_pgvector_snapshot}"
RELEASE_TAG=""

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
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --prefix)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    --release-tag)
      RELEASE_TAG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${TIMESTAMP}.sql.gz"

echo "Exporting ${DB_NAME} from container ${CONTAINER_NAME} to ${OUTPUT_FILE}"
docker exec -i "$CONTAINER_NAME" \
  pg_dump -U "$DB_USER" -d "$DB_NAME" --format=plain --no-owner --no-privileges \
  | gzip -9 > "$OUTPUT_FILE"

echo "Snapshot created: ${OUTPUT_FILE}"
echo "Size: $(du -h "$OUTPUT_FILE" | awk '{print $1}')"

if [[ -n "$RELEASE_TAG" ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI not found. Install GitHub CLI to upload release assets." >&2
    exit 1
  fi
  if ! gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
    echo "Creating release ${RELEASE_TAG}"
    gh release create "$RELEASE_TAG" --title "$RELEASE_TAG" --notes "PGVector snapshot"
  fi
  echo "Uploading snapshot to release ${RELEASE_TAG}"
  gh release upload "$RELEASE_TAG" "$OUTPUT_FILE" --clobber
  echo "Upload complete."
fi

echo "Tip: avoid committing DB dumps to git; use GitHub Releases assets instead."
