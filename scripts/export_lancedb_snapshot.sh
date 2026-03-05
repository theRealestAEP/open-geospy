#!/usr/bin/env bash
set -euo pipefail

LANCEDB_DIR="${LANCEDB_DIR:-.lancedb}"
OUTPUT_DIR="${OUTPUT_DIR:-backups}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-geospy_lancedb_snapshot}"
RELEASE_TAG=""
MAX_RELEASE_ASSET_MB="${MAX_RELEASE_ASSET_MB:-1900}"
EXISTING_SNAPSHOT_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lancedb-dir)
      LANCEDB_DIR="$2"
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
    --snapshot-file)
      EXISTING_SNAPSHOT_FILE="$2"
      shift 2
      ;;
    --max-release-asset-mb)
      MAX_RELEASE_ASSET_MB="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
if [[ -n "$EXISTING_SNAPSHOT_FILE" ]]; then
  OUTPUT_FILE="$EXISTING_SNAPSHOT_FILE"
  if [[ ! -f "$OUTPUT_FILE" ]]; then
    echo "Snapshot file not found: $OUTPUT_FILE" >&2
    exit 1
  fi
  echo "Using existing snapshot: ${OUTPUT_FILE}"
else
  if [[ ! -d "$LANCEDB_DIR" ]]; then
    echo "LanceDB directory not found: $LANCEDB_DIR" >&2
    exit 1
  fi
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${TIMESTAMP}.tar.gz"
  echo "Archiving ${LANCEDB_DIR} to ${OUTPUT_FILE}"
  tar -C "$(dirname "$LANCEDB_DIR")" -czf "$OUTPUT_FILE" "$(basename "$LANCEDB_DIR")"
fi

echo "Snapshot created: ${OUTPUT_FILE}"
echo "Size: $(du -h "$OUTPUT_FILE" | awk '{print $1}')"

if [[ -n "$RELEASE_TAG" ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI not found. Install GitHub CLI to upload release assets." >&2
    exit 1
  fi
  if ! gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
    echo "Creating release ${RELEASE_TAG}"
    gh release create "$RELEASE_TAG" --title "$RELEASE_TAG" --notes "LanceDB snapshot"
  fi
  file_bytes="$(wc -c < "$OUTPUT_FILE" | tr -d ' ')"
  max_bytes="$((MAX_RELEASE_ASSET_MB * 1024 * 1024))"
  if [[ "$file_bytes" -le "$max_bytes" ]]; then
    echo "Uploading snapshot to release ${RELEASE_TAG}"
    gh release upload "$RELEASE_TAG" "$OUTPUT_FILE" --clobber
    echo "Upload complete."
  else
    echo "Snapshot exceeds per-asset limit; splitting into ${MAX_RELEASE_ASSET_MB}MB chunks"
    part_prefix="${OUTPUT_FILE}.part-"
    split -b "${MAX_RELEASE_ASSET_MB}m" -d -a 4 "$OUTPUT_FILE" "$part_prefix"
    manifest_file="${OUTPUT_FILE}.parts.txt"
    ls "${part_prefix}"* | while read -r part; do
      basename "$part"
    done > "$manifest_file"
    total_parts="$(wc -l < "$manifest_file" | tr -d ' ')"
    current_part=0
    while read -r part_name; do
      [[ -z "$part_name" ]] && continue
      current_part=$((current_part + 1))
      echo "Uploading part ${current_part}/${total_parts}: ${part_name}"
      gh release upload "$RELEASE_TAG" "$(dirname "$OUTPUT_FILE")/$part_name" --clobber
    done < "$manifest_file"
    echo "Uploading parts manifest: $(basename "$manifest_file")"
    gh release upload "$RELEASE_TAG" "$manifest_file" --clobber
    echo "Uploaded parts + manifest:"
    echo "  manifest: $(basename "$manifest_file")"
    echo "Restore with:"
    echo "  curl -L -o /tmp/lance.parts.txt https://github.com/<org>/<repo>/releases/download/${RELEASE_TAG}/$(basename "$manifest_file")"
    echo "  # then download each part from the same release URL base and concatenate"
  fi
fi

