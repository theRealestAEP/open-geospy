#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

if [[ -f "${ROOT_DIR}/.modal.toml" && -z "${MODAL_CONFIG_PATH:-}" ]]; then
  export MODAL_CONFIG_PATH="${ROOT_DIR}/.modal.toml"
fi

exec modal "$@"
