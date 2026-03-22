import os
from pathlib import Path

_ENV_LOADED = False


def _parse_env_line(line: str):
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None, None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None, None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def load_project_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, value = _parse_env_line(line)
            if key:
                os.environ.setdefault(key, value)

    modal_config_path = project_root / ".modal.toml"
    if modal_config_path.exists():
        os.environ.setdefault("MODAL_CONFIG_PATH", str(modal_config_path))

    # Default new backend launches to LanceDB unless explicitly overridden.
    os.environ.setdefault("GEOSPY_VECTOR_BACKEND", "lancedb")
    if os.environ.get("GEOSPY_VECTOR_BACKEND", "").strip().lower() in {"lance", "lancedb"}:
        os.environ.setdefault("GEOSPY_LANCEDB_URI", str(project_root / ".lancedb"))
        os.environ.setdefault("GEOSPY_LANCEDB_TABLE", "capture_embeddings")
        os.environ.setdefault("GEOSPY_LANCEDB_VECTOR_COLUMN", "embedding")

    _ENV_LOADED = True
