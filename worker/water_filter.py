"""Fast local land/water checks for seed filtering."""

import json
import logging
import os
import subprocess
import sys
import threading
from typing import Iterable, List, Tuple

log = logging.getLogger(__name__)

WATER_FILTER_MODE = os.getenv("GEOSPY_WATER_FILTER_MODE", "global_land_mask").strip().lower()
_water_filter_available = True
_water_filter_lock = threading.Lock()
_GLOBAL_LAND_MASK_SCRIPT = (
    "import json,sys\n"
    "from global_land_mask import globe\n"
    "points=json.load(sys.stdin)\n"
    "result=[(not bool(globe.is_land(float(lat), float(lon)))) for lat, lon in points]\n"
    "json.dump(result, sys.stdout)\n"
)


def _can_use_water_filter() -> bool:
    if WATER_FILTER_MODE in {"off", "0", "false", "disabled", "none"}:
        return False
    with _water_filter_lock:
        return _water_filter_available


def _disable_water_filter(reason: str) -> None:
    global _water_filter_available
    with _water_filter_lock:
        if _water_filter_available:
            _water_filter_available = False
            log.warning("Disabling water filter for this process: %s", reason)


def _run_global_land_mask(points: List[Tuple[float, float]]) -> List[bool]:
    payload = json.dumps(points)
    proc = subprocess.run(
        [sys.executable, "-c", _GLOBAL_LAND_MASK_SCRIPT],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"global_land_mask subprocess failed with code {proc.returncode}: {stderr}"
        )
    try:
        flags = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid water-filter output: {exc}") from exc
    if len(flags) != len(points):
        raise RuntimeError(
            f"water-filter output length mismatch: expected {len(points)} got {len(flags)}"
        )
    return [bool(x) for x in flags]


def is_water(lat: float, lon: float) -> bool:
    return len(filter_water_points([(lat, lon)])) == 0


def filter_water_points(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    normalized_points = [(float(lat), float(lon)) for lat, lon in points]
    if not normalized_points:
        return []
    if not _can_use_water_filter():
        return normalized_points
    try:
        water_flags = _run_global_land_mask(normalized_points)
    except Exception as exc:
        _disable_water_filter(str(exc))
        return normalized_points
    return [
        point for point, is_water_point in zip(normalized_points, water_flags) if not is_water_point
    ]
