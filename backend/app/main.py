"""
Coverage Viewer — FastAPI app serving a Leaflet map showing crawl progress.

Features:
    - View captured panoramas on a map with color coding
    - One-shot capture at a clicked coordinate
    - Draw a bounding box to scan an area with N workers (local or Modal)
    - Real-time scan progress monitoring

Usage:
    python -m backend.app.main
    # Open http://localhost:8000
"""

import asyncio
import csv
import io
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.app.api.retrieval import create_retrieval_router
from config import CrawlerConfig
from backend.app.clip_embeddings import (
    encode_image_batch_for_all_models,
    select_retrieval_embedders,
)
from db.postgres_database import Database
from utils.seed_grid import generate_grid
from worker.water_filter import filter_water_points

log = logging.getLogger(__name__)
app = FastAPI(title="Street View Coverage Tracker")
config = CrawlerConfig()
BASE_DIR = PROJECT_ROOT
oneshot_lock: Optional[asyncio.Lock] = None
MODAL_ENVIRONMENT = os.getenv("MODAL_ENVIRONMENT", "google-map-walkers")
SCAN_LOGS_DIR = os.path.join(BASE_DIR, "scan_logs")
SEEDS_DIR = os.path.join(BASE_DIR, "seeds")
FRONTEND_DIST_DIR = os.path.join(BASE_DIR, "frontend", "dist")
FRONTEND_DIST_INDEX = os.path.join(FRONTEND_DIST_DIR, "index.html")
FRONTEND_DIST_ASSETS = os.path.join(FRONTEND_DIST_DIR, "assets")
BACKEND_FALLBACK_INDEX = os.path.join(BASE_DIR, "backend", "app", "fallback_index.html")
CAPTURE_PROFILES = {
    "base": {
        "headings": [0.0, 90.0, 180.0, 270.0],
        "pitches": [75.0],
    },
    "high_v1": {
        "headings": [float(x) for x in range(0, 360, 15)],
        "pitches": [45.0, 60.0, 75.0, 90.0, 105.0],
    },
}
AUTO_INDEX_ENABLED = os.getenv("GEOSPY_AUTO_INDEX_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
AUTO_INDEX_INTERVAL_SECONDS = max(
    5, int(os.getenv("GEOSPY_AUTO_INDEX_INTERVAL_SECONDS", "20"))
)
AUTO_INDEX_BATCH_SIZE = max(1, int(os.getenv("GEOSPY_AUTO_INDEX_BATCH_SIZE", "4")))
AUTO_INDEX_EMBEDDING_BASE = str(
    os.getenv("GEOSPY_AUTO_INDEX_EMBEDDING_BASE", "clip")
).strip().lower()
MODAL_PROGRESS_STALE_SECONDS = max(
    30, int(os.getenv("GEOSPY_MODAL_PROGRESS_STALE_SECONDS", "180"))
)
MODAL_NO_PROGRESS_GRACE_SECONDS = max(
    MODAL_PROGRESS_STALE_SECONDS,
    int(os.getenv("GEOSPY_MODAL_NO_PROGRESS_GRACE_SECONDS", "900")),
)
auto_index_task: Optional[asyncio.Task] = None
auto_index_stop_event: Optional[asyncio.Event] = None

if not os.path.isabs(config.CAPTURES_DIR):
    config.CAPTURES_DIR = os.path.join(BASE_DIR, config.CAPTURES_DIR)

# Serve captured images (mount even before first capture exists).
os.makedirs(config.CAPTURES_DIR, exist_ok=True)
os.makedirs(SCAN_LOGS_DIR, exist_ok=True)
os.makedirs(SEEDS_DIR, exist_ok=True)
app.mount("/captures", StaticFiles(directory=config.CAPTURES_DIR), name="captures")
if os.path.isdir(FRONTEND_DIST_ASSETS):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_ASSETS), name="frontend-assets")

# Backfill mixed absolute/relative capture paths so frontend URLs stay stable.
_migration_db = Database(config.DATABASE_URL)
_path_stats = _migration_db.normalize_capture_filepaths(config.CAPTURES_DIR)
_migration_db.close()
if _path_stats["updated"] > 0:
    log.info(
        "Normalized capture paths in DB updated=%s total=%s",
        _path_stats["updated"],
        _path_stats["total"],
    )

# ─── Active scan tracking ──────────────────────────────────────────────────
# Maps scan_id -> list of subprocess PIDs (local mode) or Modal call IDs.
active_scans: Dict[str, dict] = {}


def _reconcile_modal_scan_state(scan_id: str, info: dict, now_ts: float) -> int:
    progress = info.get("modal_progress", {})
    workers_total = int(progress.get("workers_total", info.get("num_workers", 0)))
    workers_submitted = int(progress.get("workers_submitted", 0))
    workers_completed = int(progress.get("workers_completed", 0))
    workers_failed = int(progress.get("workers_failed", 0))
    workers_cancelled = int(progress.get("workers_cancelled", 0))
    computed_running = max(
        0, workers_submitted - workers_completed - workers_failed - workers_cancelled
    )
    terminal_status = {"finished", "stopped"}
    status_str = str(info.get("status", "running"))

    if (
        workers_total > 0
        and workers_completed + workers_failed + workers_cancelled >= workers_total
        and status_str not in terminal_status
        and not status_str.startswith("failed:")
    ):
        info["status"] = "stopped" if info.get("stop_requested") else "finished"
        status_str = str(info.get("status", "running"))
        _append_scan_log(
            scan_id,
            (
                f"scan reconciled to terminal state={info['status']} workers_total={workers_total} "
                f"workers_completed={workers_completed} workers_failed={workers_failed} "
                f"workers_cancelled={workers_cancelled}"
            ),
        )

    last_event_ts = float(
        progress.get(
            "last_event_ts",
            info.get("started_at_ts", now_ts),
        )
    )
    idle_seconds = max(0.0, now_ts - last_event_ts)

    if status_str in terminal_status or status_str.startswith("failed:"):
        computed_running = 0
    elif computed_running > 0 and idle_seconds > float(MODAL_PROGRESS_STALE_SECONDS):
        computed_running = 0
        progress["workers_running"] = 0
        progress["stale"] = True
        progress["stale_seconds"] = int(idle_seconds)
    else:
        progress["stale"] = False
        progress["stale_seconds"] = int(idle_seconds)

    if (
        status_str in {"running", "stopping"}
        and idle_seconds > float(MODAL_NO_PROGRESS_GRACE_SECONDS)
    ):
        info["status"] = (
            "stopped" if status_str == "stopping" else "failed: stale-no-progress-timeout"
        )
        computed_running = 0
        progress["workers_running"] = 0
        progress["stale"] = True
        progress["stale_seconds"] = int(idle_seconds)
        _append_scan_log(
            scan_id,
            f"scan stale timeout status={info['status']} idle_seconds={int(idle_seconds)}",
        )

    return computed_running


def get_db():
    return Database(config.DATABASE_URL)


@app.on_event("startup")
async def _startup_auto_index():
    global auto_index_task, auto_index_stop_event
    if not AUTO_INDEX_ENABLED:
        log.info("Auto-indexer disabled (GEOSPY_AUTO_INDEX_ENABLED=0)")
        return
    if auto_index_task and not auto_index_task.done():
        return
    auto_index_stop_event = asyncio.Event()
    auto_index_task = asyncio.create_task(_auto_index_loop())
    log.info(
        "Auto-indexer started batch_size=%s interval_s=%s embedding_base=%s",
        AUTO_INDEX_BATCH_SIZE,
        AUTO_INDEX_INTERVAL_SECONDS,
        AUTO_INDEX_EMBEDDING_BASE,
    )


@app.on_event("shutdown")
async def _shutdown_auto_index():
    global auto_index_task, auto_index_stop_event
    if auto_index_stop_event:
        auto_index_stop_event.set()
    if auto_index_task:
        try:
            await asyncio.wait_for(auto_index_task, timeout=5.0)
        except asyncio.TimeoutError:
            auto_index_task.cancel()
        except Exception:
            pass
    auto_index_task = None
    auto_index_stop_event = None


# ─── Request / Response models ─────────────────────────────────────────────

class OneShotRequest(BaseModel):
    lat: float
    lon: float


class ScanAreaRequest(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    polygon_coords: Optional[List[List[float]]] = None  # [[lat, lon], ...]
    num_workers: int = 4
    step_meters: float = 50.0
    dedup_radius: float = 25.0
    mode: str = "local"  # "local" or "modal"
    job_type: str = "scan"  # "scan", "enrich", "fill"
    capture_profile: str = "high_v1"
    fill_gap_meters: float = 40.0
    enrich_missing_only: bool = True


class ScanStopRequest(BaseModel):
    scan_id: Optional[str] = None


# ─── Helpers ───────────────────────────────────────────────────────────────

def _tail_text(data: bytes, max_chars: int = 1200) -> str:
    text = data.decode("utf-8", errors="replace")
    return text[-max_chars:]


def _make_seeds_csv_bytes(points: List[tuple]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["lat", "lon"])
    writer.writerows(points)
    return buf.getvalue().encode("utf-8")


def _scan_log_path(scan_id: str) -> str:
    return os.path.join(SCAN_LOGS_DIR, f"scan_{scan_id}.log")


def _append_scan_log(scan_id: str, message: str):
    ts = datetime.utcnow().isoformat()
    with open(_scan_log_path(scan_id), "a", encoding="utf-8") as f:
        f.write(f"{ts} {message}\n")


def _capture_web_path(filepath: str) -> str:
    """
    Convert DB capture filepath to a browser path under /captures.
    Supports both legacy absolute paths and relative captures/... paths.
    """
    raw = (filepath or "").strip()
    unix = raw.replace("\\", "/")
    if not unix:
        return ""
    if unix.startswith("/captures/"):
        return unix
    if unix.startswith("captures/"):
        return f"/{unix}"
    if "/captures/" in unix:
        return f"/captures/{unix.split('/captures/', 1)[1].lstrip('/')}"
    return f"/{unix.lstrip('/')}"


def _capture_abs_path(filepath: str) -> str:
    raw = (filepath or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    normalized = raw.replace("\\", "/")
    if normalized.startswith("captures/"):
        relative_inside = normalized[len("captures/") :]
        return os.path.join(config.CAPTURES_DIR, relative_inside)
    return os.path.join(BASE_DIR, raw)


async def _index_missing_embeddings_once(limit: int) -> dict:
    limit = max(1, int(limit))
    try:
        embedders = list(
            select_retrieval_embedders(
                AUTO_INDEX_EMBEDDING_BASE, allow_fallback=False
            )
        )
    except RuntimeError as exc:
        return {"attempted": 0, "indexed": 0, "skipped": 0, "error": str(exc)}
    if not embedders:
        return {"attempted": 0, "indexed": 0, "skipped": 0, "error": "no-models-configured"}

    db = get_db()
    indexed = 0
    skipped = 0
    attempted = 0
    indexed_by_model: Dict[str, int] = {}
    try:
        if not db.is_vector_ready():
            return {"attempted": 0, "indexed": 0, "skipped": 0}
        rows = db.list_captures_missing_any_embeddings(
            [(embedder.model_name, embedder.model_version) for embedder in embedders],
            limit=limit,
            embedding_base=AUTO_INDEX_EMBEDDING_BASE,
        )
        attempted = len(rows)
        valid_batch = []
        for row in rows:
            capture_path = _capture_abs_path(row.get("filepath", ""))
            if not capture_path or not os.path.exists(capture_path):
                skipped += 1
                continue
            try:
                with open(capture_path, "rb") as f:
                    valid_batch.append((int(row["capture_id"]), f.read()))
            except Exception:
                skipped += 1
        if valid_batch:
            try:
                model_vectors = await asyncio.to_thread(
                    encode_image_batch_for_all_models,
                    [image_bytes for _, image_bytes in valid_batch],
                    embedders,
                )
                if not model_vectors:
                    raise RuntimeError("No retrieval models encoded successfully")
                for embedder, vectors in model_vectors:
                    if len(vectors) != len(valid_batch):
                        raise RuntimeError(
                            f"batch embedding size mismatch for {embedder.model_id} expected={len(valid_batch)} got={len(vectors)}"
                        )
                    upserted = db.upsert_capture_embeddings_batch(
                        [
                            (capture_id, vector)
                            for (capture_id, _), vector in zip(valid_batch, vectors)
                        ],
                        embedder.model_name,
                        embedder.model_version,
                        embedding_base=str(
                            getattr(embedder, "embedding_base", AUTO_INDEX_EMBEDDING_BASE)
                        ),
                    )
                    model_key = f"{embedder.model_name}:{embedder.model_version}"
                    indexed_by_model[model_key] = indexed_by_model.get(model_key, 0) + int(
                        upserted
                    )
                indexed += len(valid_batch)
            except Exception:
                for capture_id, image_bytes in valid_batch:
                    try:
                        for embedder in embedders:
                            vector = await asyncio.to_thread(
                                embedder.encode_image_bytes, image_bytes
                            )
                            db.upsert_capture_embedding(
                                capture_id,
                                embedder.model_name,
                                embedder.model_version,
                                vector,
                                embedding_base=str(
                                    getattr(
                                        embedder,
                                        "embedding_base",
                                        AUTO_INDEX_EMBEDDING_BASE,
                                    )
                                ),
                            )
                            model_key = f"{embedder.model_name}:{embedder.model_version}"
                            indexed_by_model[model_key] = indexed_by_model.get(
                                model_key, 0
                            ) + 1
                        indexed += 1
                    except Exception:
                        skipped += 1
        return {
            "attempted": attempted,
            "indexed": indexed,
            "skipped": skipped,
            "indexed_by_model": indexed_by_model,
        }
    finally:
        db.close()


async def _auto_index_loop():
    if not AUTO_INDEX_ENABLED:
        return
    await asyncio.sleep(3.0)
    while True:
        if auto_index_stop_event and auto_index_stop_event.is_set():
            return
        try:
            summary = await _index_missing_embeddings_once(AUTO_INDEX_BATCH_SIZE)
            if summary.get("indexed", 0) > 0:
                log.info(
                    "Auto-index embeddings indexed=%s attempted=%s skipped=%s by_model=%s",
                    summary.get("indexed", 0),
                    summary.get("attempted", 0),
                    summary.get("skipped", 0),
                    summary.get("indexed_by_model", {}),
                )
        except Exception as exc:
            log.warning("Auto-index loop error: %s", exc)

        try:
            if auto_index_stop_event:
                await asyncio.wait_for(
                    auto_index_stop_event.wait(), timeout=AUTO_INDEX_INTERVAL_SECONDS
                )
            else:
                await asyncio.sleep(AUTO_INDEX_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            continue


def _get_oneshot_lock() -> asyncio.Lock:
    global oneshot_lock
    if oneshot_lock is None:
        oneshot_lock = asyncio.Lock()
    return oneshot_lock


def _point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
    """
    Ray-casting point-in-polygon test.
    Polygon vertices are (lat, lon). Returns True if point is inside polygon.
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    x = lon
    y = lat
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _capture_profile_settings(profile_name: str) -> dict:
    profile = CAPTURE_PROFILES.get(profile_name)
    if profile:
        return profile
    return CAPTURE_PROFILES["base"]


# ─── Existing endpoints ───────────────────────────────────────────────────

@app.get("/api/panoramas")
async def get_panoramas():
    """Return all panoramas as GeoJSON."""
    db = get_db()
    geojson = db.get_panoramas_geojson()
    db.close()
    return JSONResponse(geojson)


@app.get("/api/panoramas/bbox")
async def get_panoramas_bbox(
    min_lat: float = Query(...),
    min_lon: float = Query(...),
    max_lat: float = Query(...),
    max_lon: float = Query(...),
    zoom: int = Query(14),
    limit: int = Query(6000),
    cluster_zoom_threshold: int = Query(16),
):
    if min_lat > max_lat or min_lon > max_lon:
        raise HTTPException(status_code=400, detail="Invalid bounding box")
    db = get_db()
    try:
        limit = max(100, min(10000, int(limit)))
        zoom = max(1, min(22, int(zoom)))
        if zoom < int(cluster_zoom_threshold):
            rows = db.get_panoramas_bbox_clusters(
                min_lat=min_lat,
                min_lon=min_lon,
                max_lat=max_lat,
                max_lon=max_lon,
                zoom=zoom,
                limit=limit,
            )
            features = []
            for row in rows:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [float(row["lon"]), float(row["lat"])],
                        },
                        "properties": {
                            "cluster": True,
                            "point_count": int(row.get("point_count", 0)),
                            "sample_panorama_id": int(row.get("sample_panorama_id", 0)),
                            "timestamp": row.get("newest_ts"),
                        },
                    }
                )
            return JSONResponse(
                {
                    "type": "FeatureCollection",
                    "features": features,
                    "meta": {
                        "mode": "clusters",
                        "zoom": zoom,
                        "returned": len(features),
                        "limit": limit,
                    },
                }
            )
        rows = db.get_panoramas_bbox_points(
            min_lat=min_lat,
            min_lon=min_lon,
            max_lat=max_lat,
            max_lon=max_lon,
            limit=limit,
        )
        features = []
        for row in rows:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["lon"]), float(row["lat"])],
                    },
                    "properties": {
                        "id": int(row["id"]),
                        "pano_id": row.get("pano_id"),
                        "heading": float(row.get("heading", 0.0)),
                        "timestamp": row.get("timestamp"),
                        "capture_count": int(row.get("capture_count", 0)),
                        "cluster": False,
                    },
                }
            )
        return JSONResponse(
            {
                "type": "FeatureCollection",
                "features": features,
                "meta": {
                    "mode": "points",
                    "zoom": zoom,
                    "returned": len(features),
                    "limit": limit,
                },
            }
        )
    finally:
        db.close()


@app.get("/api/stats")
async def get_stats():
    """Return crawl statistics."""
    db = get_db()
    stats = db.get_stats()
    db.close()
    return JSONResponse(stats)


@app.get("/api/panorama/{panorama_id}")
async def get_panorama_detail(panorama_id: int):
    """Return captures for a specific panorama."""
    db = get_db()
    captures = db.get_captures_for_panorama(panorama_id)
    db.close()
    for capture in captures:
        raw_path = capture.get("filepath", "")
        abs_path = _capture_abs_path(raw_path)
        capture["web_path"] = (
            _capture_web_path(raw_path)
            if abs_path and os.path.exists(abs_path)
            else ""
        )
    return JSONResponse(captures)


@app.get("/api/queue")
async def get_queue_stats():
    """Return seed task queue stats for worker monitoring."""
    db = get_db()
    stats = db.get_seed_task_stats()
    db.close()
    return JSONResponse(stats)




app.include_router(
    create_retrieval_router(
        get_db=get_db,
        capture_web_path=_capture_web_path,
        capture_abs_path=_capture_abs_path,
    )
)


@app.post("/api/capture-once")
async def capture_once(req: OneShotRequest):
    """Run a one-shot capture at a clicked map coordinate."""
    if not (-90 <= req.lat <= 90 and -180 <= req.lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid lat/lon")

    lock = _get_oneshot_lock()
    if lock.locked():
        raise HTTPException(
            status_code=409, detail="Another one-shot capture is already running"
        )

    async with lock:
        cmd = [
            sys.executable,
            "-m",
            "worker.crawler",
            "--lat",
            f"{req.lat:.7f}",
            "--lon",
            f"{req.lon:.7f}",
            "--max",
            "1",
            "--radius",
            "0.2",
            "--strategy",
            "bfs",
            "--headless",
            "--dedup-radius",
            str(config.DEDUP_RADIUS_METERS),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise HTTPException(status_code=504, detail="One-shot capture timed out")

        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "One-shot capture failed",
                    "command": " ".join(cmd),
                    "cwd": BASE_DIR,
                    "stdout_tail": _tail_text(stdout),
                    "stderr_tail": _tail_text(stderr),
                },
            )

        return JSONResponse(
            {
                "ok": True,
                "lat": req.lat,
                "lon": req.lon,
                "stdout_tail": _tail_text(stdout, max_chars=500),
            }
        )


# ─── New scan-area endpoints ──────────────────────────────────────────────

@app.post("/api/scan-area")
async def scan_area(req: ScanAreaRequest):
    """Dispatch a scan/enrich/fill job for a selected area."""
    if req.min_lat >= req.max_lat or req.min_lon >= req.max_lon:
        raise HTTPException(status_code=400, detail="Invalid bounding box")
    if not (1 <= req.num_workers <= 32):
        raise HTTPException(status_code=400, detail="num_workers must be 1-32")
    if req.step_meters < 5:
        raise HTTPException(status_code=400, detail="step_meters must be >= 5")
    job_type = str(req.job_type or "scan").strip().lower()
    if job_type not in {"scan", "enrich", "fill"}:
        raise HTTPException(status_code=400, detail="job_type must be scan|enrich|fill")
    capture_profile = str(req.capture_profile or "high_v1").strip().lower() or "high_v1"
    if capture_profile not in CAPTURE_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"capture_profile must be one of: {', '.join(sorted(CAPTURE_PROFILES.keys()))}",
        )
    profile_cfg = _capture_profile_settings(capture_profile)
    effective_capture_profile = capture_profile
    profile_cfg = _capture_profile_settings(effective_capture_profile)
    profile_views = [
        (float(heading), float(pitch))
        for pitch in profile_cfg["pitches"]
        for heading in profile_cfg["headings"]
    ]

    polygon_coords = req.polygon_coords or []
    polygon_points: List[Tuple[float, float]] = []
    if polygon_coords:
        try:
            polygon_points = [(float(p[0]), float(p[1])) for p in polygon_coords]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid polygon_coords format")
        if len(polygon_points) < 3:
            raise HTTPException(status_code=400, detail="polygon_coords requires at least 3 points")

    db = get_db()
    raw_points: List[Tuple[float, float]] = []
    polygon_filtered_points: List[Tuple[float, float]] = []
    land_points: List[Tuple[float, float]] = []
    missing_panoramas = 0
    missing_views_total = 0
    water_removed = 0
    polygon_removed = 0
    gap_removed = 0
    try:
        if job_type in {"scan", "fill"}:
            raw_points = generate_grid(
                req.min_lat, req.min_lon, req.max_lat, req.max_lon, req.step_meters
            )
            if not raw_points:
                raise HTTPException(status_code=400, detail="Bounding box produced zero seeds")

            if polygon_points:
                polygon_filtered_points = [
                    (lat, lon)
                    for lat, lon in raw_points
                    if _point_in_polygon(lat, lon, polygon_points)
                ]
            else:
                polygon_filtered_points = raw_points
            polygon_removed = len(raw_points) - len(polygon_filtered_points)

            land_points = filter_water_points(polygon_filtered_points)
            water_removed = len(polygon_filtered_points) - len(land_points)

            if job_type == "fill":
                existing_rows = db.get_panoramas_in_bbox(
                    req.min_lat, req.min_lon, req.max_lat, req.max_lon
                )
                existing_points = [
                    (float(row["lat"]), float(row["lon"])) for row in existing_rows
                ]
                gap_kept: List[Tuple[float, float]] = []
                for lat, lon in land_points:
                    is_far_enough = True
                    for existing_lat, existing_lon in existing_points:
                        if (
                            _haversine_m(lat, lon, existing_lat, existing_lon)
                            <= float(req.fill_gap_meters)
                        ):
                            is_far_enough = False
                            break
                    if is_far_enough:
                        gap_kept.append((lat, lon))
                gap_removed = len(land_points) - len(gap_kept)
                land_points = gap_kept
        else:
            panorama_rows = db.get_panoramas_in_bbox(
                req.min_lat, req.min_lon, req.max_lat, req.max_lon
            )
            if polygon_points:
                panorama_rows = [
                    row
                    for row in panorama_rows
                    if _point_in_polygon(float(row["lat"]), float(row["lon"]), polygon_points)
                ]
            if not panorama_rows:
                raise HTTPException(
                    status_code=400,
                    detail="No existing panoramas found in selected area to enrich",
                )
            panorama_ids = [int(row["id"]) for row in panorama_rows]
            missing_by_panorama = db.get_missing_views_for_panoramas(
                panorama_ids,
                required_views=profile_views,
                capture_profile=effective_capture_profile,
            )
            selected_rows = []
            for row in panorama_rows:
                missing = missing_by_panorama.get(int(row["id"]), [])
                if req.enrich_missing_only and not missing:
                    continue
                selected_rows.append(row)
                missing_panoramas += 1 if missing else 0
                missing_views_total += len(missing)
            land_points = [
                (float(row["lat"]), float(row["lon"]))
                for row in selected_rows
            ]
            raw_points = land_points
            polygon_filtered_points = land_points
    finally:
        db.close()

    if not land_points:
        if job_type == "fill":
            raise HTTPException(
                status_code=400,
                detail="No fill candidates remained after gap filtering",
            )
        if job_type == "enrich":
            raise HTTPException(
                status_code=400,
                detail="No un-enriched panoramas found for selected profile/area",
            )
        raise HTTPException(
            status_code=400,
            detail=f"All {len(raw_points)} seeds are in water; nothing to scan",
        )

    scan_id = uuid.uuid4().hex[:12]
    _append_scan_log(
        scan_id,
        f"job requested type={job_type} mode={req.mode} profile={effective_capture_profile} "
        f"bbox=({req.min_lat},{req.min_lon},{req.max_lat},{req.max_lon}) workers={req.num_workers} "
        f"step_m={req.step_meters} dedup_m={req.dedup_radius} fill_gap_m={req.fill_gap_meters} "
        f"raw={len(raw_points)} polygon_filtered={len(polygon_filtered_points)} "
        f"polygon_removed={polygon_removed} water_filtered={water_removed} gap_removed={gap_removed} "
        f"targets={len(land_points)} missing_panoramas={missing_panoramas} missing_views={missing_views_total}",
    )

    local_kwargs = {
        "job_kind": job_type,
        "capture_profile": effective_capture_profile,
        "headings": profile_cfg["headings"],
        "pitches": profile_cfg["pitches"],
        "missing_only": bool(req.enrich_missing_only and job_type == "enrich"),
        "skip_location_dedup": bool(job_type == "enrich"),
        "allow_existing_panorama": bool(job_type == "enrich"),
    }
    modal_kwargs = {
        "capture_profile": effective_capture_profile,
        "capture_kind": job_type,
        "headings": profile_cfg["headings"],
        "pitches": profile_cfg["pitches"],
        "missing_only": bool(req.enrich_missing_only and job_type == "enrich"),
    }
    if req.mode == "modal":
        result = await _dispatch_modal_workers(
            scan_id,
            land_points,
            req.num_workers,
            req.dedup_radius,
            job_type=job_type,
            modal_options=modal_kwargs,
        )
    else:
        result = await _dispatch_local_workers(
            scan_id,
            land_points,
            req.num_workers,
            req.dedup_radius,
            job_type=job_type,
            worker_options=local_kwargs,
        )

    return JSONResponse({
        "scan_id": scan_id,
        "job_type": job_type,
        "capture_profile": effective_capture_profile,
        "total_seeds_generated": len(raw_points),
        "polygon_filtered_out": polygon_removed,
        "water_filtered": water_removed,
        "gap_filtered": gap_removed,
        "missing_panoramas": missing_panoramas,
        "missing_views": missing_views_total,
        "land_seeds": len(land_points),
        "scan_log_url": f"/api/scan-log/{scan_id}",
        **result,
    })


@app.get("/api/scan-status")
async def scan_status():
    """Return progress of active scans and overall queue stats."""
    db = get_db()
    task_stats = db.get_seed_task_stats()
    overall = db.get_stats()
    db.close()

    scans = []
    modal_summary = {
        "active_scans": 0,
        "running_scans": 0,
        "workers_total": 0,
        "workers_submitted": 0,
        "workers_running": 0,
        "workers_completed": 0,
        "workers_failed": 0,
        "workers_cancelled": 0,
        "retries_queued": 0,
        "retries_completed": 0,
        "retries_failed": 0,
        "panoramas_saved": 0,
        "captures_saved": 0,
        "embeddings_saved": 0,
        "embedding_errors": 0,
    }
    now_ts = time.time()
    for scan_id, info in list(active_scans.items()):
        alive = 0
        if info["mode"] == "local":
            new_pids = []
            for pid in info.get("pids", []):
                try:
                    os.kill(pid, 0)
                    alive += 1
                    new_pids.append(pid)
                except OSError:
                    pass
            info["pids"] = new_pids
            if not new_pids and info.get("status") == "running":
                info["status"] = "finished"
        elif info["mode"] == "modal":
            progress = info.get("modal_progress", {})
            workers_total = int(progress.get("workers_total", info.get("num_workers", 0)))
            workers_submitted = int(progress.get("workers_submitted", 0))
            workers_completed = int(progress.get("workers_completed", 0))
            workers_failed = int(progress.get("workers_failed", 0))
            workers_cancelled = int(progress.get("workers_cancelled", 0))
            alive = _reconcile_modal_scan_state(scan_id, info, now_ts)
            progress["workers_running"] = alive

            modal_summary["active_scans"] += 1
            if str(info.get("status", "")) in {"running", "stopping"}:
                modal_summary["running_scans"] += 1
            modal_summary["workers_total"] += workers_total
            modal_summary["workers_submitted"] += workers_submitted
            modal_summary["workers_running"] += alive
            modal_summary["workers_completed"] += workers_completed
            modal_summary["workers_failed"] += workers_failed
            modal_summary["workers_cancelled"] += workers_cancelled
            modal_summary["retries_queued"] += int(progress.get("retries_queued", 0))
            modal_summary["retries_completed"] += int(progress.get("retries_completed", 0))
            modal_summary["retries_failed"] += int(progress.get("retries_failed", 0))

            modal_summary["panoramas_saved"] += int(progress.get("panoramas_saved", 0))
            modal_summary["captures_saved"] += int(progress.get("captures_saved", 0))
            modal_summary["embeddings_saved"] += int(progress.get("embeddings_saved", 0))
            modal_summary["embedding_errors"] += int(progress.get("embedding_errors", 0))

        scans.append({
            "scan_id": scan_id,
            "mode": info["mode"],
            "job_type": info.get("job_type", "scan"),
            "workers_requested": info["num_workers"],
            "workers_alive": alive,
            "status": info.get("status", "running"),
            "seeds_submitted": info.get("seeds_submitted", 0),
            "scan_log_url": info.get("scan_log_url", f"/api/scan-log/{scan_id}"),
            "modal_progress": info.get("modal_progress", {}),
            "result": info.get("result", {}),
        })

    return JSONResponse({
        "queue": task_stats,
        "modal": modal_summary,
        "panoramas": overall["total_panoramas"],
        "captures": overall["total_captures"],
        "scans": scans,
    })


@app.get("/api/scan-log/{scan_id}")
async def get_scan_log(scan_id: str):
    path = _scan_log_path(scan_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Scan log not found")
    with open(path, "r", encoding="utf-8") as f:
        body = f.read()
    return PlainTextResponse(body)


@app.get("/api/scan-log/{scan_id}/download")
async def download_scan_log(scan_id: str):
    path = _scan_log_path(scan_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Scan log not found")
    return FileResponse(path=path, filename=f"scan_{scan_id}.log", media_type="text/plain")


@app.post("/api/scan-stop")
async def scan_stop(req: ScanStopRequest):
    """Kill active scan workers. If scan_id is None, stop all scans."""
    killed = 0
    modal_stop_signals = 0
    target_ids = [req.scan_id] if req.scan_id else list(active_scans.keys())

    for sid in target_ids:
        info = active_scans.get(sid)
        if not info:
            continue
        _append_scan_log(sid, "scan stop requested")
        if info["mode"] == "local":
            signaled_pids = []
            for pid in info.get("pids", []):
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed += 1
                    signaled_pids.append(pid)
                except OSError:
                    pass
            if signaled_pids:
                time.sleep(0.2)
            for pid in signaled_pids:
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            info["pids"] = []
            info["status"] = "stopped"
            _append_scan_log(sid, "scan marked stopped")
        elif info["mode"] == "modal":
            info["stop_requested"] = True
            stop_fn = info.get("request_stop")
            if callable(stop_fn):
                try:
                    stop_fn()
                    modal_stop_signals += 1
                except Exception:
                    pass
            info["status"] = "stopping"
            _append_scan_log(sid, "modal stop signal sent")

    return JSONResponse(
        {
            "stopped": killed,
            "modal_stop_signals": modal_stop_signals,
            "scan_ids": target_ids,
        }
    )


# ─── Worker dispatch helpers ──────────────────────────────────────────────

async def _dispatch_local_workers(
    scan_id: str,
    points: List[tuple],
    num_workers: int,
    dedup_radius: float,
    job_type: str = "scan",
    worker_options: Optional[dict] = None,
) -> dict:
    """Write a temp seeds CSV, insert into DB, spawn N local batch_crawler processes."""
    worker_options = worker_options or {}
    seeds_filename = f"scan_{scan_id}.csv"
    seeds_path = os.path.join(SEEDS_DIR, seeds_filename)

    csv_bytes = _make_seeds_csv_bytes(points)
    with open(seeds_path, "wb") as f:
        f.write(csv_bytes)

    # Pre-insert seeds into DB so workers can claim them
    db = get_db()
    inserted = db.queue_seed_points(points)
    db.close()
    _append_scan_log(
        scan_id,
        f"local dispatch: job_type={job_type} requested_workers={num_workers} seeds={len(points)} inserted={inserted}",
    )

    pids = []
    for i in range(num_workers):
        cmd = [
            sys.executable,
            "-m", "worker.batch_crawler",
            "--seeds", seeds_path,
            "--max", str(max(1, len(points) // num_workers + 10)),
            "--dedup-radius", str(dedup_radius),
            "--headless",
            "--worker-id", f"{scan_id}-w{i}",
            "--lease-seconds", "300",
            "--job-kind", str(worker_options.get("job_kind", job_type)),
            "--capture-profile", str(worker_options.get("capture_profile", "base")),
        ]
        headings = worker_options.get("headings") or []
        pitches = worker_options.get("pitches") or []
        if headings:
            cmd.extend(["--headings-csv", ",".join(str(h) for h in headings)])
        if pitches:
            cmd.extend(["--pitches-csv", ",".join(str(p) for p in pitches)])
        if worker_options.get("missing_only"):
            cmd.append("--missing-only")
        if worker_options.get("skip_location_dedup"):
            cmd.append("--skip-location-dedup")
        if worker_options.get("allow_existing_panorama"):
            cmd.append("--allow-existing-panorama")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        pids.append(proc.pid)
        _append_scan_log(scan_id, f"local worker started worker_index={i} pid={proc.pid}")

    active_scans[scan_id] = {
        "mode": "local",
        "job_type": job_type,
        "num_workers": num_workers,
        "pids": pids,
        "status": "running",
        "seeds_submitted": len(points),
        "scan_log_url": f"/api/scan-log/{scan_id}",
    }

    return {
        "mode": "local",
        "job_type": job_type,
        "workers_spawned": len(pids),
        "seeds_inserted": inserted,
        "scan_log_url": f"/api/scan-log/{scan_id}",
    }


async def _dispatch_modal_workers(
    scan_id: str,
    points: List[tuple],
    num_workers: int,
    dedup_radius: float,
    job_type: str = "scan",
    modal_options: Optional[dict] = None,
) -> dict:
    """
    Dispatch Modal workers, wait for results, and save everything into the
    local DB and captures directory.  Runs in a background thread so the
    event loop stays responsive.
    """
    try:
        from worker import modal_worker
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="modal_worker module not available; install modal and configure",
        )

    modal_options = modal_options or {}
    active_scans[scan_id] = {
        "mode": "modal",
        "job_type": job_type,
        "num_workers": num_workers,
        "status": "running",
        "stop_requested": False,
        "seeds_submitted": len(points),
        "modal_environment": MODAL_ENVIRONMENT,
        "started_at_ts": time.time(),
        "scan_log_url": f"/api/scan-log/{scan_id}",
        "modal_progress": {
            "workers_total": 0,
            "workers_submitted": 0,
            "workers_completed": 0,
            "workers_failed": 0,
            "workers_cancelled": 0,
            "workers_running": 0,
            "retries_queued": 0,
            "retries_completed": 0,
            "retries_failed": 0,
            "panoramas_saved": 0,
            "captures_saved": 0,
            "embeddings_saved": 0,
            "embedding_errors": 0,
            "last_event": "queued",
            "last_event_ts": time.time(),
            "stale": False,
            "stale_seconds": 0,
            "jobs": {},
        },
    }
    _append_scan_log(
        scan_id,
        f"modal dispatch queued job_type={job_type} workers={num_workers} seeds={len(points)} env={MODAL_ENVIRONMENT}",
    )
    stop_state = {"requested": False}

    def _request_stop():
        stop_state["requested"] = True

    active_scans[scan_id]["request_stop"] = _request_stop

    def _progress_update(event: dict):
        info = active_scans.get(scan_id)
        if not info:
            return
        progress = info.setdefault("modal_progress", {})
        progress["last_event"] = event.get("event", progress.get("last_event", "unknown"))
        progress["last_event_ts"] = time.time()
        progress["stale"] = False
        progress["stale_seconds"] = 0
        progress["workers_total"] = int(event.get("workers_total", progress.get("workers_total", 0)))
        progress["workers_submitted"] = int(
            event.get("workers_submitted", progress.get("workers_submitted", 0))
        )
        progress["workers_completed"] = int(
            event.get("workers_completed", progress.get("workers_completed", 0))
        )
        progress["workers_failed"] = int(
            event.get("workers_failed", progress.get("workers_failed", 0))
        )
        progress["workers_cancelled"] = int(
            event.get("workers_cancelled", progress.get("workers_cancelled", 0))
        )
        progress["workers_running"] = max(
            0,
            progress["workers_submitted"]
            - progress["workers_completed"]
            - progress["workers_failed"]
            - progress["workers_cancelled"],
        )
        progress["retries_queued"] = int(
            event.get("retries_queued", progress.get("retries_queued", 0))
        )
        progress["retries_completed"] = int(
            event.get("retries_completed", progress.get("retries_completed", 0))
        )
        progress["retries_failed"] = int(
            event.get("retries_failed", progress.get("retries_failed", 0))
        )
        progress["panoramas_saved"] = progress.get("panoramas_saved", 0)
        progress["captures_saved"] = progress.get("captures_saved", 0)
        progress["embeddings_saved"] = progress.get("embeddings_saved", 0)
        progress["embedding_errors"] = progress.get("embedding_errors", 0)

        event_type = event.get("event")
        if event_type in {"worker_completed", "worker_failed", "worker_cancelled"}:
            jobs = progress.setdefault("jobs", {})
            job_id = str(event.get("job_id", f"job-{len(jobs)}"))
            jobs[job_id] = {
                "ok": bool(event.get("ok", event_type == "worker_completed")),
                "job_kind": event.get("job_kind", ""),
                "seed_count": int(event.get("seed_count", 0)),
                "locations_returned": int(event.get("locations_returned", 0)),
                "panoramas_saved": int(event.get("panoramas_saved", 0)),
                "captures_saved": int(event.get("captures_saved", 0)),
                "embeddings_saved": int(event.get("embeddings_saved", 0)),
                "embedding_errors": int(event.get("embedding_errors", 0)),
                "error": event.get("error", "cancelled" if event_type == "worker_cancelled" else ""),
                "stats": event.get("stats", {}),
            }
            progress["panoramas_saved"] += int(event.get("panoramas_saved", 0))
            progress["captures_saved"] += int(event.get("captures_saved", 0))
            progress["embeddings_saved"] += int(event.get("embeddings_saved", 0))
            progress["embedding_errors"] += int(event.get("embedding_errors", 0))

        if event_type == "retry_enqueued":
            seed = event.get("seed", {})
            _append_scan_log(
                scan_id,
                f"retry queued lat={seed.get('lat')} lon={seed.get('lon')} reason={event.get('reason', 'unknown')}",
            )
        elif event_type == "worker_submitted":
            _append_scan_log(
                scan_id,
                f"worker submitted job_id={event.get('job_id')} kind={event.get('job_kind')} call_id={event.get('call_id')}",
            )
        elif event_type == "worker_completed":
            _append_scan_log(
                scan_id,
                f"worker completed job_id={event.get('job_id')} kind={event.get('job_kind')} "
                f"locations={event.get('locations_returned', 0)} panos_saved={event.get('panoramas_saved', 0)} "
                f"captures_saved={event.get('captures_saved', 0)} embeddings_saved={event.get('embeddings_saved', 0)} "
                f"embedding_errors={event.get('embedding_errors', 0)}",
            )
        elif event_type == "worker_failed":
            err = str(event.get("error", "")).strip() or "unknown-worker-error"
            _append_scan_log(
                scan_id,
                f"worker failed job_id={event.get('job_id')} kind={event.get('job_kind')} error={err}",
            )
        elif event_type == "worker_cancelled":
            _append_scan_log(
                scan_id,
                f"worker cancelled job_id={event.get('job_id')} kind={event.get('job_kind')}",
            )
        elif event_type == "all_done":
            _append_scan_log(
                scan_id,
                f"all done workers_total={event.get('workers_total', 0)} workers_completed={event.get('workers_completed', 0)} "
                f"workers_failed={event.get('workers_failed', 0)} workers_cancelled={event.get('workers_cancelled', 0)} "
                f"retries_queued={event.get('retries_queued', 0)} embeddings_saved={event.get('total_embeddings_saved', 0)} "
                f"embedding_errors={event.get('total_embedding_errors', 0)}",
            )

    async def _run_modal_in_background():
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                modal_worker.dispatch_and_collect,
                points,
                num_workers,
                config.DATABASE_URL,
                config.CAPTURES_DIR,
                MODAL_ENVIRONMENT,
                _progress_update,
                lambda: bool(stop_state["requested"]),
                modal_options.get("headings"),
                modal_options.get("pitches"),
                modal_options.get("capture_profile", "base"),
                modal_options.get("capture_kind", job_type),
                bool(modal_options.get("missing_only", False)),
            )
            info = active_scans.get(scan_id, {})
            was_stopped = bool(stop_state["requested"]) or bool(result.get("stopped"))
            info["status"] = "stopped" if was_stopped else "finished"
            info["result"] = result
            info.pop("request_stop", None)
            _append_scan_log(
                scan_id,
                (
                    f"modal stopped panoramas_saved={result.get('total_panoramas_saved', 0)} "
                    f"captures_saved={result.get('total_captures_saved', 0)} "
                    f"embeddings_saved={result.get('total_embeddings_saved', 0)} "
                    f"embedding_errors={result.get('total_embedding_errors', 0)}"
                    if was_stopped
                    else f"modal finished panoramas_saved={result.get('total_panoramas_saved', 0)} "
                    f"captures_saved={result.get('total_captures_saved', 0)} "
                    f"embeddings_saved={result.get('total_embeddings_saved', 0)} "
                    f"embedding_errors={result.get('total_embedding_errors', 0)}"
                ),
            )
        except Exception as e:
            err = str(e).strip() or repr(e)
            info = active_scans.get(scan_id, {})
            if stop_state["requested"]:
                info["status"] = "stopped"
                _append_scan_log(scan_id, f"modal stopped during shutdown error={err}")
            else:
                info["status"] = f"failed: {err}"
                _append_scan_log(scan_id, f"modal failed error={err}")
            info.pop("request_stop", None)

    asyncio.ensure_future(_run_modal_in_background())

    return {
        "mode": "modal",
        "job_type": job_type,
        "workers_spawned": num_workers,
        "seeds_queued": len(points),
        "modal_environment": MODAL_ENVIRONMENT,
        "scan_log_url": f"/api/scan-log/{scan_id}",
    }


# ─── Index page ────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    if os.path.exists(FRONTEND_DIST_INDEX):
        return FileResponse(FRONTEND_DIST_INDEX, media_type="text/html")
    if os.path.exists(BACKEND_FALLBACK_INDEX):
        return FileResponse(BACKEND_FALLBACK_INDEX, media_type="text/html")
    return PlainTextResponse("GeoSpy backend is running, but frontend assets are missing.")


def run() -> None:
    import uvicorn

    host = os.getenv("GEOSPY_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("GEOSPY_SERVER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
