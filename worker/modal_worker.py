"""
Modal.com worker for parallelized Street View scraping.

Self-contained worker: no project imports needed inside Modal containers.
Results are always written into the local DB/captures directory by the caller.
"""

import csv
import io
import logging
import os
import threading
from collections import deque
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

try:
    from env_bootstrap import load_project_env
except ModuleNotFoundError:
    load_project_env = None

if load_project_env is not None:
    load_project_env()

import modal

LOG_LEVEL = os.getenv("GEOSPY_MODAL_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
log = logging.getLogger(__name__)

app = modal.App("geospy-crawler")
DEFAULT_MODAL_ENVIRONMENT = "google-map-walkers"
DEFAULT_BATCH_SIZE = 5
DEFAULT_EMBED_ON_INGEST = os.getenv("GEOSPY_EMBED_ON_INGEST", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DEFAULT_EMBED_BATCH_SIZE = max(1, int(os.getenv("GEOSPY_EMBED_BATCH_SIZE", "128")))

_modal_app_context_lock = threading.Lock()
_modal_app_context_manager = None
_modal_app_context_env: Optional[str] = None
_modal_app_context_refcount = 0

crawler_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget",
        "gnupg",
        "ca-certificates",
        "fonts-liberation",
        "libasound2",
        "libatk-bridge2.0-0",
        "libatk1.0-0",
        "libatspi2.0-0",
        "libcairo2",
        "libdbus-1-3",
        "libdrm2",
        "libgbm1",
        "libglib2.0-0",
        "libgtk-3-0",
        "libnspr4",
        "libnss3",
        "libpango-1.0-0",
        "libpangocairo-1.0-0",
        "libx11-6",
        "libx11-xcb1",
        "libxcb1",
        "libxcomposite1",
        "libxdamage1",
        "libxext6",
        "libxfixes3",
        "libxkbcommon0",
        "libxrandr2",
    )
    .pip_install("playwright>=1.40.0", "Pillow>=10.0.0")
    .run_commands("playwright install --with-deps chromium")
)

# Worker constants.
VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080
HEADINGS = [0, 90, 180, 270]
PITCH = 75.0
CAPTURE_DELAY = 2.0
BLACK_MEAN_THRESHOLD = 8.0
BLACK_DARK_RATIO_THRESHOLD = 0.98
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _chunk_points(points: List[Tuple[float, float]], size: int) -> List[List[Tuple[float, float]]]:
    size = max(1, int(size))
    chunks: List[List[Tuple[float, float]]] = []
    for i in range(0, len(points), size):
        chunks.append(points[i: i + size])
    return chunks


@contextmanager
def _shared_modal_app_context(environment_name: str):
    """
    Reuse a single Modal app context across concurrent dispatchers in this process.
    Prevents re-entrant `app.run(...)` crashes when multiple scans start together.
    """
    global _modal_app_context_manager
    global _modal_app_context_env
    global _modal_app_context_refcount

    with _modal_app_context_lock:
        if _modal_app_context_manager is None:
            manager = app.run(environment_name=environment_name)
            manager.__enter__()
            _modal_app_context_manager = manager
            _modal_app_context_env = environment_name
        elif _modal_app_context_env != environment_name:
            raise RuntimeError(
                "Modal app context already active in a different environment "
                f"({_modal_app_context_env} != {environment_name})."
            )
        _modal_app_context_refcount += 1

    try:
        yield
    finally:
        manager_to_close = None
        with _modal_app_context_lock:
            _modal_app_context_refcount = max(0, _modal_app_context_refcount - 1)
            if _modal_app_context_refcount == 0 and _modal_app_context_manager is not None:
                manager_to_close = _modal_app_context_manager
                _modal_app_context_manager = None
                _modal_app_context_env = None
        if manager_to_close is not None:
            manager_to_close.__exit__(None, None, None)


@app.function(
    image=crawler_image,
    timeout=3600,
    cpu=2.0,
    memory=2048,
)
async def scrape_locations(job_payload: dict) -> dict:
    """
    Scrape a small set of coordinates and return data+stats.
    Returns:
        {
            "results": [...valid locations...],
            "failed_seeds": [{"lat","lon","reason"}, ...],
            "stats": {...}
        }
    """
    import asyncio
    import re
    from io import BytesIO
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    from PIL import Image, ImageStat
    from playwright.async_api import async_playwright

    worker_log = logging.getLogger("modal_worker.scrape")

    SV_PANO_RE = re.compile(r"!1s([A-Za-z0-9_-]{10,})!2e\d+")
    THUMB_PANO_RE = re.compile(r"panoid(?:=|%3D)([A-Za-z0-9_-]{10,})")

    def fetch_image(url: str) -> bytes:
        req = Request(url, headers={"User-Agent": USER_AGENT, "Referer": "https://www.google.com/maps/"})
        try:
            with urlopen(req, timeout=20) as resp:
                ct = resp.headers.get("Content-Type", "")
                body = resp.read()
        except HTTPError as e:
            raise RuntimeError(f"http-{e.code}") from e
        except URLError as e:
            raise RuntimeError(f"url-error-{e.reason}") from e
        except Exception as e:
            raise RuntimeError(f"request-error-{e}") from e
        if "image" not in ct.lower() or len(body) < 128:
            raise RuntimeError("non-image-response")
        return body

    def analyze_image_quality(image_bytes: bytes) -> Tuple[bool, float, float]:
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                gray = img.convert("L").resize((160, 90))
                stat = ImageStat.Stat(gray)
                mean_val = float(stat.mean[0]) if stat.mean else 0.0
                hist = gray.histogram()
                total = max(1, sum(hist))
                dark_pixels = sum(hist[:8])  # near-black pixels
                dark_ratio = dark_pixels / float(total)
                is_black = (
                    mean_val <= BLACK_MEAN_THRESHOLD
                    and dark_ratio >= BLACK_DARK_RATIO_THRESHOLD
                )
                return is_black, mean_val, dark_ratio
        except Exception as e:
            raise RuntimeError(f"quality-check-failed-{e}") from e

    def parse_url(url: str) -> Tuple[Optional[float], Optional[float], float]:
        if "3a," not in url:
            return None, None, 0.0
        m = re.search(r"@(-?\d+\.?\d*),(-?\d+\.?\d*),3a,[\d.]+y,([\d.]+)h", url)
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))
        m = re.search(r"@(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
        if m:
            return float(m.group(1)), float(m.group(2)), 0.0
        return None, None, 0.0

    def extract_pano_id(url_or_html: str) -> Optional[str]:
        m = SV_PANO_RE.search(url_or_html)
        if not m:
            m = THUMB_PANO_RE.search(url_or_html)
        return m.group(1) if m else None

    async def wait_for_sv(page, timeout_ms=10000) -> bool:
        try:
            await page.wait_for_function(
                """() => {
                    const href = window.location.href || '';
                    if (!href.includes(',3a,')) return false;
                    return /!1s[A-Za-z0-9_-]{10,}!2e\\d+/.test(href)
                        || /panoid(?:=|%3D)[A-Za-z0-9_-]{10,}/.test(href)
                        || (document.body?.innerText || '').includes('Street View');
                }""",
                timeout=timeout_ms,
            )
            return True
        except Exception:
            return ",3a," in page.url

    async def safe_goto(page, url, timeout_ms=12000) -> bool:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            return True
        except Exception as e:
            worker_log.warning("goto failed url=%s error=%s", url, e)
            return False

    coords_raw = job_payload.get("coords", [])
    coords: List[Tuple[float, float]] = [
        (float(pair[0]), float(pair[1]))
        for pair in coords_raw
        if isinstance(pair, (list, tuple)) and len(pair) >= 2
    ]
    headings = [float(h) for h in (job_payload.get("headings") or HEADINGS)]
    pitches = [float(p) for p in (job_payload.get("pitches") or [PITCH])]
    capture_profile = str(job_payload.get("capture_profile") or "base")
    capture_kind = str(job_payload.get("capture_kind") or "scan")
    results: List[dict] = []
    failed_seeds: List[dict] = []
    stats = {
        "coords_total": len(coords),
        "coords_ok": 0,
        "street_view_unavailable": 0,
        "url_parse_failed": 0,
        "pano_id_missing": 0,
        "thumbnail_request_failures": 0,
        "tile_request_failures": 0,
        "black_frames_rejected": 0,
        "all_headings_invalid": 0,
        "unexpected_errors": 0,
    }
    worker_log.info("worker starting coords=%d", len(coords))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            user_agent=USER_AGENT,
        )
        page = await context.new_page()

        # Warm page and consent once.
        if coords:
            first_lat, first_lon = coords[0]
            url = (
                "https://www.google.com/maps"
                f"?layer=c&cbll={first_lat:.7f},{first_lon:.7f}&cbp=12,0,0,0,0"
            )
            await safe_goto(page, url, timeout_ms=18000)
            await asyncio.sleep(1.5)
            try:
                for sel in [
                    'button[aria-label="Accept all"]',
                    'button:has-text("Accept all")',
                    'button:has-text("I agree")',
                ]:
                    btn = page.locator(sel).first
                    if await btn.is_visible(timeout=2000):
                        await btn.click()
                        await asyncio.sleep(1)
                        break
            except Exception:
                pass

        for idx, (lat, lon) in enumerate(coords, start=1):
            seed_reason = ""
            worker_log.info("seed %d/%d lat=%.6f lon=%.6f", idx, len(coords), lat, lon)
            try:
                sv_url = (
                    "https://www.google.com/maps"
                    f"?layer=c&cbll={lat:.7f},{lon:.7f}&cbp=12,0,0,0,0"
                )
                await safe_goto(page, sv_url, timeout_ms=12000)
                if not await wait_for_sv(page, timeout_ms=8000):
                    stats["street_view_unavailable"] += 1
                    seed_reason = "street-view-unavailable"
                    failed_seeds.append({"lat": lat, "lon": lon, "reason": seed_reason})
                    worker_log.warning("seed=%d reason=%s", idx, seed_reason)
                    continue

                await asyncio.sleep(CAPTURE_DELAY)
                actual_url = page.url
                actual_lat, actual_lon, heading = parse_url(actual_url)
                if actual_lat is None:
                    stats["url_parse_failed"] += 1
                    seed_reason = "url-parse-failed"
                    failed_seeds.append({"lat": lat, "lon": lon, "reason": seed_reason})
                    worker_log.warning("seed=%d reason=%s url=%s", idx, seed_reason, actual_url)
                    continue

                pano_id = extract_pano_id(actual_url)
                if not pano_id:
                    try:
                        html = await page.content()
                        pano_id = extract_pano_id(html)
                    except Exception:
                        pano_id = None
                if not pano_id:
                    stats["pano_id_missing"] += 1
                    seed_reason = "pano-id-missing"
                    failed_seeds.append({"lat": lat, "lon": lon, "reason": seed_reason})
                    worker_log.warning("seed=%d reason=%s url=%s", idx, seed_reason, actual_url)
                    continue

                captures = []
                for map_pitch in pitches:
                    thumb_pitch = max(-90.0, min(90.0, 90.0 - float(map_pitch)))
                    for h in headings:
                        try:
                            image_bytes = fetch_image(
                                "https://streetviewpixels-pa.googleapis.com/v1/thumbnail?"
                                + urlencode(
                                    {
                                        "cb_client": "maps_sv.tactile",
                                        "w": VIEWPORT_WIDTH,
                                        "h": VIEWPORT_HEIGHT,
                                        "pitch": float(thumb_pitch),
                                        "panoid": pano_id,
                                        "yaw": float(h),
                                    }
                                )
                            )
                        except Exception as e:
                            stats["thumbnail_request_failures"] += 1
                            worker_log.warning(
                                "seed=%d pano=%s heading=%s pitch=%s reason=thumbnail-request-failed error=%s",
                                idx,
                                pano_id,
                                h,
                                map_pitch,
                                e,
                            )
                            continue

                        try:
                            is_black, mean_val, dark_ratio = analyze_image_quality(image_bytes)
                        except Exception as e:
                            stats["thumbnail_request_failures"] += 1
                            worker_log.warning(
                                "seed=%d pano=%s heading=%s pitch=%s reason=quality-check-failed error=%s",
                                idx,
                                pano_id,
                                h,
                                map_pitch,
                                e,
                            )
                            continue

                        if is_black:
                            stats["black_frames_rejected"] += 1
                            worker_log.warning(
                                "seed=%d pano=%s heading=%s pitch=%s reason=black-frame mean=%.2f dark_ratio=%.3f",
                                idx,
                                pano_id,
                                h,
                                map_pitch,
                                mean_val,
                                dark_ratio,
                            )
                            continue

                        rounded_pitch = int(round(float(map_pitch)))
                        profile_slug = "".join(
                            c
                            if c.isalnum() or c in {"-", "_"}
                            else "-"
                            for c in (capture_profile or "base")
                        )
                        filename = (
                            f"h{int(round(float(h))):03d}.jpg"
                            if rounded_pitch == 75 and profile_slug == "base"
                            else f"h{int(round(float(h))):03d}_p{rounded_pitch:03d}_{profile_slug}.jpg"
                        )
                        captures.append(
                            {
                                "heading": float(h),
                                "pitch": float(map_pitch),
                                "capture_profile": capture_profile,
                                "capture_kind": capture_kind,
                                "filename": filename,
                                "image_bytes": image_bytes,
                                "width": VIEWPORT_WIDTH,
                                "height": VIEWPORT_HEIGHT,
                                "brightness_mean": mean_val,
                                "quality_reason": "ok",
                            }
                        )

                tile_bytes = b""
                try:
                    tile_bytes = fetch_image(
                        "https://streetviewpixels-pa.googleapis.com/v1/tile?"
                        + urlencode(
                            {
                                "cb_client": "maps_sv.tactile",
                                "panoid": pano_id,
                                "x": 0,
                                "y": 0,
                                "zoom": 1,
                                "nbt": 1,
                                "fover": 2,
                            }
                        )
                    )
                except Exception as e:
                    stats["tile_request_failures"] += 1
                    worker_log.warning(
                        "seed=%d pano=%s reason=tile-request-failed error=%s",
                        idx,
                        pano_id,
                        e,
                    )

                if not captures:
                    stats["all_headings_invalid"] += 1
                    seed_reason = "all-headings-invalid"
                    failed_seeds.append({"lat": lat, "lon": lon, "reason": seed_reason})
                    worker_log.warning(
                        "seed=%d pano=%s reason=%s snapped_lat=%.6f snapped_lon=%.6f",
                        idx,
                        pano_id,
                        seed_reason,
                        actual_lat,
                        actual_lon,
                    )
                    continue

                stats["coords_ok"] += 1
                results.append(
                    {
                        "lat": actual_lat,
                        "lon": actual_lon,
                        "pano_id": pano_id,
                        "heading": heading,
                        "pitch": float(pitches[0] if pitches else PITCH),
                        "capture_profile": capture_profile,
                        "capture_kind": capture_kind,
                        "source_url": actual_url,
                        "captures": captures,
                        "tile_bytes": tile_bytes,
                    }
                )
                worker_log.info(
                    "seed success seed=%d pano_id=%s captures=%d snapped_lat=%.6f snapped_lon=%.6f",
                    idx,
                    pano_id,
                    len(captures),
                    actual_lat,
                    actual_lon,
                )

            except Exception as e:
                stats["unexpected_errors"] += 1
                seed_reason = "unexpected-error"
                failed_seeds.append({"lat": lat, "lon": lon, "reason": seed_reason})
                worker_log.exception(
                    "seed=%d lat=%.6f lon=%.6f reason=%s error=%s",
                    idx,
                    lat,
                    lon,
                    seed_reason,
                    e,
                )

        await browser.close()

    worker_log.info(
        "worker finished coords_total=%d coords_ok=%d sv_unavailable=%d url_parse_failed=%d pano_id_missing=%d thumb_failures=%d tile_failures=%d black_rejected=%d all_headings_invalid=%d unexpected_errors=%d",
        stats["coords_total"],
        stats["coords_ok"],
        stats["street_view_unavailable"],
        stats["url_parse_failed"],
        stats["pano_id_missing"],
        stats["thumbnail_request_failures"],
        stats["tile_request_failures"],
        stats["black_frames_rejected"],
        stats["all_headings_invalid"],
        stats["unexpected_errors"],
    )
    return {"results": results, "failed_seeds": failed_seeds, "stats": stats}


def save_results_to_local_db(
    results: list[dict],
    db_path: str,
    captures_dir: str,
    capture_profile: str = "base",
    capture_kind: str = "scan",
    missing_only: bool = False,
    embed_on_ingest: bool = DEFAULT_EMBED_ON_INGEST,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> dict:
    """Write scrape results into local DB and captures directory."""
    from datetime import datetime

    from backend.app.embedding_ingest import CaptureEmbeddingIngestor
    from db.postgres_database import Capture, Database, Panorama

    os.makedirs(captures_dir, exist_ok=True)
    db = Database(db_path)
    saved_panos = 0
    saved_captures = 0
    embedding_ingestor = CaptureEmbeddingIngestor(
        db,
        enabled=embed_on_ingest,
        batch_size=embed_batch_size,
        logger=log,
    )

    for pano_data in results:
        pano_id_str = pano_data.get("pano_id") or ""
        dir_name = pano_id_str or f"{pano_data['lat']:.5f}_{pano_data['lon']:.5f}"
        pano_dir = os.path.join(captures_dir, dir_name)
        os.makedirs(pano_dir, exist_ok=True)

        if pano_data.get("tile_bytes"):
            with open(os.path.join(pano_dir, "tile_z1_x0_y0.jpg"), "wb") as f:
                f.write(pano_data["tile_bytes"])

        pano = Panorama(
            id=None,
            lat=pano_data["lat"],
            lon=pano_data["lon"],
            pano_id=pano_data.get("pano_id"),
            heading=pano_data["heading"],
            pitch=pano_data.get("pitch", PITCH),
            timestamp=datetime.utcnow().isoformat(),
            source_url=pano_data.get("source_url", ""),
        )
        pano_db_id, created = db.add_panorama_if_new(pano, dedup_radius_meters=25.0)
        if created:
            saved_panos += 1

        for cap in pano_data.get("captures", []):
            image_bytes = cap.get("image_bytes")
            if not image_bytes:
                continue
            filepath = os.path.join(pano_dir, cap["filename"])
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            capture_row = Capture(
                id=None,
                panorama_id=pano_db_id,
                heading=cap["heading"],
                pitch=float(cap.get("pitch", pano_data.get("pitch", PITCH))),
                filepath=filepath,
                width=cap["width"],
                height=cap["height"],
                capture_profile=str(
                    cap.get(
                        "capture_profile",
                        pano_data.get("capture_profile", capture_profile),
                    )
                ),
                capture_kind=str(
                    cap.get("capture_kind", pano_data.get("capture_kind", capture_kind))
                ),
                is_black_frame=0,
                quality_reason=cap.get("quality_reason", "ok"),
                brightness_mean=cap.get("brightness_mean"),
            )
            if missing_only:
                capture_id, created_capture = db.add_capture_if_missing(capture_row)
                if created_capture:
                    saved_captures += 1
            else:
                capture_id = db.add_capture(capture_row)
                saved_captures += 1
                created_capture = True

            if created_capture:
                embedding_ingestor.add_capture(int(capture_id), image_bytes)

    embedding_ingestor.close()
    db.close()
    return {
        "panoramas_saved": saved_panos,
        "captures_saved": saved_captures,
        "embeddings_saved": embedding_ingestor.saved_embeddings,
        "embedding_errors": embedding_ingestor.embed_errors,
    }


def dispatch_and_collect(
    points: List[Tuple[float, float]],
    num_workers: int,
    db_path: str,
    captures_dir: str,
    modal_environment: Optional[str] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    stop_callback: Optional[Callable[[], bool]] = None,
    headings: Optional[List[float]] = None,
    pitches: Optional[List[float]] = None,
    capture_profile: str = "base",
    capture_kind: str = "scan",
    missing_only: bool = False,
) -> dict:
    """
    Run Modal scrape in micro-batches, retry failed seeds individually, and
    write successful results locally as each job finishes.
    """
    seeds = [(float(lat), float(lon)) for lat, lon in points]
    initial_batches = _chunk_points(seeds, DEFAULT_BATCH_SIZE)
    pending_jobs = deque()
    job_counter = 0

    for batch in initial_batches:
        pending_jobs.append(
            {
                "job_id": f"batch-{job_counter}",
                "kind": "batch",
                "coords": batch,
            }
        )
        job_counter += 1

    workers_total = len(pending_jobs)
    workers_submitted = 0
    workers_completed = 0
    workers_failed = 0
    retries_queued = 0
    retries_completed = 0
    retries_failed = 0
    workers_cancelled = 0
    retried_seed_keys = set()
    failed_seed_reasons: Dict[str, str] = {}

    total_panos = 0
    total_captures = 0
    total_embeddings = 0
    total_embedding_errors = 0
    worker_results = []

    def emit(event: dict):
        if progress_callback:
            try:
                progress_callback(event)
            except Exception:
                pass

    def stop_requested() -> bool:
        if not stop_callback:
            return False
        try:
            return bool(stop_callback())
        except Exception:
            return False

    environment_name = (
        modal_environment
        or os.getenv("MODAL_ENVIRONMENT")
        or DEFAULT_MODAL_ENVIRONMENT
    )
    print(f"Starting Modal app context (env={environment_name})...")
    emit(
        {
            "event": "dispatch_started",
            "workers_total": workers_total,
            "workers_submitted": workers_submitted,
            "workers_completed": workers_completed,
            "workers_failed": workers_failed,
        }
    )

    max_parallel_workers = max(1, int(num_workers))
    active = deque()

    def _submit_next_job() -> bool:
        nonlocal workers_submitted
        if stop_requested() or not pending_jobs or len(active) >= max_parallel_workers:
            return False
        job = pending_jobs.popleft()
        worker_payload = {
            "coords": [[lat, lon] for lat, lon in job["coords"]],
            "headings": headings or HEADINGS,
            "pitches": pitches or [PITCH],
            "capture_profile": capture_profile,
            "capture_kind": capture_kind,
        }
        handle = scrape_locations.spawn(worker_payload)
        workers_submitted += 1
        active.append((job, handle))
        print(
            f"  Submitted {job['job_id']} kind={job['kind']} "
            f"seeds={len(job['coords'])} call_id={handle.object_id}"
        )
        emit(
            {
                "event": "worker_submitted",
                "job_id": job["job_id"],
                "job_kind": job["kind"],
                "seed_count": len(job["coords"]),
                "call_id": handle.object_id,
                "workers_total": workers_total,
                "workers_submitted": workers_submitted,
                "workers_completed": workers_completed,
                "workers_failed": workers_failed,
                "retries_queued": retries_queued,
                "retries_completed": retries_completed,
                "retries_failed": retries_failed,
            }
        )
        return True

    def _top_up_active_jobs() -> None:
        while _submit_next_job():
            pass

    try:
        with _shared_modal_app_context(environment_name):
            print("Modal app context active. Submitting workers...")
            _top_up_active_jobs()
            while active or pending_jobs:
                if stop_requested():
                    emit(
                        {
                            "event": "dispatch_stopping",
                            "workers_total": workers_total,
                            "workers_submitted": workers_submitted,
                            "workers_completed": workers_completed,
                            "workers_failed": workers_failed,
                            "workers_cancelled": workers_cancelled,
                        }
                    )
                    while active:
                        job, handle = active.popleft()
                        try:
                            handle.cancel(terminate_containers=True)
                        except Exception:
                            pass
                        workers_cancelled += 1
                        emit(
                            {
                                "event": "worker_cancelled",
                                "job_id": job["job_id"],
                                "job_kind": job["kind"],
                                "workers_total": workers_total,
                                "workers_submitted": workers_submitted,
                                "workers_completed": workers_completed,
                                "workers_failed": workers_failed,
                                "workers_cancelled": workers_cancelled,
                            }
                        )
                    break

                if not active:
                    _top_up_active_jobs()
                    if not active:
                        break

                job, handle = active.popleft()
                print(f"  Waiting for {job['job_id']}...")
                if stop_requested():
                    try:
                        handle.cancel(terminate_containers=True)
                    except Exception:
                        pass
                    workers_cancelled += 1
                    emit(
                        {
                            "event": "worker_cancelled",
                            "job_id": job["job_id"],
                            "job_kind": job["kind"],
                            "workers_total": workers_total,
                            "workers_submitted": workers_submitted,
                            "workers_completed": workers_completed,
                            "workers_failed": workers_failed,
                            "workers_cancelled": workers_cancelled,
                        }
                    )
                    continue
                try:
                    while True:
                        if stop_requested():
                            try:
                                handle.cancel(terminate_containers=True)
                            except Exception:
                                pass
                            workers_cancelled += 1
                            emit(
                                {
                                    "event": "worker_cancelled",
                                    "job_id": job["job_id"],
                                    "job_kind": job["kind"],
                                    "workers_total": workers_total,
                                    "workers_submitted": workers_submitted,
                                    "workers_completed": workers_completed,
                                    "workers_failed": workers_failed,
                                    "workers_cancelled": workers_cancelled,
                                }
                            )
                            payload = None
                            break
                        try:
                            payload = handle.get(timeout=1.0)
                            break
                        except TimeoutError:
                            continue
                except Exception as e:
                    if stop_requested():
                        workers_cancelled += 1
                        emit(
                            {
                                "event": "worker_cancelled",
                                "job_id": job["job_id"],
                                "job_kind": job["kind"],
                                "workers_total": workers_total,
                                "workers_submitted": workers_submitted,
                                "workers_completed": workers_completed,
                                "workers_failed": workers_failed,
                                "workers_cancelled": workers_cancelled,
                            }
                        )
                        continue
                    error_text = str(e).strip() or repr(e)
                    workers_failed += 1
                    worker_results.append(
                        {
                            "job_id": job["job_id"],
                            "job_kind": job["kind"],
                            "ok": False,
                            "error": error_text,
                            "seed_count": len(job["coords"]),
                            "locations_returned": 0,
                            "panoramas_saved": 0,
                            "captures_saved": 0,
                            "stats": {},
                        }
                    )
                    if job["kind"] == "seed-retry":
                        retries_failed += 1
                    emit(
                        {
                            "event": "worker_failed",
                            "job_id": job["job_id"],
                            "job_kind": job["kind"],
                            "error": error_text,
                            "workers_total": workers_total,
                            "workers_submitted": workers_submitted,
                            "workers_completed": workers_completed,
                            "workers_failed": workers_failed,
                            "retries_queued": retries_queued,
                            "retries_completed": retries_completed,
                            "retries_failed": retries_failed,
                        }
                    )
                    # If a whole batch call fails, retry each seed individually once.
                    if job["kind"] == "batch" and not stop_requested():
                        for lat, lon in job["coords"]:
                            key = f"{round(lat, 6)},{round(lon, 6)}"
                            if key in retried_seed_keys:
                                continue
                            retried_seed_keys.add(key)
                            pending_jobs.append(
                                {
                                    "job_id": f"retry-{job_counter}",
                                    "kind": "seed-retry",
                                    "coords": [(lat, lon)],
                                }
                            )
                            job_counter += 1
                            workers_total += 1
                            retries_queued += 1
                            failed_seed_reasons[key] = "batch-call-failed"
                            emit(
                                {
                                    "event": "retry_enqueued",
                                    "seed": {"lat": lat, "lon": lon},
                                    "reason": "batch-call-failed",
                                    "workers_total": workers_total,
                                    "workers_submitted": workers_submitted,
                                    "workers_completed": workers_completed,
                                    "workers_failed": workers_failed,
                                    "retries_queued": retries_queued,
                                }
                            )
                    _top_up_active_jobs()
                    continue

                if payload is None:
                    _top_up_active_jobs()
                    continue

                results = payload.get("results", []) if isinstance(payload, dict) else []
                failed_seeds = payload.get("failed_seeds", []) if isinstance(payload, dict) else []
                worker_stats = payload.get("stats", {}) if isinstance(payload, dict) else {}

                # Refill the freed remote slot before local DB/image work so the
                # ephemeral Modal app stays active even when local post-processing
                # is slower than the remote scrape calls.
                _top_up_active_jobs()

                saved = save_results_to_local_db(
                    results,
                    db_path,
                    captures_dir,
                    capture_profile=capture_profile,
                    capture_kind=capture_kind,
                    missing_only=missing_only,
                )
                workers_completed += 1
                if job["kind"] == "seed-retry":
                    retries_completed += 1
                total_panos += saved["panoramas_saved"]
                total_captures += saved["captures_saved"]
                total_embeddings += int(saved.get("embeddings_saved", 0))
                total_embedding_errors += int(saved.get("embedding_errors", 0))

                worker_result = {
                    "job_id": job["job_id"],
                    "job_kind": job["kind"],
                    "ok": True,
                    "seed_count": len(job["coords"]),
                    "locations_returned": len(results),
                    "panoramas_saved": saved["panoramas_saved"],
                    "captures_saved": saved["captures_saved"],
                    "embeddings_saved": int(saved.get("embeddings_saved", 0)),
                    "embedding_errors": int(saved.get("embedding_errors", 0)),
                    "stats": worker_stats,
                }
                worker_results.append(worker_result)
                emit(
                    {
                        "event": "worker_completed",
                        **worker_result,
                        "workers_total": workers_total,
                        "workers_submitted": workers_submitted,
                        "workers_completed": workers_completed,
                        "workers_failed": workers_failed,
                        "retries_queued": retries_queued,
                        "retries_completed": retries_completed,
                        "retries_failed": retries_failed,
                    }
                )

                # Retry failed seeds individually once (for initial batch jobs only).
                if job["kind"] == "batch" and not stop_requested():
                    for seed in failed_seeds:
                        lat = float(seed["lat"])
                        lon = float(seed["lon"])
                        reason = seed.get("reason", "unknown")
                        key = f"{round(lat, 6)},{round(lon, 6)}"
                        if key in retried_seed_keys:
                            continue
                        retried_seed_keys.add(key)
                        pending_jobs.append(
                            {
                                "job_id": f"retry-{job_counter}",
                                "kind": "seed-retry",
                                "coords": [(lat, lon)],
                            }
                        )
                        job_counter += 1
                        workers_total += 1
                        retries_queued += 1
                        failed_seed_reasons[key] = reason
                        emit(
                            {
                                "event": "retry_enqueued",
                                "seed": {"lat": lat, "lon": lon},
                                "reason": reason,
                                "workers_total": workers_total,
                                "workers_submitted": workers_submitted,
                                "workers_completed": workers_completed,
                                "workers_failed": workers_failed,
                                "retries_queued": retries_queued,
                            }
                        )
                _top_up_active_jobs()
    except Exception as e:
        error_text = str(e).strip() or repr(e)
        if "app is stopped or disabled" in error_text.lower():
            raise RuntimeError(
                f"Modal app unavailable in environment '{environment_name}': {error_text}"
            ) from e
        raise

    result = {
        "total_panoramas_saved": total_panos,
        "total_captures_saved": total_captures,
        "total_embeddings_saved": total_embeddings,
        "total_embedding_errors": total_embedding_errors,
        "workers_total": workers_total,
        "workers_submitted": workers_submitted,
        "workers_completed": workers_completed,
        "workers_failed": workers_failed,
        "workers_cancelled": workers_cancelled,
        "retries_queued": retries_queued,
        "retries_completed": retries_completed,
        "retries_failed": retries_failed,
        "workers": worker_results,
        "failed_seed_reasons": failed_seed_reasons,
        "stopped": stop_requested(),
    }
    emit({"event": "all_done", **result})
    return result


@app.local_entrypoint()
def main(
    seeds: str = "seeds.csv",
    num_workers: int = 4,
    max_captures: int = 2000,
    modal_environment: str = DEFAULT_MODAL_ENVIRONMENT,
):
    """CLI: modal run worker/modal_worker.py --seeds seeds.csv --num-workers 4"""
    from config import CrawlerConfig

    cfg = CrawlerConfig()
    with open(seeds, "r", newline="") as f:
        reader = csv.DictReader(f)
        points = [(float(r["lat"]), float(r["lon"])) for r in reader]
    if max_captures < len(points):
        points = points[:max_captures]

    print(f"Loaded {len(points)} seeds from {seeds}")
    print(f"Dispatching {num_workers} Modal workers (batch-size={DEFAULT_BATCH_SIZE})...")
    result = dispatch_and_collect(
        points=points,
        num_workers=num_workers,
        db_path=cfg.DATABASE_URL,
        captures_dir=cfg.CAPTURES_DIR,
        modal_environment=modal_environment,
    )
    print(
        f"\nDone. {result['total_panoramas_saved']} panoramas, "
        f"{result['total_captures_saved']} captures saved to local DB."
    )
