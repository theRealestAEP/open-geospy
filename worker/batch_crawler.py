"""
Batch Crawler — Systematically visits a CSV of seed coordinates instead of
relying on click-based Street View navigation. More reliable and predictable.

Usage:
    # First generate seeds
    python -m utils.seed_grid --bbox 37.70,-122.52,37.82,-122.35 --step 50

    # Then crawl them
    python -m worker.batch_crawler --seeds seeds.csv --max 2000
"""

import asyncio
import argparse
import csv
import os
import re
import socket
import logging
from datetime import datetime
from typing import List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from playwright.async_api import async_playwright, Page

from backend.app.embedding_ingest import CaptureEmbeddingIngestor
from config import CrawlerConfig
from db.postgres_database import Capture, Database, Panorama
from worker.water_filter import is_water

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
SV_PANO_RE = re.compile(r"!1s([A-Za-z0-9_-]{10,})!2e\d+")
THUMB_PANO_RE = re.compile(r"panoid(?:=|%3D)([A-Za-z0-9_-]{10,})")


class BatchCrawler:
    """Visit a list of seed coordinates systematically with resumable task claims."""

    def __init__(
        self,
        config: CrawlerConfig,
        seeds_file: str,
        worker_id: str,
        lease_seconds: int,
        headless: bool,
        reset_queue: bool,
        job_kind: str = "scan",
        capture_profile: str = "base",
        headings: Optional[Sequence[float]] = None,
        pitches: Optional[Sequence[float]] = None,
        missing_only: bool = False,
        skip_location_dedup: bool = False,
        allow_existing_panorama: bool = False,
    ):
        self.config = config
        self.db = Database(config.DATABASE_URL)
        self.embedding_ingestor = CaptureEmbeddingIngestor(self.db, logger=log)
        self.seeds = self._load_seeds(seeds_file)
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.headless = headless
        self.reset_queue = reset_queue
        self.job_kind = job_kind
        self.capture_profile = capture_profile
        self.missing_only = bool(missing_only)
        self.skip_location_dedup = bool(skip_location_dedup)
        self.allow_existing_panorama = bool(allow_existing_panorama)
        self.capture_headings = [float(h) for h in (headings or config.HEADINGS)]
        self.capture_pitches = [float(p) for p in (pitches or [config.PITCH])]
        self.capture_views = [
            (float(h), float(p))
            for p in self.capture_pitches
            for h in self.capture_headings
        ]

        self.captured = 0
        self.skipped = 0
        self.failed = 0

        os.makedirs(config.CAPTURES_DIR, exist_ok=True)
        if self.reset_queue:
            self.db.clear_seed_tasks()
            log.info("Seed queue reset")
        inserted = self.db.queue_seed_points(self.seeds)
        log.info(
            "Seed queue ready. loaded=%s inserted_new=%s total_tasks=%s worker=%s",
            len(self.seeds),
            inserted,
            self.db.get_seed_task_stats()["total"],
            self.worker_id,
        )

    def _load_seeds(self, path: str) -> List[Tuple[float, float]]:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "lat" not in reader.fieldnames or "lon" not in reader.fieldnames:
                raise ValueError("Seeds CSV must contain 'lat' and 'lon' columns")
            return [(float(row["lat"]), float(row["lon"])) for row in reader]

    async def run(self):
        if not self.seeds:
            log.warning("No seeds found in CSV; nothing to do")
            self.db.close()
            return

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(
                viewport={
                    "width": self.config.VIEWPORT_WIDTH,
                    "height": self.config.VIEWPORT_HEIGHT,
                },
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            # Initial load to handle consent
            url = self.config.get_streetview_url(self.seeds[0][0], self.seeds[0][1])
            await self._safe_goto(page, url, timeout_ms=18000)
            await asyncio.sleep(1.5)
            await self._dismiss_dialogs(page)

            while True:
                if self.captured >= self.config.MAX_CAPTURES:
                    log.info("Reached max captures (%s)", self.config.MAX_CAPTURES)
                    break

                task = self.db.claim_next_seed(self.worker_id, self.lease_seconds)
                if task is None:
                    log.info("No claimable seed tasks remain")
                    break

                task_id = int(task["id"])
                lat = float(task["lat"])
                lon = float(task["lon"])
                try:
                    if is_water(lat, lon):
                        self.skipped += 1
                        self.db.mark_seed_status(task_id, "skipped", "water-point")
                        continue

                    # Quick dedup check before navigation.
                    if (
                        not self.skip_location_dedup
                        and self.db.is_duplicate(lat, lon, self.config.DEDUP_RADIUS_METERS)
                    ):
                        self.skipped += 1
                        self.db.mark_seed_status(task_id, "skipped", "duplicate-nearby")
                        continue

                    # Navigate
                    if not await self._goto_real_street_view(
                        page, lat=lat, lon=lon, heading=0.0, pano_id="", timeout_ms=12000
                    ):
                        log.info(
                            "Skip seed (%0.6f,%0.6f): could not enter Street View",
                            lat,
                            lon,
                        )
                        self.skipped += 1
                        self.db.mark_seed_status(task_id, "skipped", "street-view-not-loaded")
                        continue
                    await asyncio.sleep(self.config.CAPTURE_DELAY)

                    # Check if Street View actually loaded (vs. aerial/map view)
                    actual_url = page.url
                    if "3a," not in actual_url:
                        self.skipped += 1
                        self.db.mark_seed_status(task_id, "skipped", "street-view-unavailable")
                        continue

                    # Parse snapped position
                    actual_lat, actual_lon, heading = self._parse_url(actual_url)
                    if actual_lat is None:
                        self.failed += 1
                        self.db.mark_seed_status(task_id, "failed", "url-parse-failed")
                        continue

                    # Save panorama record (dedup by pano_id and location radius)
                    pano_id = self._extract_pano_id_from_url(actual_url)
                    if not pano_id:
                        pano_id = await self._extract_pano_id_from_page(page)
                    pano = Panorama(
                        id=None,
                        lat=actual_lat,
                        lon=actual_lon,
                        pano_id=pano_id,
                        heading=heading,
                        pitch=self.config.PITCH,
                        timestamp=datetime.utcnow().isoformat(),
                        source_url=actual_url,
                    )
                    pano_db_id, created = self.db.add_panorama_if_new(
                        pano, self.config.DEDUP_RADIUS_METERS
                    )
                    if not created and not self.allow_existing_panorama:
                        self.skipped += 1
                        self.db.mark_seed_status(task_id, "skipped", "duplicate-panorama")
                        continue

                    # Capture configured view combinations.
                    capture_count = await self._capture_views(
                        page, pano_db_id, actual_lat, actual_lon, pano_id
                    )
                    log.info(
                        "Captured %s/%s views at (%0.6f,%0.6f) pano_id=%s profile=%s kind=%s",
                        capture_count,
                        len(self.capture_views),
                        actual_lat,
                        actual_lon,
                        pano_id or "N/A",
                        self.capture_profile,
                        self.job_kind,
                    )
                    if capture_count == 0:
                        if self.job_kind == "enrich" and self.missing_only:
                            self.db.mark_seed_status(task_id, "done")
                            continue
                        self.failed += 1
                        self.db.mark_seed_status(task_id, "failed", "zero-captures")
                        continue

                    self.captured += 1
                    self.db.mark_seed_status(task_id, "done")

                except Exception as e:
                    log.warning("Task failed for (%s, %s): %s", lat, lon, e)
                    self.failed += 1
                    self.db.mark_seed_status(task_id, "failed", str(e))
                    continue

                if self.captured % 10 == 0:
                    stats = self.db.get_stats()
                    task_stats = self.db.get_seed_task_stats()
                    log.info(
                        "[%s/%s] captured=%s skipped=%s failed=%s db_panos=%s queue(p:%s i:%s d:%s s:%s f:%s)",
                        self.captured,
                        self.config.MAX_CAPTURES,
                        self.captured,
                        self.skipped,
                        self.failed,
                        stats["total_panoramas"],
                        task_stats["pending"],
                        task_stats["in_progress"],
                        task_stats["done"],
                        task_stats["skipped"],
                        task_stats["failed"],
                    )

                await asyncio.sleep(self.config.NAV_DELAY)

            await browser.close()

        self.embedding_ingestor.close()
        self.db.close()
        log.info(
            "Done. captured=%s skipped=%s failed=%s embeddings_saved=%s embedding_errors=%s",
            self.captured,
            self.skipped,
            self.failed,
            self.embedding_ingestor.saved_embeddings,
            self.embedding_ingestor.embed_errors,
        )

    async def _capture_views(
        self, page: Page, pano_db_id: int,
        lat: float, lon: float, pano_id: Optional[str]
    ) -> int:
        if not pano_id:
            return 0
        dir_name = pano_id or f"{lat:.5f}_{lon:.5f}"
        pano_dir = os.path.join(self.config.CAPTURES_DIR, dir_name)
        os.makedirs(pano_dir, exist_ok=True)

        success_count = 0
        # Save one raw tile sample from the same endpoint family for debugging/inspection.
        try:
            tile_bytes = self._fetch_image_bytes(
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
            with open(os.path.join(pano_dir, "tile_z1_x0_y0.jpg"), "wb") as f:
                f.write(tile_bytes)
        except Exception:
            pass

        views_to_capture = list(self.capture_views)
        if self.missing_only:
            existing = self.db.get_existing_capture_views(
                pano_db_id, capture_profile=self.capture_profile
            )
            views_to_capture = [
                (heading, pitch)
                for heading, pitch in views_to_capture
                if (round(float(heading), 3), round(float(pitch), 3)) not in existing
            ]

        for heading, pitch in views_to_capture:
            try:
                thumb_pitch = self._thumbnail_pitch(pitch)
                image_bytes = self._fetch_image_bytes(
                    "https://streetviewpixels-pa.googleapis.com/v1/thumbnail?"
                    + urlencode(
                        {
                            "cb_client": "maps_sv.tactile",
                            "w": self.config.VIEWPORT_WIDTH,
                            "h": self.config.VIEWPORT_HEIGHT,
                            "pitch": thumb_pitch,
                            "panoid": pano_id,
                            "yaw": float(heading),
                        }
                    )
                )

                filepath = os.path.join(
                    pano_dir,
                    self._capture_filename(heading, pitch, self.capture_profile),
                )
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                capture = Capture(
                    id=None,
                    panorama_id=pano_db_id,
                    heading=heading,
                    pitch=pitch,
                    filepath=filepath,
                    width=self.config.VIEWPORT_WIDTH,
                    height=self.config.VIEWPORT_HEIGHT,
                    capture_profile=self.capture_profile,
                    capture_kind=self.job_kind,
                )
                if self.missing_only:
                    capture_id, created = self.db.add_capture_if_missing(capture)
                    if created:
                        success_count += 1
                else:
                    capture_id = self.db.add_capture(capture)
                    created = True
                    success_count += 1
                if created:
                    self.embedding_ingestor.add_capture(int(capture_id), image_bytes)
            except Exception as e:
                log.debug(
                    "Capture failed heading=%s pitch=%s profile=%s error=%s",
                    heading,
                    pitch,
                    self.capture_profile,
                    e,
                )
        self.embedding_ingestor.flush()
        return success_count

    def _thumbnail_pitch(self, map_pitch: float) -> float:
        """
        Convert Maps URL tilt convention ("...{pitch}t") to thumbnail API pitch.
        Empirically: thumbnail_pitch ~= 90 - map_pitch.
        """
        val = 90.0 - float(map_pitch)
        return max(-90.0, min(90.0, val))

    @staticmethod
    def _capture_filename(heading: float, pitch: float, capture_profile: str) -> str:
        rounded_pitch = int(round(float(pitch)))
        profile_slug = "".join(
            c if c.isalnum() or c in {"-", "_"} else "-" for c in (capture_profile or "base")
        )
        if rounded_pitch == 75 and profile_slug == "base":
            return f"h{int(round(float(heading))):03d}.jpg"
        return (
            f"h{int(round(float(heading))):03d}_p{rounded_pitch:03d}_{profile_slug}.jpg"
        )

    @staticmethod
    def _fetch_image_bytes(url: str, timeout_seconds: int = 20) -> bytes:
        req = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.google.com/maps/",
            },
        )
        with urlopen(req, timeout=timeout_seconds) as resp:
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read()
        if "image" not in content_type.lower() or len(body) < 128:
            raise ValueError("Non-image response from streetviewpixels")
        return body

    async def _dismiss_dialogs(self, page: Page):
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

    @staticmethod
    async def _safe_goto(page: Page, url: str, timeout_ms: int = 12000):
        """
        Google Maps often never becomes 'networkidle', so treat goto timeouts as recoverable.
        """
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception as e:
            log.debug("goto timeout for %s: %s", url, e)

    def _street_view_candidate_urls(
        self, lat: float, lon: float, heading: float, pano_id: str = ""
    ) -> List[str]:
        primary = self.config.get_streetview_url(lat, lon, heading=heading, pano_id=pano_id)
        secondary = self.config.get_streetview_url(lat, lon, heading=heading)
        # Old-style cbll URL often forces Street View more reliably for some points.
        fallback = (
            "https://www.google.com/maps"
            f"?layer=c&cbll={lat:.7f},{lon:.7f}&cbp=12,{heading:.2f},0,0,0"
        )
        # Preserve order but drop duplicates.
        return list(dict.fromkeys([primary, secondary, fallback]))

    async def _goto_real_street_view(
        self,
        page: Page,
        lat: float,
        lon: float,
        heading: float,
        pano_id: str = "",
        timeout_ms: int = 12000,
    ) -> bool:
        wait_timeout = max(3000, timeout_ms - 2000)
        for url in self._street_view_candidate_urls(lat, lon, heading, pano_id=pano_id):
            await self._safe_goto(page, url, timeout_ms=timeout_ms)
            if await self._wait_for_street_view(page, timeout_ms=wait_timeout):
                return True
            if await self._enter_street_view_by_click(page, timeout_ms=wait_timeout):
                return True
        return False

    @staticmethod
    async def _enter_street_view_by_click(page: Page, timeout_ms: int = 8000) -> bool:
        """Fallback: click Street View UI elements/thumbnails to enter pano mode."""
        selectors = [
            "button[aria-label*='Street View']",
            "a[aria-label*='Street View']",
            "[role='button'][aria-label*='Street View']",
            "button:has-text('Street View')",
            "a:has-text('Street View')",
        ]
        for sel in selectors:
            try:
                el = page.locator(sel).first
                if await el.is_visible(timeout=1200):
                    await el.click()
                    if await BatchCrawler._wait_for_street_view(page, timeout_ms=timeout_ms):
                        return True
            except Exception:
                pass

        # Try the thumbnail-card path (streetviewpixels image usually appears there).
        try:
            clicked = await page.evaluate(
                """
                () => {
                    const img = document.querySelector("img[src*='streetviewpixels-pa.googleapis.com']");
                    if (!img) return false;
                    const clickable = img.closest("button, a, [role='button'], div");
                    if (!clickable) return false;
                    clickable.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
                    return true;
                }
                """
            )
            if clicked and await BatchCrawler._wait_for_street_view(page, timeout_ms=timeout_ms):
                return True
        except Exception:
            pass
        return False

    @staticmethod
    async def _wait_for_street_view(page: Page, timeout_ms: int = 10000) -> bool:
        """Wait until Street View mode is visible enough to safely screenshot."""
        try:
            await page.wait_for_function(
                """
                () => {
                    const href = window.location.href || '';
                    if (!href.includes(',3a,')) return false;
                    const txt = (document.body?.innerText || '');
                    return /!1s[A-Za-z0-9_-]{10,}!2e\\d+/.test(href)
                        || /panoid(?:=|%3D)[A-Za-z0-9_-]{10,}/.test(href)
                        || txt.includes('Hide imagery')
                        || txt.includes('Google Street View');
                }
                """,
                timeout=timeout_ms,
            )
            return True
        except Exception:
            return ",3a," in page.url

    @staticmethod
    def _parse_url(url: str) -> Tuple[Optional[float], Optional[float], float]:
        if "3a," not in url:
            return None, None, 0.0
        match = re.search(
            r"@(-?\d+\.?\d*),(-?\d+\.?\d*),3a,[\d.]+y,([\d.]+)h", url
        )
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        match = re.search(r"@(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
        if match:
            return float(match.group(1)), float(match.group(2)), 0.0
        return None, None, 0.0

    @staticmethod
    def _extract_pano_id_from_url(url: str) -> Optional[str]:
        match = SV_PANO_RE.search(url)
        if not match:
            match = THUMB_PANO_RE.search(url)
        return match.group(1) if match else None

    @staticmethod
    async def _extract_pano_id_from_page(page: Page) -> Optional[str]:
        try:
            text = await page.content()
        except Exception:
            return None
        match = SV_PANO_RE.search(text)
        if not match:
            match = THUMB_PANO_RE.search(text)
        return match.group(1) if match else None


def main():
    def parse_float_csv(text: str) -> List[float]:
        values = []
        for item in str(text or "").split(","):
            item = item.strip()
            if not item:
                continue
            values.append(float(item))
        return values

    parser = argparse.ArgumentParser(description="Batch Street View Crawler")
    parser.add_argument("--seeds", type=str, required=True, help="CSV with lat,lon columns")
    parser.add_argument("--max", type=int, default=2000, help="Max panoramas")
    parser.add_argument("--dedup-radius", type=float, default=25.0, help="Dedup radius meters")
    parser.add_argument("--pitch", type=float, default=75.0, help="Street View tilt angle (default 75)")
    parser.add_argument("--lease-seconds", type=int, default=300, help="Task claim lease time")
    parser.add_argument("--worker-id", type=str, default="", help="Unique worker identity")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--reset-queue", action="store_true", help="Delete existing seed queue before loading CSV")
    parser.add_argument("--job-kind", type=str, default="scan", choices=["scan", "enrich", "fill"])
    parser.add_argument("--capture-profile", type=str, default="base")
    parser.add_argument("--headings-csv", type=str, default="")
    parser.add_argument("--pitches-csv", type=str, default="")
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--skip-location-dedup", action="store_true")
    parser.add_argument("--allow-existing-panorama", action="store_true")

    args = parser.parse_args()

    config = CrawlerConfig(
        MAX_CAPTURES=args.max,
        DEDUP_RADIUS_METERS=args.dedup_radius,
        PITCH=args.pitch,
    )

    worker_id = args.worker_id.strip() or f"{socket.gethostname()}-{os.getpid()}"
    capture_headings = parse_float_csv(args.headings_csv) or list(config.HEADINGS)
    capture_pitches = parse_float_csv(args.pitches_csv) or [float(config.PITCH)]
    crawler = BatchCrawler(
        config=config,
        seeds_file=args.seeds,
        worker_id=worker_id,
        lease_seconds=args.lease_seconds,
        headless=args.headless,
        reset_queue=args.reset_queue,
        job_kind=args.job_kind,
        capture_profile=args.capture_profile,
        headings=capture_headings,
        pitches=capture_pitches,
        missing_only=args.missing_only,
        skip_location_dedup=args.skip_location_dedup,
        allow_existing_panorama=args.allow_existing_panorama,
    )
    asyncio.run(crawler.run())


if __name__ == "__main__":
    main()
