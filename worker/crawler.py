"""
Street View Crawler — Playwright-based automation that navigates Google Street View,
captures screenshots at each location, and stores metadata for coverage tracking.

Usage:
    python -m worker.crawler --lat 37.7749 --lon -122.4194 --max 500 --strategy bfs
"""

import asyncio
import argparse
import math
import re
import os
import logging
from datetime import datetime
from collections import deque
from typing import Optional, Tuple, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from playwright.async_api import async_playwright, Page, Browser

from config import CrawlerConfig
from db.postgres_database import Capture, Database, Panorama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
SV_PANO_RE = re.compile(r"!1s([A-Za-z0-9_-]{10,})!2e\d+")
THUMB_PANO_RE = re.compile(r"panoid(?:=|%3D)([A-Za-z0-9_-]{10,})")


class StreetViewCrawler:
    """
    Navigates Street View in a real browser, captures panoramic screenshots,
    and tracks coverage with deduplication.
    """

    def __init__(self, config: CrawlerConfig, headless: bool = False):
        self.config = config
        self.headless = headless
        self.db = Database(config.DATABASE_URL)
        self.visited_count = 0
        self.queue: deque = deque()  # BFS queue of (lat, lon) to visit
        self.queued_points: set = set()  # rounded (lat, lon) currently in queue
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        os.makedirs(config.CAPTURES_DIR, exist_ok=True)

    async def start(self, seed_lat: float, seed_lon: float):
        """Initialize browser and begin crawling from seed location."""
        log.info(f"Starting crawl from ({seed_lat}, {seed_lon})")
        log.info(f"Strategy: {self.config.NAV_STRATEGY}, Max: {self.config.MAX_CAPTURES}")

        async with async_playwright() as pw:
            self.browser = await pw.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )

            context = await self.browser.new_context(
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

            self.page = await context.new_page()

            # Navigate to the seed location
            url = self.config.get_streetview_url(seed_lat, seed_lon)
            await self._safe_goto(url, timeout_ms=18000)
            await asyncio.sleep(1.5)  # Let Street View fully initialize

            # Dismiss any consent/cookie dialogs
            await self._dismiss_dialogs()

            # Begin the crawl loop
            await self._crawl_loop(seed_lat, seed_lon)

        self.db.close()
        log.info(f"Crawl complete. Total panoramas captured: {self.visited_count}")

    async def _dismiss_dialogs(self):
        """Try to dismiss Google consent/cookie banners."""
        try:
            # Common consent button selectors
            for selector in [
                'button[aria-label="Accept all"]',
                'button:has-text("Accept all")',
                'button:has-text("I agree")',
                'button:has-text("Reject all")',
                "[data-ved] button",
            ]:
                btn = self.page.locator(selector).first
                if await btn.is_visible(timeout=2000):
                    await btn.click()
                    await asyncio.sleep(1)
                    break
        except Exception:
            pass  # No dialog found, continue

    async def _crawl_loop(self, seed_lat: float, seed_lon: float):
        """Main crawl loop — navigate Street View and capture at each stop."""
        self._enqueue_target(seed_lat, seed_lon)

        while self.queue and self.visited_count < self.config.MAX_CAPTURES:
            if self.config.NAV_STRATEGY == "dfs":
                target_lat, target_lon = self.queue.pop()  # LIFO
            elif self.config.NAV_STRATEGY == "random":
                import random
                idx = random.randint(0, len(self.queue) - 1)
                target_lat, target_lon = self.queue[idx]
                del self.queue[idx]
            else:  # bfs
                target_lat, target_lon = self.queue.popleft()  # FIFO

            self.queued_points.discard((round(target_lat, 6), round(target_lon, 6)))

            # Check distance from seed
            dist_from_seed = self.db._haversine(seed_lat, seed_lon, target_lat, target_lon)
            if dist_from_seed > self.config.MAX_RADIUS_KM * 1000:
                log.debug(f"Skipping ({target_lat:.5f}, {target_lon:.5f}) — too far from seed")
                continue

            # Check dedup
            if self.db.is_duplicate(target_lat, target_lon, self.config.DEDUP_RADIUS_METERS):
                log.debug(f"Skipping ({target_lat:.5f}, {target_lon:.5f}) — duplicate")
                continue

            # Navigate to this point
            await self._navigate_to(target_lat, target_lon)
            if not await self._wait_for_street_view():
                log.debug(
                    "Skipping (%0.5f, %0.5f): map view (Street View not loaded)",
                    target_lat,
                    target_lon,
                )
                continue
            await asyncio.sleep(self.config.CAPTURE_DELAY)

            # Parse the actual position from the URL (Google may snap to nearest pano)
            actual_lat, actual_lon, heading = await self._parse_position()
            if actual_lat is None:
                log.warning("Could not parse position from URL, skipping")
                continue

            # Re-check dedup with the snapped position
            if self.db.is_duplicate(actual_lat, actual_lon, self.config.DEDUP_RADIUS_METERS):
                log.debug(f"Skipping snapped position ({actual_lat:.5f}, {actual_lon:.5f}) — duplicate")
                continue

            # Capture at all configured headings
            pano_id = await self._extract_pano_id()
            if not pano_id:
                pano_id = await self._extract_pano_id_from_page()
            if not pano_id:
                log.debug("Skipping (%0.5f, %0.5f): could not resolve pano_id", actual_lat, actual_lon)
                continue
            pano = Panorama(
                id=None,
                lat=actual_lat,
                lon=actual_lon,
                pano_id=pano_id,
                heading=heading,
                pitch=self.config.PITCH,
                timestamp=datetime.utcnow().isoformat(),
                source_url=self.page.url,
            )
            pano_db_id, created = self.db.add_panorama_if_new(
                pano, self.config.DEDUP_RADIUS_METERS
            )
            if not created:
                continue

            await self._capture_all_headings(pano_db_id, actual_lat, actual_lon, pano_id)
            self.visited_count += 1

            log.info(
                f"[{self.visited_count}/{self.config.MAX_CAPTURES}] "
                f"Captured ({actual_lat:.5f}, {actual_lon:.5f}) "
                f"pano_id={pano_id} | queue={len(self.queue)}"
            )

            # Discover adjacent panoramas by finding navigation links
            neighbors = await self._find_navigation_targets()
            for nlat, nlon in neighbors:
                if not self.db.is_duplicate(nlat, nlon, self.config.DEDUP_RADIUS_METERS):
                    self._enqueue_target(nlat, nlon)

            await asyncio.sleep(self.config.NAV_DELAY)

    def _enqueue_target(self, lat: float, lon: float):
        key = (round(lat, 6), round(lon, 6))
        if key in self.queued_points:
            return
        self.queued_points.add(key)
        self.queue.append((lat, lon))

    async def _navigate_to(self, lat: float, lon: float):
        """Navigate the browser to a Street View location."""
        await self._goto_real_street_view(
            lat=lat, lon=lon, heading=0.0, pano_id="", timeout_ms=15000
        )

    async def _safe_goto(self, url: str, timeout_ms: int = 12000):
        """Maps can keep background requests alive; do not fail hard on goto timeout."""
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception as e:
            log.debug("goto timeout for %s: %s", url, e)

    def _street_view_candidate_urls(
        self, lat: float, lon: float, heading: float, pano_id: str = ""
    ) -> List[str]:
        primary = self.config.get_streetview_url(lat, lon, heading=heading, pano_id=pano_id)
        secondary = self.config.get_streetview_url(lat, lon, heading=heading)
        fallback = (
            "https://www.google.com/maps"
            f"?layer=c&cbll={lat:.7f},{lon:.7f}&cbp=12,{heading:.2f},0,0,0"
        )
        return list(dict.fromkeys([primary, secondary, fallback]))

    async def _goto_real_street_view(
        self, lat: float, lon: float, heading: float, pano_id: str = "", timeout_ms: int = 12000
    ) -> bool:
        wait_timeout = max(3000, timeout_ms - 2000)
        for url in self._street_view_candidate_urls(lat, lon, heading, pano_id=pano_id):
            await self._safe_goto(url, timeout_ms=timeout_ms)
            if await self._wait_for_street_view(timeout_ms=wait_timeout):
                return True
            if await self._enter_street_view_by_click(timeout_ms=wait_timeout):
                return True
        return False

    async def _enter_street_view_by_click(self, timeout_ms: int = 8000) -> bool:
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
                el = self.page.locator(sel).first
                if await el.is_visible(timeout=1200):
                    await el.click()
                    if await self._wait_for_street_view(timeout_ms=timeout_ms):
                        return True
            except Exception:
                pass

        try:
            clicked = await self.page.evaluate(
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
            if clicked and await self._wait_for_street_view(timeout_ms=timeout_ms):
                return True
        except Exception:
            pass
        return False

    async def _parse_position(self) -> Tuple[Optional[float], Optional[float], float]:
        """
        Extract lat, lon, heading from the current Street View URL.
        URL format: .../@{lat},{lon},3a,{fov}y,{heading}h,{pitch}t/...
        """
        await asyncio.sleep(1)  # Let URL settle
        url = self.page.url
        if "3a," not in url:
            return None, None, 0.0

        # Pattern: @lat,lon,3a,FOVy,HEADINGh,PITCHt
        match = re.search(
            r"@(-?\d+\.?\d*),(-?\d+\.?\d*),3a,[\d.]+y,([\d.]+)h,([\d.-]+)t",
            url,
        )
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        match = re.search(r"@(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
        if match:
            return float(match.group(1)), float(match.group(2)), 0.0

        return None, None, 0.0

    async def _extract_pano_id(self) -> Optional[str]:
        """Extract Google's panorama ID from URL only (strict mode)."""
        url = self.page.url
        match = SV_PANO_RE.search(url)
        if not match:
            match = THUMB_PANO_RE.search(url)
        if match:
            return match.group(1)
        return None

    async def _extract_pano_id_from_page(self) -> Optional[str]:
        try:
            text = await self.page.content()
        except Exception:
            return None
        match = SV_PANO_RE.search(text)
        if not match:
            match = THUMB_PANO_RE.search(text)
        return match.group(1) if match else None

    async def _capture_all_headings(
        self, pano_db_id: int, lat: float, lon: float, pano_id: Optional[str]
    ):
        """Rotate to each configured heading and take a screenshot."""
        if not pano_id:
            return
        # Create directory for this panorama
        dir_name = pano_id or f"{lat:.5f}_{lon:.5f}"
        pano_dir = os.path.join(self.config.CAPTURES_DIR, dir_name)
        os.makedirs(pano_dir, exist_ok=True)

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

        for heading in self.config.HEADINGS:
            filepath = os.path.join(pano_dir, f"h{int(heading):03d}.jpg")
            try:
                thumb_pitch = self._thumbnail_pitch()
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
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
            except Exception as e:
                log.warning(f"Screenshot failed at heading {heading}: {e}")
                continue

            capture = Capture(
                id=None,
                panorama_id=pano_db_id,
                heading=heading,
                pitch=float(self.config.PITCH),
                filepath=filepath,
                width=self.config.VIEWPORT_WIDTH,
                height=self.config.VIEWPORT_HEIGHT,
                capture_profile="base",
                capture_kind="scan",
            )
            self.db.add_capture(capture)

    def _thumbnail_pitch(self) -> float:
        """
        Convert Maps URL tilt convention ("...{pitch}t") to thumbnail API pitch.
        Empirically: thumbnail_pitch ~= 90 - map_pitch.
        """
        val = 90.0 - float(self.config.PITCH)
        return max(-90.0, min(90.0, val))

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

    async def _wait_for_street_view(self, timeout_ms: int = 10000) -> bool:
        """Wait until Street View mode is visible enough to safely screenshot."""
        try:
            await self.page.wait_for_function(
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
            return ",3a," in self.page.url

    async def _find_navigation_targets(self) -> List[Tuple[float, float]]:
        """
        Find adjacent Street View locations by detecting navigation arrows/links
        on the current page and estimating where they lead.

        Strategy: Look for the clickable navigation arrows in Street View,
        simulate clicking each one, read the resulting URL, then go back.
        
        Alternative simpler approach: Generate a grid of nearby candidate points.
        """
        neighbors = []

        # --- Approach 1: Click-based discovery ---
        # Street View navigation arrows are typically in the canvas overlay
        # We can try clicking in the forward direction areas of the viewport
        try:
            # The center-bottom area usually has the forward arrow
            # Try clicking several directions from center
            cx, cy = self.config.VIEWPORT_WIDTH // 2, self.config.VIEWPORT_HEIGHT // 2

            # Potential click targets: forward (center-bottom), left, right
            click_targets = [
                (cx, cy + 200),       # Forward (center low)
                (cx, cy + 100),       # Forward (center mid-low)
                (cx - 300, cy + 100), # Forward-left
                (cx + 300, cy + 100), # Forward-right
            ]

            current_url = self.page.url
            current_lat, current_lon, _ = await self._parse_position()

            for tx, ty in click_targets:
                try:
                    await self.page.mouse.click(tx, ty, click_count=2)
                    await asyncio.sleep(1.5)

                    new_lat, new_lon, _ = await self._parse_position()
                    if new_lat is not None and current_lat is not None:
                        dist = self.db._haversine(current_lat, current_lon, new_lat, new_lon)
                        # Only count as a neighbor if we actually moved 5-200 meters
                        if 5 < dist < 200:
                            neighbors.append((new_lat, new_lon))

                    # Go back to where we were
                    await self.page.goto(current_url, wait_until="domcontentloaded", timeout=10000)
                    await asyncio.sleep(1)

                except Exception:
                    # Navigation click didn't work, try next
                    try:
                        await self.page.goto(current_url, wait_until="domcontentloaded", timeout=10000)
                        await asyncio.sleep(1)
                    except Exception:
                        pass

        except Exception as e:
            log.debug(f"Click-based discovery failed: {e}")

        # --- Approach 2: Grid-based fallback ---
        # If click-based didn't find enough neighbors, add cardinal points
        if len(neighbors) < 2:
            current_lat, current_lon, _ = await self._parse_position()
            if current_lat is not None:
                lat_step = 33 / 111320.0
                lon_scale = max(0.01, abs(math.cos(math.radians(current_lat))))
                lon_step = 33 / (111320.0 * lon_scale)
                for dlat, dlon in [(lat_step, 0), (-lat_step, 0), (0, lon_step), (0, -lon_step)]:
                    neighbors.append((current_lat + dlat, current_lon + dlon))

        log.debug(f"Found {len(neighbors)} navigation targets")
        return neighbors


def main():
    parser = argparse.ArgumentParser(description="Street View Crawler")
    parser.add_argument("--lat", type=float, required=True, help="Seed latitude")
    parser.add_argument("--lon", type=float, required=True, help="Seed longitude")
    parser.add_argument("--max", type=int, default=500, help="Max panoramas to capture")
    parser.add_argument("--radius", type=float, default=2.0, help="Max radius in km from seed")
    parser.add_argument("--strategy", choices=["bfs", "dfs", "random"], default="bfs")
    parser.add_argument("--pitch", type=float, default=75.0, help="Street View tilt angle (default 75)")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--dedup-radius", type=float, default=25.0, help="Dedup radius in meters")

    args = parser.parse_args()

    config = CrawlerConfig(
        MAX_CAPTURES=args.max,
        MAX_RADIUS_KM=args.radius,
        NAV_STRATEGY=args.strategy,
        PITCH=args.pitch,
        DEDUP_RADIUS_METERS=args.dedup_radius,
    )

    crawler = StreetViewCrawler(config, headless=args.headless)
    asyncio.run(crawler.start(args.lat, args.lon))


if __name__ == "__main__":
    main()
