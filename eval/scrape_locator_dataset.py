"""Harvest random Street View query images for locator eval manifests."""

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import CrawlerConfig
from eval.common import EvalCase, haversine_m, write_csv, write_json
from worker.water_filter import is_water

SV_PANO_RE = re.compile(r"!1s([A-Za-z0-9_-]{10,})!2e\d+")
THUMB_PANO_RE = re.compile(r"panoid(?:=|%3D)([A-Za-z0-9_-]{10,})")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _parse_bbox(raw: str) -> Tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox must be min_lat,min_lon,max_lat,max_lon")
    min_lat, min_lon, max_lat, max_lon = parts
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("--bbox must be ordered as min_lat,min_lon,max_lat,max_lon")
    return min_lat, min_lon, max_lat, max_lon


def _parse_float_csv(raw: str) -> List[float]:
    values: List[float] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def _parse_polygon(raw: str) -> List[Tuple[float, float]]:
    text = str(raw or "").strip()
    if not text:
        return []
    if os.path.exists(text):
        with open(text, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [
                [float(part.strip()) for part in pair.split(",")]
                for pair in text.split(";")
                if pair.strip()
            ]

    if isinstance(payload, dict):
        payload = payload.get("coordinates") or payload.get("polygon") or []
        if payload and isinstance(payload[0], list) and payload[0] and isinstance(payload[0][0], list):
            payload = payload[0]

    polygon = [(float(point[0]), float(point[1])) for point in payload]
    if polygon and len(polygon) < 3:
        raise ValueError("polygon needs at least 3 points")
    return polygon


def _point_in_polygon(lat: float, lon: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return True
    inside = False
    x = lon
    y = lat
    j = len(polygon) - 1
    for i, (yi, xi) in enumerate(polygon):
        yj, xj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _random_point_in_boundary(
    rng: random.Random,
    bbox: Tuple[float, float, float, float],
    polygon: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    min_lat, min_lon, max_lat, max_lon = bbox
    for _ in range(1000):
        lat = rng.uniform(min_lat, max_lat)
        lon = rng.uniform(min_lon, max_lon)
        if _point_in_polygon(lat, lon, polygon):
            return lat, lon
    raise RuntimeError("Could not sample a point inside the requested boundary")


def _choose_heading(rng: random.Random, headings: Sequence[float], mode: str) -> float:
    if headings:
        return float(rng.choice(list(headings)))
    if mode == "cardinal":
        return float(rng.choice([0, 90, 180, 270]))
    if mode == "ordinal":
        return float(rng.choice(list(range(0, 360, 45))))
    return round(rng.uniform(0, 360), 2)


def _thumbnail_pitch(map_pitch: float) -> float:
    return max(-90.0, min(90.0, 90.0 - float(map_pitch)))


def _capture_url(
    *,
    pano_id: str,
    heading: float,
    pitch: float,
    width: int,
    height: int,
) -> str:
    return "https://streetviewpixels-pa.googleapis.com/v1/thumbnail?" + urlencode(
        {
            "cb_client": "maps_sv.tactile",
            "w": int(width),
            "h": int(height),
            "pitch": _thumbnail_pitch(pitch),
            "panoid": pano_id,
            "yaw": float(heading),
        }
    )


def _candidate_urls(config: CrawlerConfig, lat: float, lon: float) -> List[str]:
    return list(
        dict.fromkeys(
            [
                config.get_streetview_url(lat, lon, heading=0.0),
                (
                    "https://www.google.com/maps"
                    f"?layer=c&cbll={lat:.7f},{lon:.7f}&cbp=12,0,0,0,0"
                ),
            ]
        )
    )


async def _safe_goto(page, url: str, timeout_ms: int = 12000) -> None:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass


async def _wait_for_street_view(page, timeout_ms: int = 10000) -> bool:
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


async def _enter_street_view_by_click(page, timeout_ms: int = 8000) -> bool:
    selectors = [
        "button[aria-label*='Street View']",
        "a[aria-label*='Street View']",
        "[role='button'][aria-label*='Street View']",
        "button:has-text('Street View')",
        "a:has-text('Street View')",
    ]
    for selector in selectors:
        try:
            el = page.locator(selector).first
            if await el.is_visible(timeout=1200):
                await el.click()
                if await _wait_for_street_view(page, timeout_ms=timeout_ms):
                    return True
        except Exception:
            pass
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
        return bool(clicked and await _wait_for_street_view(page, timeout_ms=timeout_ms))
    except Exception:
        return False


def _parse_street_view_url(url: str) -> Tuple[Optional[float], Optional[float], float]:
    if "3a," not in url:
        return None, None, 0.0
    match = re.search(r"@(-?\d+\.?\d*),(-?\d+\.?\d*),3a,[\d.]+y,([\d.]+)h", url)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    match = re.search(r"@(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
    if match:
        return float(match.group(1)), float(match.group(2)), 0.0
    return None, None, 0.0


def _extract_pano_id_from_url(url: str) -> Optional[str]:
    match = SV_PANO_RE.search(url) or THUMB_PANO_RE.search(url)
    return match.group(1) if match else None


async def _extract_pano_id_from_page(page) -> Optional[str]:
    try:
        text = await page.content()
    except Exception:
        return None
    match = SV_PANO_RE.search(text) or THUMB_PANO_RE.search(text)
    return match.group(1) if match else None


def _fetch_image_bytes(url: str, timeout_seconds: int = 20) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Referer": "https://www.google.com/maps/"})
    with urlopen(req, timeout=timeout_seconds) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read()
    if "image" not in content_type.lower() or len(body) < 128:
        raise ValueError("Non-image response from streetviewpixels")
    return body


async def _resolve_street_view(page, config: CrawlerConfig, lat: float, lon: float):
    for url in _candidate_urls(config, lat, lon):
        await _safe_goto(page, url, timeout_ms=14000)
        if await _wait_for_street_view(page, timeout_ms=10000):
            break
        if await _enter_street_view_by_click(page, timeout_ms=9000):
            break
    else:
        return None

    actual_lat, actual_lon, _ = _parse_street_view_url(page.url)
    if actual_lat is None or actual_lon is None:
        return None
    pano_id = _extract_pano_id_from_url(page.url)
    if not pano_id:
        pano_id = await _extract_pano_id_from_page(page)
    if not pano_id:
        return None
    return {
        "lat": float(actual_lat),
        "lon": float(actual_lon),
        "pano_id": str(pano_id),
        "source_url": page.url,
    }


def _open_db_for_optional_ids(disabled: bool) -> Optional[Any]:
    if disabled:
        return None
    try:
        from db.postgres_database import Database

        return Database(CrawlerConfig().DATABASE_URL)
    except Exception:
        return None


async def scrape_dataset(args: argparse.Namespace) -> dict:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for scraping. Install requirements and run `playwright install chromium`."
        ) from exc

    bbox = _parse_bbox(args.bbox)
    polygon = _parse_polygon(args.polygon)
    output_dir = os.path.abspath(args.output_dir)
    image_dir = os.path.join(output_dir, "images")
    manifest_path = os.path.abspath(
        args.output or os.path.join(output_dir, "locator_cases.csv")
    )
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    rng = random.Random(int(args.seed))
    pitches = _parse_float_csv(args.pitches_csv) or [60.0, 75.0, 90.0]
    headings = _parse_float_csv(args.headings_csv)
    config = CrawlerConfig(
        VIEWPORT_WIDTH=max(256, int(args.width)),
        VIEWPORT_HEIGHT=max(256, int(args.height)),
    )
    db = _open_db_for_optional_ids(bool(args.no_db_ids))
    rows: List[dict] = []
    metadata: List[dict] = []
    seen_panos: set[str] = set()
    seen_locations: List[Tuple[float, float]] = []
    attempts = 0
    max_attempts = max(int(args.max_attempts), int(args.count) * 20)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=not bool(args.headed),
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            viewport={"width": config.VIEWPORT_WIDTH, "height": config.VIEWPORT_HEIGHT},
            user_agent=USER_AGENT,
        )
        page = await context.new_page()

        while len(rows) < int(args.count) and attempts < max_attempts:
            attempts += 1
            lat, lon = _random_point_in_boundary(rng, bbox, polygon)
            if not args.allow_water and is_water(lat, lon):
                continue

            resolved = await _resolve_street_view(page, config, lat, lon)
            if not resolved:
                continue
            pano_id = str(resolved["pano_id"])
            if not args.allow_duplicate_panos and pano_id in seen_panos:
                continue
            if args.min_distance_meters > 0:
                too_close = any(
                    haversine_m(resolved["lat"], resolved["lon"], old_lat, old_lon)
                    < float(args.min_distance_meters)
                    for old_lat, old_lon in seen_locations
                )
                if too_close:
                    continue

            seen_panos.add(pano_id)
            seen_locations.append((float(resolved["lat"]), float(resolved["lon"])))
            expected_panorama_id = (
                db.get_panorama_id_by_pano_id(pano_id) if db is not None else None
            )
            views_for_pano = max(1, int(args.views_per_panorama))
            for view_idx in range(views_for_pano):
                if len(rows) >= int(args.count):
                    break
                heading = _choose_heading(rng, headings, args.heading_mode)
                pitch = float(rng.choice(pitches))
                case_id = f"scrape-{len(rows) + 1:05d}"
                filename = f"{case_id}_pano-{pano_id}_h{int(round(heading)) % 360:03d}_p{int(round(pitch)):03d}.jpg"
                image_path = os.path.join(image_dir, filename)
                try:
                    image_bytes = _fetch_image_bytes(
                        _capture_url(
                            pano_id=pano_id,
                            heading=heading,
                            pitch=pitch,
                            width=config.VIEWPORT_WIDTH,
                            height=config.VIEWPORT_HEIGHT,
                        )
                    )
                except Exception as exc:
                    print(
                        f"skip_view pano_id={pano_id} heading={heading:.1f} "
                        f"pitch={pitch:.1f} error={exc}"
                    )
                    continue
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                rel_image_path = os.path.relpath(image_path, os.path.dirname(manifest_path))
                case = EvalCase(
                    case_id=case_id,
                    image_path=rel_image_path,
                    expected_lat=float(resolved["lat"]),
                    expected_lon=float(resolved["lon"]),
                    expected_panorama_id=expected_panorama_id,
                    expected_reject=False,
                    split=str(args.split or "scraped"),
                    notes=(
                        f"scraped pano_id={pano_id} requested={lat:.7f},{lon:.7f} "
                        f"heading={heading:.2f} pitch={pitch:.2f}"
                    ),
                )
                rows.append(asdict(case))
                metadata.append(
                    {
                        "case_id": case_id,
                        "image_path": rel_image_path,
                        "pano_id": pano_id,
                        "expected_panorama_id": expected_panorama_id,
                        "expected_lat": float(resolved["lat"]),
                        "expected_lon": float(resolved["lon"]),
                        "requested_lat": lat,
                        "requested_lon": lon,
                        "heading": heading,
                        "pitch": pitch,
                        "source_url": resolved["source_url"],
                    }
                )
                print(
                    f"[{len(rows)}/{args.count}] case_id={case_id} "
                    f"pano_id={pano_id} heading={heading:.1f} pitch={pitch:.1f}"
                )
            await asyncio.sleep(max(0.0, float(args.delay_seconds)))

        await browser.close()

    if db is not None:
        db.close()

    write_csv(manifest_path, rows)
    meta_path = os.path.join(output_dir, "scrape_metadata.json")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bbox": {
            "min_lat": bbox[0],
            "min_lon": bbox[1],
            "max_lat": bbox[2],
            "max_lon": bbox[3],
        },
        "polygon_points": len(polygon),
        "requested_count": int(args.count),
        "written_cases": len(rows),
        "attempts": attempts,
        "image_dir": image_dir,
        "manifest_path": manifest_path,
        "metadata_path": meta_path,
    }
    write_json(meta_path, {"summary": summary, "cases": metadata})
    print(f"wrote_manifest={manifest_path}")
    print(f"written_cases={len(rows)}")
    print(f"attempts={attempts}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape random Street View images into a locator eval manifest."
    )
    parser.add_argument("--bbox", required=True, help="min_lat,min_lon,max_lat,max_lon")
    parser.add_argument(
        "--polygon",
        default="",
        help="Optional polygon as JSON, GeoJSON-ish coordinates, file path, or 'lat,lon;lat,lon;...'.",
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output-dir", default="eval/datasets/scraped_locator")
    parser.add_argument("--output", default="")
    parser.add_argument("--split", default="scraped")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--views-per-panorama", type=int, default=1)
    parser.add_argument("--min-distance-meters", type=float, default=20.0)
    parser.add_argument("--allow-duplicate-panos", action="store_true")
    parser.add_argument("--allow-water", action="store_true")
    parser.add_argument("--no-db-ids", action="store_true")
    parser.add_argument("--headed", action="store_true", help="Show the Playwright browser.")
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument(
        "--heading-mode",
        choices=["random", "cardinal", "ordinal"],
        default="random",
    )
    parser.add_argument(
        "--headings-csv",
        default="",
        help="Optional fixed headings to sample from, e.g. '0,45,90,135'.",
    )
    parser.add_argument(
        "--pitches-csv",
        default="60,75,90",
        help="Map tilt values to sample from. 75 is roughly horizon-forward.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if int(args.count) <= 0:
        raise SystemExit("--count must be > 0")
    asyncio.run(scrape_dataset(args))


if __name__ == "__main__":
    main()
