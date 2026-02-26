"""Estimate fill candidate count for a selected area before dispatching a fill job."""

import argparse
import math
from typing import List, Tuple

from config import CrawlerConfig
from db.postgres_database import Database
from utils.seed_grid import generate_grid
from worker.water_filter import filter_water_points


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def gap_filter(
    candidates: List[Tuple[float, float]],
    existing: List[Tuple[float, float]],
    gap_meters: float,
) -> List[Tuple[float, float]]:
    kept: List[Tuple[float, float]] = []
    for lat, lon in candidates:
        is_far = True
        for existing_lat, existing_lon in existing:
            if haversine_m(lat, lon, existing_lat, existing_lon) <= gap_meters:
                is_far = False
                break
        if is_far:
            kept.append((lat, lon))
    return kept


def main():
    parser = argparse.ArgumentParser(description="Preview fill candidates in area.")
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--step-meters", type=float, default=50.0)
    parser.add_argument("--gap-meters", type=float, default=40.0)
    args = parser.parse_args()

    if args.min_lat >= args.max_lat or args.min_lon >= args.max_lon:
        raise SystemExit("Invalid bbox: min values must be less than max values.")

    raw = generate_grid(args.min_lat, args.min_lon, args.max_lat, args.max_lon, args.step_meters)
    land = filter_water_points(raw)

    config = CrawlerConfig()
    db = Database(config.DATABASE_URL)
    try:
        existing_rows = db.get_panoramas_in_bbox(args.min_lat, args.min_lon, args.max_lat, args.max_lon)
    finally:
        db.close()
    existing_points = [(float(row["lat"]), float(row["lon"])) for row in existing_rows]

    gap_kept = gap_filter(land, existing_points, float(args.gap_meters))
    print(
        f"raw={len(raw)} land={len(land)} existing_in_bbox={len(existing_points)} "
        f"gap_candidates={len(gap_kept)} gap_removed={len(land) - len(gap_kept)}"
    )


if __name__ == "__main__":
    main()
