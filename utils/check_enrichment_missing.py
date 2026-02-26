"""Report missing view coverage for a capture profile in a selected area."""

import argparse
from typing import Dict, List, Tuple

from config import CrawlerConfig
from db.postgres_database import Database

CAPTURE_PROFILES: Dict[str, Dict[str, List[float]]] = {
    "base": {"headings": [0.0, 90.0, 180.0, 270.0], "pitches": [75.0]},
    "high_v1": {
        "headings": [float(v) for v in range(0, 360, 15)],
        "pitches": [45.0, 60.0, 75.0, 90.0, 105.0],
    },
}


def build_required_views(profile_name: str) -> List[Tuple[float, float]]:
    profile = CAPTURE_PROFILES.get(profile_name, CAPTURE_PROFILES["base"])
    return [
        (float(heading), float(pitch))
        for pitch in profile["pitches"]
        for heading in profile["headings"]
    ]


def main():
    parser = argparse.ArgumentParser(description="Audit missing enrichment views in area.")
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--profile", type=str, default="high_v1", choices=sorted(CAPTURE_PROFILES.keys()))
    parser.add_argument("--top", type=int, default=10, help="Show top N panoramas with most missing views.")
    args = parser.parse_args()

    if args.min_lat >= args.max_lat or args.min_lon >= args.max_lon:
        raise SystemExit("Invalid bbox: min values must be less than max values.")

    config = CrawlerConfig()
    db = Database(config.DATABASE_URL)
    try:
        rows = db.get_panoramas_in_bbox(args.min_lat, args.min_lon, args.max_lat, args.max_lon)
        pano_ids = [int(row["id"]) for row in rows]
        required_views = build_required_views(args.profile)
        missing = db.get_missing_views_for_panoramas(
            pano_ids, required_views=required_views, capture_profile=args.profile
        )
    finally:
        db.close()

    total_panoramas = len(rows)
    required_per_pano = len(required_views)
    missing_counts = [(pid, len(missing.get(pid, []))) for pid in pano_ids]
    complete = sum(1 for _, count in missing_counts if count == 0)
    incomplete = total_panoramas - complete
    total_missing = sum(count for _, count in missing_counts)

    print(
        f"profile={args.profile} panoramas={total_panoramas} required_views_per_panorama={required_per_pano} "
        f"complete={complete} incomplete={incomplete} total_missing_views={total_missing}"
    )
    if incomplete:
        print("top_missing:")
        by_missing = sorted(missing_counts, key=lambda item: item[1], reverse=True)
        top_n = max(1, int(args.top))
        pano_lookup = {int(row["id"]): row for row in rows}
        for pano_id, count in by_missing[:top_n]:
            if count <= 0:
                break
            row = pano_lookup[pano_id]
            print(
                f"  pano_id={pano_id} external_pano_id={row.get('pano_id') or 'N/A'} "
                f"lat={float(row['lat']):.6f} lon={float(row['lon']):.6f} missing={count}"
            )


if __name__ == "__main__":
    main()
