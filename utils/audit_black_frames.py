"""
Backfill audit to flag existing black captures in the local DB.

Usage:
    python audit_black_frames.py
    python audit_black_frames.py --db postgresql://geospy:geospy@127.0.0.1:5432/geospy --dry-run
"""

import argparse
import os
from io import BytesIO

from PIL import Image, ImageStat

from db.postgres_database import Database


def analyze_image(filepath: str, dark_pixel_threshold: int, mean_threshold: float, dark_ratio_threshold: float):
    with open(filepath, "rb") as f:
        data = f.read()
    with Image.open(BytesIO(data)) as img:
        gray = img.convert("L").resize((160, 90))
        stat = ImageStat.Stat(gray)
        mean_val = float(stat.mean[0]) if stat.mean else 0.0
        hist = gray.histogram()
        total = max(1, sum(hist))
        dark_pixels = sum(hist[:dark_pixel_threshold])
        dark_ratio = dark_pixels / float(total)
        is_black = mean_val <= mean_threshold and dark_ratio >= dark_ratio_threshold
    return is_black, mean_val


def main():
    parser = argparse.ArgumentParser(description="Flag existing black frames in captures DB")
    parser.add_argument(
        "--db",
        type=str,
        default=os.getenv("DATABASE_URL", "postgresql://geospy:geospy@127.0.0.1:5432/geospy"),
        help="Postgres DATABASE_URL",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write DB updates")
    parser.add_argument("--mean-threshold", type=float, default=8.0)
    parser.add_argument("--dark-ratio-threshold", type=float, default=0.98)
    parser.add_argument("--dark-pixel-threshold", type=int, default=8)
    args = parser.parse_args()

    db = Database(args.db)

    scanned = 0
    flagged = 0
    missing = 0
    errors = 0

    for row in db.iter_capture_rows():
        scanned += 1
        capture_id = int(row["id"])
        filepath = row["filepath"]
        if not filepath or not os.path.exists(filepath):
            missing += 1
            if not args.dry_run:
                db.mark_capture_quality(capture_id, False, "missing-file", None)
            continue

        try:
            is_black, mean_val = analyze_image(
                filepath=filepath,
                dark_pixel_threshold=max(1, args.dark_pixel_threshold),
                mean_threshold=args.mean_threshold,
                dark_ratio_threshold=args.dark_ratio_threshold,
            )
            if is_black:
                flagged += 1
                if not args.dry_run:
                    db.mark_capture_quality(capture_id, True, "black-frame-backfill", mean_val)
            else:
                if not args.dry_run:
                    db.mark_capture_quality(capture_id, False, "ok", mean_val)
        except Exception:
            errors += 1
            if not args.dry_run:
                db.mark_capture_quality(capture_id, False, "analysis-error", None)

    db.close()

    print(f"Scanned: {scanned}")
    print(f"Flagged black: {flagged}")
    print(f"Missing files: {missing}")
    print(f"Errors: {errors}")
    print("Dry run." if args.dry_run else "DB updated.")


if __name__ == "__main__":
    main()

