"""
Smoke test for headless mode.

Runs worker.batch_crawler in headless mode against a handful of known-good
Street View coordinates and verifies that capture files are produced.

Usage:
    python -m utils.test_headless
"""

import asyncio
import csv
import os
import shutil
import sys
import tempfile

from config import CrawlerConfig
from db.postgres_database import Database

KNOWN_GOOD_COORDS = [
    (37.7749, -122.4194),   # San Francisco - Market St area
    (37.7851, -122.4094),   # SF Chinatown
    (37.7694, -122.4862),   # SF Golden Gate Park entrance
]

TEMP_DIR_PREFIX = "headless_test_"


def write_temp_seeds(coords, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon"])
        writer.writerows(coords)


def check_captures(captures_dir):
    """Return a list of valid JPEG files found under captures_dir."""
    valid = []
    for root, _dirs, files in os.walk(captures_dir):
        for fname in files:
            if not fname.endswith(".jpg"):
                continue
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            if size < 128:
                print(f"  WARN: {fpath} is suspiciously small ({size} bytes)")
                continue
            # Quick JPEG header check
            with open(fpath, "rb") as img:
                header = img.read(3)
            if header[:2] != b"\xff\xd8":
                print(f"  WARN: {fpath} is not a valid JPEG")
                continue
            valid.append(fpath)
    return valid


async def run_test():
    work_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    seeds_path = os.path.join(work_dir, "test_seeds.csv")
    captures_dir = os.path.join(work_dir, "captures")
    db_url = CrawlerConfig().DATABASE_URL

    write_temp_seeds(KNOWN_GOOD_COORDS, seeds_path)

    config = CrawlerConfig(
        MAX_CAPTURES=len(KNOWN_GOOD_COORDS),
        DEDUP_RADIUS_METERS=25.0,
        DATABASE_URL=db_url,
        CAPTURES_DIR=captures_dir,
    )

    print(f"Work directory: {work_dir}")
    print(f"Testing {len(KNOWN_GOOD_COORDS)} seed coordinates in headless mode...")
    print()

    # Import here so top-level module loads don't crash if playwright isn't installed
    from worker.batch_crawler import BatchCrawler

    crawler = BatchCrawler(
        config=config,
        seeds_file=seeds_path,
        worker_id="headless-test",
        lease_seconds=300,
        headless=True,
        reset_queue=True,
    )
    await crawler.run()

    # Verify results
    db = Database(db_url)
    stats = db.get_stats()
    task_stats = db.get_seed_task_stats()
    db.close()

    valid_captures = check_captures(captures_dir)

    print()
    print("=" * 60)
    print("HEADLESS TEST RESULTS")
    print("=" * 60)
    print(f"  Panoramas in DB:     {stats['total_panoramas']}")
    print(f"  Captures in DB:      {stats['total_captures']}")
    print(f"  Valid JPEG files:    {len(valid_captures)}")
    print(f"  Seeds done:          {task_stats['done']}")
    print(f"  Seeds skipped:       {task_stats['skipped']}")
    print(f"  Seeds failed:        {task_stats['failed']}")
    print()

    passed = stats["total_panoramas"] > 0 and len(valid_captures) > 0
    if passed:
        print("RESULT: PASS -- Headless mode produced valid captures.")
    else:
        print("RESULT: FAIL -- No valid captures were produced.")
        print("  Check the output above for errors.")

    # Cleanup prompt
    print()
    print(f"Test artifacts in: {work_dir}")
    if passed:
        shutil.rmtree(work_dir, ignore_errors=True)
        print("  (cleaned up)")

    return 0 if passed else 1


def main():
    code = asyncio.run(run_test())
    sys.exit(code)


if __name__ == "__main__":
    main()

