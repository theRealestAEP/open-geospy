"""
Test script for Modal worker deployment.

Sends a small batch of known-good Street View coordinates to Modal,
waits for results, saves them locally, and verifies captures came back.

Prerequisites:
    modal setup   (or modal token set)

Usage:
    python -m utils.test_modal
    python -m utils.test_modal --workers 2
    python -m utils.test_modal --live   # save to real DB instead of temp
"""

import argparse
import os
import shutil
import sys
import tempfile

from worker.modal_worker import dispatch_and_collect

TEST_COORDS = [
    (37.7749, -122.4194),   # San Francisco - Market St
    (37.7851, -122.4094),   # SF Chinatown
    (37.7694, -122.4862),   # SF Golden Gate Park entrance
]


def count_valid_jpegs(captures_dir):
    valid = []
    for root, _dirs, files in os.walk(captures_dir):
        for fname in files:
            if not fname.endswith(".jpg"):
                continue
            fpath = os.path.join(root, fname)
            if os.path.getsize(fpath) < 128:
                continue
            with open(fpath, "rb") as f:
                if f.read(2) == b"\xff\xd8":
                    valid.append(fpath)
    return valid


def main():
    parser = argparse.ArgumentParser(description="Test Modal worker")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--modal-env",
        type=str,
        default="google-map-walkers",
        help="Modal environment to run worker calls in",
    )
    parser.add_argument("--live", action="store_true",
                        help="Write to real local DB instead of temp")
    args = parser.parse_args()

    if args.live:
        from config import CrawlerConfig
        cfg = CrawlerConfig()
        database_url = cfg.DATABASE_URL
        captures_dir = cfg.CAPTURES_DIR
        work_dir = None
        print(f"LIVE mode: writing to {database_url} and {captures_dir}/")
    else:
        work_dir = tempfile.mkdtemp(prefix="modal_test_")
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://geospy:geospy@127.0.0.1:5432/geospy",
        )
        captures_dir = os.path.join(work_dir, "captures")
        print(f"Test mode: captures at {work_dir}, database via {database_url}")

    print(
        f"Sending {len(TEST_COORDS)} seeds to {args.workers} Modal worker(s) "
        f"in env '{args.modal_env}'..."
    )
    print("First run can take several minutes while Modal builds the Playwright image.")
    print("Wait for 'Submitted worker ... call_id=...' before expecting runtime logs.\n")

    result = dispatch_and_collect(
        points=TEST_COORDS,
        num_workers=args.workers,
        db_path=database_url,
        captures_dir=captures_dir,
        modal_environment=args.modal_env,
    )

    valid = count_valid_jpegs(captures_dir)

    print()
    print("=" * 60)
    print("MODAL TEST RESULTS")
    print("=" * 60)
    print(f"  Panoramas saved:  {result['total_panoramas_saved']}")
    print(f"  Captures saved:   {result['total_captures_saved']}")
    print(f"  Valid JPEGs:      {len(valid)}")
    print()

    passed = result["total_panoramas_saved"] > 0 and len(valid) > 0
    if passed:
        print("PASS -- Modal workers scraped and returned data to local DB.")
    else:
        print("FAIL -- No captures returned.")
        print("  Check logs: modal app logs geospy-crawler")

    if work_dir and not args.live:
        if passed:
            shutil.rmtree(work_dir, ignore_errors=True)
            print("  Temp dir cleaned up.")
        else:
            print(f"  Artifacts at: {work_dir}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

