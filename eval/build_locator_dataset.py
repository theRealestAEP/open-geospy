"""Build a locator evaluation manifest from local captures and optional negative files."""

import argparse
import os
import random
from typing import List

from config import CrawlerConfig
from db.postgres_database import Database
from eval.common import EvalCase, cases_to_rows, write_csv


def _capture_abs_path(captures_dir: str, filepath: str) -> str:
    raw = str(filepath or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    normalized = raw.replace("\\", "/")
    if normalized.startswith("captures/"):
        return os.path.join(captures_dir, normalized[len("captures/") :])
    return os.path.abspath(raw)


def _sample_positive_cases(
    *,
    db: Database,
    captures_dir: str,
    sample_size: int,
    seed: int,
    split: str,
    capture_profile: str,
) -> List[EvalCase]:
    rows = db.conn.execute(
        """
        SELECT
            c.id AS capture_id,
            c.panorama_id,
            c.filepath,
            c.heading,
            c.pitch,
            p.lat,
            p.lon
        FROM captures c
        JOIN panoramas p ON p.id = c.panorama_id
        WHERE c.capture_profile = %s
        ORDER BY c.id
        """,
        (capture_profile,),
    ).fetchall()
    items = [dict(row) for row in rows]
    random.Random(seed).shuffle(items)
    out: List[EvalCase] = []
    for row in items:
        image_path = _capture_abs_path(captures_dir, row.get("filepath", ""))
        if not image_path or not os.path.exists(image_path):
            continue
        capture_id = int(row["capture_id"])
        out.append(
            EvalCase(
                case_id=f"pos-{capture_id}",
                image_path=image_path,
                expected_lat=float(row["lat"]),
                expected_lon=float(row["lon"]),
                expected_panorama_id=int(row["panorama_id"]),
                expected_capture_id=capture_id,
                expected_reject=False,
                split=split,
                notes=f"profile={capture_profile} heading={float(row['heading']):.1f} pitch={float(row['pitch']):.1f}",
            )
        )
        if len(out) >= sample_size:
            break
    return out


def _scan_negative_dir(path: str, split: str) -> List[EvalCase]:
    if not path or not os.path.isdir(path):
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out: List[EvalCase] = []
    for root, _, files in os.walk(path):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            image_path = os.path.abspath(os.path.join(root, name))
            rel = os.path.relpath(image_path, path).replace(os.sep, "_")
            out.append(
                EvalCase(
                    case_id=f"neg-{os.path.splitext(rel)[0]}",
                    image_path=image_path,
                    expected_reject=True,
                    split=split,
                    notes="manual_negative",
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a locator evaluation manifest.")
    parser.add_argument("--output", default="eval/datasets/locator_cases.csv")
    parser.add_argument("--positive-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--capture-profile", default="high_v1")
    parser.add_argument(
        "--negative-dir",
        default="",
        help="Optional folder of out-of-index images to mark as expected_reject=1.",
    )
    args = parser.parse_args()

    cfg = CrawlerConfig()
    captures_dir = cfg.CAPTURES_DIR
    if not os.path.isabs(captures_dir):
        captures_dir = os.path.abspath(captures_dir)

    db = Database(cfg.DATABASE_URL)
    try:
        positives = _sample_positive_cases(
            db=db,
            captures_dir=captures_dir,
            sample_size=max(1, int(args.positive_count)),
            seed=int(args.seed),
            split=str(args.split or "dev").strip() or "dev",
            capture_profile=str(args.capture_profile or "high_v1").strip() or "high_v1",
        )
    finally:
        db.close()

    negatives = _scan_negative_dir(
        os.path.abspath(args.negative_dir) if args.negative_dir else "",
        split=str(args.split or "dev").strip() or "dev",
    )
    rows = cases_to_rows([*positives, *negatives])
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_csv(output_path, rows)
    print(f"wrote_manifest={output_path}")
    print(f"positive_cases={len(positives)}")
    print(f"negative_cases={len(negatives)}")


if __name__ == "__main__":
    main()
