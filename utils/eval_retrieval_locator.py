"""Evaluate locator distance error from a CSV of query images."""

import argparse
import csv
import io
import math
import os
from statistics import median
from typing import Dict, List, Optional

from PIL import Image

from backend.app.clip_embeddings import get_retrieval_embedders
from backend.app.services.retrieval_locator import locate_image_bytes
from config import CrawlerConfig
from db.postgres_database import Database


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def _capture_abs_path(captures_dir: str, filepath: str) -> str:
    raw = (filepath or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    normalized = raw.replace("\\", "/")
    if normalized.startswith("captures/"):
        relative_inside = normalized[len("captures/") :]
        return os.path.join(captures_dir, relative_inside)
    return os.path.abspath(raw)


def _read_queries(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = str(row.get("image_path", "")).strip()
            if not image_path:
                continue
            try:
                lat = float(row.get("lat", ""))
                lon = float(row.get("lon", ""))
            except Exception:
                continue
            rows.append(
                {
                    "id": str(row.get("id", image_path)),
                    "image_path": image_path,
                    "lat": lat,
                    "lon": lon,
                }
            )
    return rows


def _format_bucket(errors: List[float], threshold: float) -> str:
    if not errors:
        return "0/0 (0.0%)"
    hits = sum(1 for e in errors if e <= threshold)
    pct = 100.0 * hits / len(errors)
    return f"{hits}/{len(errors)} ({pct:.1f}%)"


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    p = max(0.0, min(100.0, float(p)))
    sorted_vals = sorted(values)
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def _partial_query_variants(image_bytes: bytes, mode: str) -> List[tuple]:
    mode = str(mode or "none").strip().lower()
    if mode == "none":
        return [("full", image_bytes)]
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        w, h = img.size

        def to_bytes(crop_img: Image.Image) -> bytes:
            out = io.BytesIO()
            crop_img.save(out, format="JPEG", quality=92)
            return out.getvalue()

        if mode == "center60":
            cw, ch = int(w * 0.6), int(h * 0.6)
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            return [("center60", to_bytes(img.crop((x0, y0, x0 + cw, y0 + ch))))]
        if mode == "center40":
            cw, ch = int(w * 0.4), int(h * 0.4)
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            return [("center40", to_bytes(img.crop((x0, y0, x0 + cw, y0 + ch))))]
        if mode == "quarters":
            return [
                ("q1", to_bytes(img.crop((0, 0, w // 2, h // 2)))),
                ("q2", to_bytes(img.crop((w // 2, 0, w, h // 2)))),
                ("q3", to_bytes(img.crop((0, h // 2, w // 2, h)))),
                ("q4", to_bytes(img.crop((w // 2, h // 2, w, h)))),
            ]
        return [("full", image_bytes)]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate locator error from CSV: id,image_path,lat,lon"
    )
    parser.add_argument("--queries-csv", required=True)
    parser.add_argument("--top-k-per-crop", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--vote-cap", type=int, default=3)
    parser.add_argument("--cluster-radius-m", type=float, default=45.0)
    parser.add_argument("--verify-top-n", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument(
        "--partial-mode",
        choices=["none", "center60", "center40", "quarters"],
        default="none",
    )
    parser.add_argument("--far-error-m", type=float, default=250.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = CrawlerConfig()
    captures_dir = cfg.CAPTURES_DIR
    if not os.path.isabs(captures_dir):
        captures_dir = os.path.abspath(captures_dir)

    queries = _read_queries(args.queries_csv)
    if args.limit > 0:
        queries = queries[: args.limit]
    if not queries:
        print("No valid queries found.")
        return

    db = Database(cfg.DATABASE_URL)
    embedders = list(get_retrieval_embedders())
    model_weights = {
        str(getattr(embedder, "model_id", f"m{idx}")): float(
            getattr(embedder, "weight", 1.0)
        )
        for idx, embedder in enumerate(embedders)
    }
    errors: List[float] = []
    missing_estimate = 0
    far_errors = 0
    rows_out: List[Dict] = []
    try:
        for query in queries:
            image_path = query["image_path"]
            if not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)
            if not os.path.exists(image_path):
                missing_estimate += 1
                rows_out.append({**query, "status": "missing_file", "error_m": ""})
                continue
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            variants = _partial_query_variants(image_bytes, args.partial_mode)
            for variant_name, variant_bytes in variants:
                result = locate_image_bytes(
                    image_bytes=variant_bytes,
                    db=db,
                    embedders=embedders,
                    model_weights=model_weights,
                    capture_abs_path=lambda fp: _capture_abs_path(captures_dir, fp),
                    top_k_per_crop=args.top_k_per_crop,
                    max_merged_candidates=args.max_candidates,
                    panorama_vote_cap=args.vote_cap,
                    cluster_radius_m=args.cluster_radius_m,
                    verify_top_n=args.verify_top_n,
                    min_similarity=args.min_similarity,
                    include_debug=False,
                )
                estimate = result.get("best_estimate")
                if not estimate:
                    missing_estimate += 1
                    rows_out.append(
                        {
                            **query,
                            "variant": variant_name,
                            "status": "no_estimate",
                            "error_m": "",
                            "flags": ",".join(result.get("flags", [])),
                        }
                    )
                    continue
                error_m = _haversine_m(
                    query["lat"],
                    query["lon"],
                    float(estimate["lat"]),
                    float(estimate["lon"]),
                )
                errors.append(error_m)
                if error_m > float(args.far_error_m):
                    far_errors += 1
                rows_out.append(
                    {
                        **query,
                        "variant": variant_name,
                        "status": "ok",
                        "pred_lat": estimate["lat"],
                        "pred_lon": estimate["lon"],
                        "confidence": estimate.get("confidence", 0),
                        "radius_m": estimate.get("radius_m", 0),
                        "error_m": round(error_m, 2),
                        "flags": ",".join(result.get("flags", [])),
                    }
                )
    finally:
        db.close()

    print(f"Queries: {len(queries)}")
    print(f"With estimate: {len(errors)}")
    print(f"No estimate: {missing_estimate}")
    if errors:
        print(f"Median error: {median(errors):.2f}m")
        p90 = _percentile(errors, 90.0)
        if p90 is not None:
            print(f"P90 error: {p90:.2f}m")
        print(f"<=25m: {_format_bucket(errors, 25.0)}")
        print(f"<=50m: {_format_bucket(errors, 50.0)}")
        print(f"<=100m: {_format_bucket(errors, 100.0)}")
        print(f"<=200m: {_format_bucket(errors, 200.0)}")
        print(f">{args.far_error_m:.0f}m: {far_errors}/{len(errors)}")

    out_path = os.path.abspath("locator_eval_results.csv")
    fieldnames = sorted({k for row in rows_out for k in row.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Saved per-query results: {out_path}")


if __name__ == "__main__":
    main()
