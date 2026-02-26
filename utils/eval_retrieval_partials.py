"""Evaluate retrieval robustness on synthetic partial crops of existing captures.

This harness samples captures from the local DB, generates partial query crops from
those exact images, and measures whether retrieval returns the original capture or
at least the same panorama.
"""

import argparse
import csv
import io
import os
import random
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from backend.app.clip_embeddings import get_retrieval_embedders
from backend.app.services.retrieval_locator import locate_image_bytes
from config import CrawlerConfig
from db.postgres_database import Database


def _capture_abs_path(captures_dir: str, filepath: str) -> str:
    raw = (filepath or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    normalized = raw.replace("\\", "/")
    if normalized.startswith("captures/"):
        return os.path.join(captures_dir, normalized[len("captures/") :])
    return os.path.abspath(raw)


def _read_bytes(path: str) -> Optional[bytes]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _jpeg_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=92)
    return out.getvalue()


def _crop_variants(image_bytes: bytes, variants: Sequence[str]) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        w, h = img.size

        def add_crop(name: str, box: Tuple[int, int, int, int]):
            x0, y0, x1, y1 = box
            if x1 - x0 < 24 or y1 - y0 < 24:
                return
            out.append((name, _jpeg_bytes(img.crop(box))))

        for variant in variants:
            key = variant.strip().lower()
            if key == "full":
                out.append(("full", image_bytes))
            elif key == "center80":
                cw, ch = int(w * 0.8), int(h * 0.8)
                x0, y0 = (w - cw) // 2, (h - ch) // 2
                add_crop("center80", (x0, y0, x0 + cw, y0 + ch))
            elif key == "center60":
                cw, ch = int(w * 0.6), int(h * 0.6)
                x0, y0 = (w - cw) // 2, (h - ch) // 2
                add_crop("center60", (x0, y0, x0 + cw, y0 + ch))
            elif key == "center40":
                cw, ch = int(w * 0.4), int(h * 0.4)
                x0, y0 = (w - cw) // 2, (h - ch) // 2
                add_crop("center40", (x0, y0, x0 + cw, y0 + ch))
            elif key == "left":
                add_crop("left", (0, 0, w // 2, h))
            elif key == "right":
                add_crop("right", (w // 2, 0, w, h))
            elif key == "top":
                add_crop("top", (0, 0, w, h // 2))
            elif key == "bottom":
                add_crop("bottom", (0, h // 2, w, h))
            elif key == "q1":
                add_crop("q1", (0, 0, w // 2, h // 2))
            elif key == "q2":
                add_crop("q2", (w // 2, 0, w, h // 2))
            elif key == "q3":
                add_crop("q3", (0, h // 2, w // 2, h))
            elif key == "q4":
                add_crop("q4", (w // 2, h // 2, w, h))
    dedup = {}
    for name, payload in out:
        dedup[name] = payload
    return [(k, dedup[k]) for k in dedup.keys()]


def _search_by_image_bytes(
    db: Database,
    embedders: Sequence,
    image_bytes: bytes,
    top_k: int,
    min_similarity: Optional[float],
    db_max_top_k: int,
    ivfflat_probes: int,
) -> List[dict]:
    merged: Dict[int, dict] = {}
    for embedder in embedders:
        try:
            vector = embedder.encode_image_bytes(image_bytes)
        except Exception:
            continue
        rows = db.search_captures_by_embedding(
            vector,
            embedder.model_name,
            embedder.model_version,
            top_k=max(1, min(int(db_max_top_k), int(top_k) * 12)),
            min_similarity=min_similarity,
            max_top_k=int(db_max_top_k),
            ivfflat_probes=int(ivfflat_probes),
        )
        model_weight = float(getattr(embedder, "weight", 1.0))
        model_id = str(getattr(embedder, "model_id", embedder.model_name))
        for row in rows:
            capture_id = int(row["capture_id"])
            similarity = float(row.get("similarity", 0.0))
            entry = merged.get(capture_id)
            if not entry:
                merged[capture_id] = {
                    **row,
                    "score": similarity * model_weight,
                    "model_hits": [model_id],
                }
            else:
                entry["score"] = float(entry["score"]) + (similarity * model_weight)
                if model_id not in entry["model_hits"]:
                    entry["model_hits"].append(model_id)
                if similarity > float(entry.get("similarity", 0.0)):
                    entry["similarity"] = similarity
    return sorted(merged.values(), key=lambda r: float(r.get("score", 0.0)), reverse=True)[
        : max(1, int(top_k))
    ]


def _get_rank(rows: Sequence[dict], *, capture_id: int, panorama_id: int) -> Tuple[int, int]:
    exact_rank = -1
    pano_rank = -1
    for idx, row in enumerate(rows, start=1):
        cid = int(row.get("capture_id", 0))
        pid = int(row.get("panorama_id", 0))
        if exact_rank < 0 and cid == capture_id:
            exact_rank = idx
        if pano_rank < 0 and pid == panorama_id:
            pano_rank = idx
        if exact_rank > 0 and pano_rank > 0:
            break
    return exact_rank, pano_rank


def _sample_capture_rows(
    db: Database,
    *,
    model_name: str,
    model_version: str,
    sample_size: int,
    seed: int,
) -> List[dict]:
    rows = db.conn.execute(
        """
        SELECT c.id AS capture_id, c.panorama_id, c.filepath, c.heading, c.pitch
        FROM captures c
        WHERE EXISTS (
            SELECT 1
            FROM capture_embeddings ce
            WHERE ce.capture_id = c.id
              AND ce.model_name = %s
              AND ce.model_version = %s
        )
        ORDER BY c.id
        """,
        (model_name, model_version),
    ).fetchall()
    items = [dict(r) for r in rows]
    if len(items) <= sample_size:
        return items
    random.Random(seed).shuffle(items)
    return items[:sample_size]


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (100.0 * float(numerator)) / float(denominator)


def _write_csv(path: str, rows: Iterable[dict]):
    rows = list(rows)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval hit-rate on synthetic partial crops from DB captures."
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["search", "locate", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument(
        "--variants",
        type=str,
        default="center80,center60,center40,left,right,top,bottom,q1,q2,q3,q4",
        help="Comma-separated crop variants.",
    )
    parser.add_argument("--db-max-top-k", type=int, default=5000)
    parser.add_argument("--ivfflat-probes", type=int, default=120)
    parser.add_argument("--locate-top-k-per-crop", type=int, default=220)
    parser.add_argument("--locate-max-candidates", type=int, default=5000)
    parser.add_argument("--locate-vote-cap", type=int, default=4)
    parser.add_argument("--locate-verify-top-n", type=int, default=120)
    parser.add_argument("--locate-cluster-radius-m", type=float, default=45.0)
    parser.add_argument("--locate-min-good-matches", type=int, default=11)
    parser.add_argument("--locate-min-inlier-ratio", type=float, default=0.16)
    parser.add_argument("--out-csv", default="partial_eval_results.csv")
    parser.add_argument(
        "--hard-negatives-csv",
        default="partial_eval_hard_negatives.csv",
        help="Writes rows where exact top1 failed.",
    )
    args = parser.parse_args()

    cfg = CrawlerConfig()
    captures_dir = cfg.CAPTURES_DIR
    if not os.path.isabs(captures_dir):
        captures_dir = os.path.abspath(captures_dir)

    db = Database(cfg.DATABASE_URL)
    embedders = list(get_retrieval_embedders())
    if not embedders:
        raise SystemExit("No retrieval embedders configured")

    primary = embedders[0]
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    samples = _sample_capture_rows(
        db,
        model_name=primary.model_name,
        model_version=primary.model_version,
        sample_size=max(1, int(args.sample_size)),
        seed=int(args.seed),
    )
    if not samples:
        raise SystemExit("No captures with embeddings found to evaluate.")

    rows_out: List[dict] = []
    hard_negatives: List[dict] = []

    search_exact_top1 = 0
    search_pano_top1 = 0
    search_exact_topk = 0
    search_pano_topk = 0
    locate_exact_top1 = 0
    locate_pano_top1 = 0
    locate_exact_topk = 0
    locate_pano_topk = 0
    search_exact_ranks: List[int] = []
    locate_exact_ranks: List[int] = []

    query_total = 0
    skipped_images = 0

    model_weights = {
        str(getattr(embedder, "model_id", f"model_{idx}")): float(
            getattr(embedder, "weight", 1.0)
        )
        for idx, embedder in enumerate(embedders)
    }

    try:
        for sample in samples:
            capture_id = int(sample["capture_id"])
            panorama_id = int(sample["panorama_id"])
            abs_path = _capture_abs_path(captures_dir, str(sample.get("filepath", "")))
            image_bytes = _read_bytes(abs_path)
            if image_bytes is None:
                skipped_images += 1
                continue
            for variant_name, query_bytes in _crop_variants(image_bytes, variants):
                query_total += 1
                row_base = {
                    "query_capture_id": capture_id,
                    "query_panorama_id": panorama_id,
                    "query_variant": variant_name,
                }

                if args.mode in {"search", "both"}:
                    search_rows = _search_by_image_bytes(
                        db,
                        embedders,
                        query_bytes,
                        top_k=max(1, int(args.top_k)),
                        min_similarity=args.min_similarity,
                        db_max_top_k=max(200, int(args.db_max_top_k)),
                        ivfflat_probes=max(1, int(args.ivfflat_probes)),
                    )
                    exact_rank, pano_rank = _get_rank(
                        search_rows,
                        capture_id=capture_id,
                        panorama_id=panorama_id,
                    )
                    if exact_rank == 1:
                        search_exact_top1 += 1
                    if pano_rank == 1:
                        search_pano_top1 += 1
                    if 0 < exact_rank <= int(args.top_k):
                        search_exact_topk += 1
                        search_exact_ranks.append(exact_rank)
                    if 0 < pano_rank <= int(args.top_k):
                        search_pano_topk += 1
                    top = search_rows[0] if search_rows else {}
                    rows_out.append(
                        {
                            **row_base,
                            "mode": "search",
                            "exact_rank": exact_rank,
                            "pano_rank": pano_rank,
                            "top_capture_id": int(top.get("capture_id", 0))
                            if top
                            else 0,
                            "top_panorama_id": int(top.get("panorama_id", 0))
                            if top
                            else 0,
                            "top_similarity": float(top.get("similarity", 0.0))
                            if top
                            else 0.0,
                        }
                    )
                    if exact_rank != 1 and top:
                        hard_negatives.append(
                            {
                                **row_base,
                                "mode": "search",
                                "pred_capture_id": int(top.get("capture_id", 0)),
                                "pred_panorama_id": int(top.get("panorama_id", 0)),
                                "pred_score": float(top.get("score", 0.0)),
                                "pred_similarity": float(top.get("similarity", 0.0)),
                                "expected_capture_id": capture_id,
                                "expected_panorama_id": panorama_id,
                            }
                        )

                if args.mode in {"locate", "both"}:
                    locate_result = locate_image_bytes(
                        image_bytes=query_bytes,
                        db=db,
                        embedders=embedders,
                        capture_abs_path=lambda fp: _capture_abs_path(captures_dir, fp),
                        top_k_per_crop=max(20, int(args.locate_top_k_per_crop)),
                        max_merged_candidates=max(100, int(args.locate_max_candidates)),
                        panorama_vote_cap=max(1, int(args.locate_vote_cap)),
                        cluster_radius_m=float(args.locate_cluster_radius_m),
                        verify_top_n=max(5, int(args.locate_verify_top_n)),
                        min_similarity=args.min_similarity,
                        model_weights=model_weights,
                        min_good_matches=max(4, int(args.locate_min_good_matches)),
                        min_inlier_ratio=max(0.01, float(args.locate_min_inlier_ratio)),
                        db_max_top_k=max(200, int(args.db_max_top_k)),
                        ivfflat_probes=max(1, int(args.ivfflat_probes)),
                        include_debug=False,
                    )
                    support = locate_result.get("supporting_matches") or []
                    exact_rank, pano_rank = _get_rank(
                        support,
                        capture_id=capture_id,
                        panorama_id=panorama_id,
                    )
                    if exact_rank == 1:
                        locate_exact_top1 += 1
                    if pano_rank == 1:
                        locate_pano_top1 += 1
                    if 0 < exact_rank <= int(args.top_k):
                        locate_exact_topk += 1
                        locate_exact_ranks.append(exact_rank)
                    if 0 < pano_rank <= int(args.top_k):
                        locate_pano_topk += 1
                    top = support[0] if support else {}
                    rows_out.append(
                        {
                            **row_base,
                            "mode": "locate",
                            "exact_rank": exact_rank,
                            "pano_rank": pano_rank,
                            "top_capture_id": int(top.get("capture_id", 0))
                            if top
                            else 0,
                            "top_panorama_id": int(top.get("panorama_id", 0))
                            if top
                            else 0,
                            "top_score": float(top.get("score", 0.0)) if top else 0.0,
                            "flags": ",".join(locate_result.get("flags") or []),
                        }
                    )
                    if exact_rank != 1 and top:
                        hard_negatives.append(
                            {
                                **row_base,
                                "mode": "locate",
                                "pred_capture_id": int(top.get("capture_id", 0)),
                                "pred_panorama_id": int(top.get("panorama_id", 0)),
                                "pred_score": float(top.get("score", 0.0)),
                                "pred_similarity": float(top.get("similarity", 0.0)),
                                "expected_capture_id": capture_id,
                                "expected_panorama_id": panorama_id,
                            }
                        )
    finally:
        db.close()

    print(f"sampled_captures={len(samples)}")
    print(f"query_total={query_total}")
    print(f"skipped_images={skipped_images}")
    if args.mode in {"search", "both"} and query_total > 0:
        print(
            "search "
            f"exact_top1={search_exact_top1}/{query_total} ({_percent(search_exact_top1, query_total):.2f}%) "
            f"pano_top1={search_pano_top1}/{query_total} ({_percent(search_pano_top1, query_total):.2f}%) "
            f"exact_top{args.top_k}={search_exact_topk}/{query_total} ({_percent(search_exact_topk, query_total):.2f}%) "
            f"pano_top{args.top_k}={search_pano_topk}/{query_total} ({_percent(search_pano_topk, query_total):.2f}%)"
        )
        if search_exact_ranks:
            print(f"search_median_exact_rank={median(search_exact_ranks):.2f}")
    if args.mode in {"locate", "both"} and query_total > 0:
        print(
            "locate "
            f"exact_top1={locate_exact_top1}/{query_total} ({_percent(locate_exact_top1, query_total):.2f}%) "
            f"pano_top1={locate_pano_top1}/{query_total} ({_percent(locate_pano_top1, query_total):.2f}%) "
            f"exact_top{args.top_k}={locate_exact_topk}/{query_total} ({_percent(locate_exact_topk, query_total):.2f}%) "
            f"pano_top{args.top_k}={locate_pano_topk}/{query_total} ({_percent(locate_pano_topk, query_total):.2f}%)"
        )
        if locate_exact_ranks:
            print(f"locate_median_exact_rank={median(locate_exact_ranks):.2f}")

    out_csv = os.path.abspath(args.out_csv)
    hard_csv = os.path.abspath(args.hard_negatives_csv)
    _write_csv(out_csv, rows_out)
    _write_csv(hard_csv, hard_negatives)
    print(f"saved_results={out_csv}")
    print(f"saved_hard_negatives={hard_csv}")


if __name__ == "__main__":
    main()

