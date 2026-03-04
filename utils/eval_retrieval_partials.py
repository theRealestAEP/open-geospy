"""Evaluate search-only retrieval robustness on synthetic partial crops."""

import argparse
import csv
import io
import os
import random
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from backend.app.clip_embeddings import get_retrieval_embedders
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

        def add_crop(name: str, box: Tuple[int, int, int, int]) -> None:
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
            embedding_base=str(getattr(embedder, "embedding_base", "clip")),
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


def _write_csv(path: str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate search-only retrieval hit-rate on synthetic partial crops."
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--out-csv", default="partial_eval_results.csv")
    parser.add_argument(
        "--hard-negatives-csv",
        default="partial_eval_hard_negatives.csv",
        help="Writes rows where exact top1 failed.",
    )
    args = parser.parse_args()

    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    if not variants:
        variants = ["center60", "center40", "left", "right", "top", "bottom"]

    cfg = CrawlerConfig()
    captures_dir = cfg.CAPTURES_DIR
    if not os.path.isabs(captures_dir):
        captures_dir = os.path.abspath(captures_dir)

    db = Database(cfg.DATABASE_URL)
    embedders = list(get_retrieval_embedders())
    if not embedders:
        raise SystemExit("No retrieval embedders configured.")

    primary = embedders[0]
    sample_rows = _sample_capture_rows(
        db,
        model_name=primary.model_name,
        model_version=primary.model_version,
        sample_size=max(1, int(args.sample_size)),
        seed=int(args.seed),
    )
    if not sample_rows:
        raise SystemExit("No capture rows available for evaluation.")

    result_rows: List[dict] = []
    hard_rows: List[dict] = []

    exact_top1 = 0
    same_pano_top1 = 0
    exact_topk = 0
    same_pano_topk = 0
    total_cases = 0

    for row in sample_rows:
        capture_id = int(row["capture_id"])
        panorama_id = int(row["panorama_id"])
        image_bytes = _read_bytes(_capture_abs_path(captures_dir, str(row.get("filepath", ""))))
        if not image_bytes:
            continue
        crops = _crop_variants(image_bytes, variants)
        for crop_name, crop_bytes in crops:
            total_cases += 1
            ranked = _search_by_image_bytes(
                db=db,
                embedders=embedders,
                image_bytes=crop_bytes,
                top_k=max(1, int(args.top_k)),
                min_similarity=args.min_similarity,
                db_max_top_k=max(200, int(args.db_max_top_k)),
                ivfflat_probes=max(1, int(args.ivfflat_probes)),
            )
            exact_rank, pano_rank = _get_rank(
                ranked, capture_id=capture_id, panorama_id=panorama_id
            )
            if exact_rank == 1:
                exact_top1 += 1
            if pano_rank == 1:
                same_pano_top1 += 1
            if 0 < exact_rank <= int(args.top_k):
                exact_topk += 1
            if 0 < pano_rank <= int(args.top_k):
                same_pano_topk += 1

            top_row = ranked[0] if ranked else {}
            out = {
                "capture_id": capture_id,
                "panorama_id": panorama_id,
                "crop_name": crop_name,
                "rank_exact": exact_rank,
                "rank_panorama": pano_rank,
                "top_capture_id": int(top_row.get("capture_id", 0)) if top_row else 0,
                "top_panorama_id": int(top_row.get("panorama_id", 0)) if top_row else 0,
                "top_similarity": float(top_row.get("similarity", 0.0)) if top_row else 0.0,
            }
            result_rows.append(out)
            if exact_rank != 1:
                hard_rows.append(out)

    _write_csv(args.out_csv, result_rows)
    _write_csv(args.hard_negatives_csv, hard_rows)

    print(f"cases={total_cases}")
    print(
        "search exact@1={:.2f}% pano@1={:.2f}% exact@{}={:.2f}% pano@{}={:.2f}%".format(
            _percent(exact_top1, total_cases),
            _percent(same_pano_top1, total_cases),
            int(args.top_k),
            _percent(exact_topk, total_cases),
            int(args.top_k),
            _percent(same_pano_topk, total_cases),
        )
    )
    if result_rows:
        rank_values = [r["rank_exact"] for r in result_rows if int(r["rank_exact"]) > 0]
        if rank_values:
            print(f"median_exact_rank={median(rank_values):.2f}")
    print(f"results_csv={args.out_csv}")
    print(f"hard_negatives_csv={args.hard_negatives_csv}")


if __name__ == "__main__":
    main()

