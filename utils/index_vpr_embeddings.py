"""Backfill VPR (EigenPlaces) embeddings for a subset of captures into LanceDB.

The full index has ~197 views per panorama (5.5M captures); embedding all of
them with a 2048-d VPR model is unnecessary for validation. This script embeds
a representative subset (default: pitches 75/90, headings every 90 degrees,
~8 views per pano) and writes them to the per-base LanceDB table
(capture_embeddings__vpr) via the vector store.

Run with:
    GEOSPY_VECTOR_BACKEND=lancedb GEOSPY_VPR_MODEL_ENABLED=1 \
      python -m utils.index_vpr_embeddings --batch-size 32
"""

import argparse
import os
import time
from typing import List, Set, Tuple

from backend.app.clip_embeddings import EMBEDDING_BASE_VPR, select_retrieval_embedders
from backend.app.vector_store import build_vector_store
from config import CrawlerConfig
from db.postgres_database import Database
from utils.index_capture_embeddings import resolve_capture_path


def _existing_capture_ids(vector_store, embedder) -> Set[int]:
    """Best-effort read of already-embedded capture ids for resume support."""
    try:
        table = vector_store._open_table_or_none(
            vector_store._table_name_for_base(EMBEDDING_BASE_VPR)
        )
        if table is None:
            return set()
        dataset = table.to_lance()
        column = dataset.to_table(columns=["capture_id"]).column("capture_id")
        return {int(v.as_py()) for v in column}
    except Exception as exc:
        print(f"resume_detection_failed error={exc}; starting from --after-capture-id")
        return set()


def _select_captures(
    db: Database,
    *,
    capture_profile: str,
    pitches: List[float],
    heading_step: int,
    after_capture_id: int,
    max_items: int,
) -> List[Tuple[int, str]]:
    rows = db.conn.execute(
        """
        SELECT id, filepath
        FROM captures
        WHERE capture_profile = %s
          AND pitch = ANY(%s)
          AND MOD(heading::numeric, %s) = 0
          AND id > %s
        ORDER BY id
        """,
        (capture_profile, pitches, heading_step, after_capture_id),
    ).fetchall()
    out = [(int(row["id"]), str(row["filepath"] or "")) for row in rows]
    if max_items > 0:
        out = out[:max_items]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill VPR embeddings into LanceDB.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--capture-profile", default="high_v1")
    parser.add_argument("--pitches", default="75,90", help="Comma-separated pitch values.")
    parser.add_argument("--heading-step", type=int, default=90)
    parser.add_argument("--after-capture-id", type=int, default=0)
    args = parser.parse_args()

    embedders = select_retrieval_embedders(EMBEDDING_BASE_VPR, allow_fallback=False)
    if not embedders:
        raise SystemExit(
            "No VPR embedder configured. Run with GEOSPY_VPR_MODEL_ENABLED=1."
        )
    embedder = embedders[0]

    cfg = CrawlerConfig()
    captures_dir = os.path.abspath(cfg.CAPTURES_DIR)
    project_root = os.path.dirname(os.path.abspath(__file__ + "/.."))

    db = Database(cfg.DATABASE_URL)
    vector_store = build_vector_store(db)
    if vector_store.backend_name != "lancedb":
        raise SystemExit("Run with GEOSPY_VECTOR_BACKEND=lancedb.")

    pitches = [float(v) for v in args.pitches.split(",") if v.strip()]
    candidates = _select_captures(
        db,
        capture_profile=args.capture_profile,
        pitches=pitches,
        heading_step=max(1, int(args.heading_step)),
        after_capture_id=int(args.after_capture_id),
        max_items=int(args.max_items),
    )
    existing = _existing_capture_ids(vector_store, embedder)
    pending = [(cid, fp) for cid, fp in candidates if cid not in existing]
    print(
        f"candidates={len(candidates)} already_embedded={len(existing)} "
        f"pending={len(pending)} model={embedder.model_name}:{embedder.model_version}"
    )

    indexed = 0
    skipped = 0
    started = time.time()
    batch: List[Tuple[int, bytes]] = []

    def flush():
        nonlocal indexed, skipped, batch
        if not batch:
            return
        try:
            vectors = embedder.encode_image_bytes_batch([payload for _, payload in batch])
            vector_store.upsert_capture_embeddings_batch(
                [(cid, vec) for (cid, _), vec in zip(batch, vectors)],
                embedder.model_name,
                embedder.model_version,
                embedding_base=EMBEDDING_BASE_VPR,
            )
            indexed += len(batch)
        except Exception as exc:
            skipped += len(batch)
            print(f"batch_failed last_capture_id={batch[-1][0]} error={exc}")
        rate = indexed / max(time.time() - started, 1e-9)
        print(
            f"progress indexed={indexed} skipped={skipped} pending={len(pending)} "
            f"rate_img_s={rate:.1f} last_capture_id={batch[-1][0]}",
            flush=True,
        )
        batch = []

    for capture_id, filepath in pending:
        path = resolve_capture_path(filepath, captures_dir, project_root)
        if not path or not os.path.exists(path):
            skipped += 1
            continue
        try:
            with open(path, "rb") as fh:
                batch.append((capture_id, fh.read()))
        except OSError as exc:
            skipped += 1
            print(f"skip capture_id={capture_id} error={exc}")
            continue
        if len(batch) >= max(1, int(args.batch_size)):
            flush()
    flush()

    db.close()
    elapsed = time.time() - started
    print(f"done indexed={indexed} skipped={skipped} elapsed_s={elapsed:.0f}")


if __name__ == "__main__":
    main()
