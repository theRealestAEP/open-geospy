"""Backfill retrieval embeddings for captures into pgvector."""

import argparse
import os
from typing import Dict, List, Sequence, Tuple

from backend.app.clip_embeddings import (
    encode_image_batch_for_all_models,
    get_retrieval_embedders,
)
from config import CrawlerConfig
from db.postgres_database import Database


def resolve_capture_path(raw_path: str, captures_dir: str, project_root: str) -> str:
    raw = (raw_path or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    normalized = raw.replace("\\", "/")
    if normalized.startswith("captures/"):
        return os.path.join(captures_dir, normalized[len("captures/") :])
    return os.path.join(project_root, raw)


def _upsert_single_capture(
    db: Database,
    embedders: Sequence,
    capture_id: int,
    image_bytes: bytes,
) -> bool:
    encoded_any = False
    for embedder in embedders:
        try:
            vector = embedder.encode_image_bytes(image_bytes)
            db.upsert_capture_embedding(
                int(capture_id),
                embedder.model_name,
                embedder.model_version,
                vector,
            )
            encoded_any = True
        except Exception as exc:
            print(
                "skip model encode "
                f"capture_id={capture_id} model={embedder.model_name}:{embedder.model_version} "
                f"error={exc}"
            )
    return encoded_any


def _process_local_batch(
    db: Database,
    embedders: Sequence,
    valid_batch: List[Tuple[int, bytes]],
) -> Tuple[int, int]:
    indexed = 0
    skipped = 0
    try:
        model_batches = encode_image_batch_for_all_models(
            [payload for _, payload in valid_batch]
        )
        if not model_batches:
            raise RuntimeError("No retrieval models encoded successfully")
        for embedder, vectors in model_batches:
            if len(vectors) != len(valid_batch):
                raise RuntimeError(
                    f"batch embedding size mismatch for {embedder.model_id} "
                    f"expected={len(valid_batch)} got={len(vectors)}"
                )
            db.upsert_capture_embeddings_batch(
                [
                    (capture_id, vector)
                    for (capture_id, _), vector in zip(valid_batch, vectors)
                ],
                embedder.model_name,
                embedder.model_version,
            )
        for capture_id, _ in valid_batch:
            indexed += 1
            print(f"indexed capture_id={capture_id} total_indexed_increment=1")
    except Exception as exc:
        print(f"batch failed; falling back to single-image mode error={exc}")
        for capture_id, image_bytes in valid_batch:
            if _upsert_single_capture(db, embedders, capture_id, image_bytes):
                indexed += 1
                print(f"indexed capture_id={capture_id} total_indexed_increment=1")
            else:
                skipped += 1
                print(f"skip capture_id={capture_id} error=all-models-failed")
    return indexed, skipped


def _process_modal_batch(
    db: Database,
    embedders: Sequence,
    valid_batch: List[Tuple[int, bytes]],
    num_workers: int,
    worker_batch_size: int,
    modal_environment: str,
    max_retries: int,
    fallback_local: bool,
) -> Tuple[int, int]:
    from worker.modal_embedding_worker import dispatch_embedding_jobs

    modal_models = [
        {
            "model_id": str(embedder.model_id),
            "model_name": str(embedder.model_name),
            "pretrained": str(embedder.pretrained),
            "model_version": str(embedder.model_version),
        }
        for embedder in embedders
    ]
    image_by_capture_id: Dict[int, bytes] = {
        int(capture_id): image_bytes for capture_id, image_bytes in valid_batch
    }

    indexed_capture_ids = set()
    skipped = 0

    def on_result(payload: dict):
        nonlocal skipped
        capture_ids = [int(v) for v in list(payload.get("capture_ids") or [])]
        model_outputs = list(payload.get("model_outputs") or [])
        decode_skips = list(payload.get("skipped") or [])
        model_errors = list(payload.get("model_errors") or [])
        skipped += len(decode_skips)
        for row in decode_skips:
            print(
                f"skip capture_id={row.get('capture_id')} "
                f"reason={row.get('reason', 'decode-failed')}"
            )
        for row in model_errors:
            print(
                "modal model error "
                f"model={row.get('model_name')}:{row.get('model_version')} "
                f"error={row.get('error')}"
            )
        if not capture_ids:
            return
        encoded_any = False
        for model_output in model_outputs:
            vectors = list(model_output.get("vectors") or [])
            if len(vectors) != len(capture_ids):
                raise RuntimeError(
                    "modal batch embedding size mismatch "
                    f"model={model_output.get('model_name')}:{model_output.get('model_version')} "
                    f"expected={len(capture_ids)} got={len(vectors)}"
                )
            db.upsert_capture_embeddings_batch(
                [
                    (capture_id, vector)
                    for capture_id, vector in zip(capture_ids, vectors)
                ],
                str(model_output.get("model_name") or ""),
                str(model_output.get("model_version") or ""),
            )
            encoded_any = True
        if encoded_any:
            indexed_capture_ids.update(capture_ids)
            return
        skipped += len(capture_ids)
        for capture_id in capture_ids:
            print(f"skip capture_id={capture_id} error=all-models-failed-modal")

    def on_progress(event: dict):
        event_name = str(event.get("event") or "")
        if event_name == "dispatch_started":
            print(
                "modal embedding dispatch started "
                f"jobs={event.get('jobs_total')} workers={event.get('workers')} "
                f"batch_size={event.get('batch_size')}"
            )
        elif event_name == "job_failed":
            print(
                f"modal embedding job failed job_id={event.get('job_id')} "
                f"attempt={event.get('attempt')} error={event.get('error')}"
            )
        elif event_name == "all_done":
            print(
                "modal embedding dispatch done "
                f"jobs_completed={event.get('jobs_completed')} jobs_failed={event.get('jobs_failed')} "
                f"jobs_retried={event.get('jobs_retried')}"
            )

    summary = dispatch_embedding_jobs(
        capture_items=valid_batch,
        model_configs=modal_models,
        num_workers=num_workers,
        batch_size=worker_batch_size,
        modal_environment=modal_environment,
        max_retries=max_retries,
        progress_callback=on_progress,
        result_callback=on_result,
    )

    failed_capture_ids = [int(v) for v in list(summary.get("failed_capture_ids") or [])]
    if failed_capture_ids and fallback_local:
        print(
            "modal failed captures fallback to local "
            f"count={len(failed_capture_ids)}"
        )
        for capture_id in failed_capture_ids:
            image_bytes = image_by_capture_id.get(int(capture_id))
            if not image_bytes:
                skipped += 1
                print(f"skip capture_id={capture_id} error=missing-image-for-fallback")
                continue
            if _upsert_single_capture(db, embedders, capture_id, image_bytes):
                indexed_capture_ids.add(capture_id)
                print(f"indexed capture_id={capture_id} source=local-fallback")
            else:
                skipped += 1
                print(f"skip capture_id={capture_id} error=all-models-failed")
    elif failed_capture_ids:
        skipped += len(failed_capture_ids)
        for capture_id in failed_capture_ids:
            print(f"skip capture_id={capture_id} error=modal-job-failed")

    indexed = len(indexed_capture_ids)
    if indexed:
        print(
            "modal batch indexed "
            f"captures={indexed} skipped={skipped} "
            f"jobs_failed={summary.get('jobs_failed', 0)}"
        )
    return indexed, skipped


def main():
    parser = argparse.ArgumentParser(description="Index capture images into pgvector.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["local", "modal"],
        default="local",
        help="Embedding compute mode. local=run CLIP locally, modal=run CLIP inside Modal workers.",
    )
    parser.add_argument(
        "--modal-workers",
        type=int,
        default=max(1, int(os.getenv("GEOSPY_MODAL_EMBED_MAX_WORKERS", "16"))),
        help="Max parallel Modal embedding containers (capped at 100).",
    )
    parser.add_argument(
        "--modal-worker-batch-size",
        type=int,
        default=max(1, int(os.getenv("GEOSPY_MODAL_EMBED_BATCH_SIZE", "64"))),
        help="Capture count per Modal embedding call.",
    )
    parser.add_argument(
        "--modal-max-retries",
        type=int,
        default=max(0, int(os.getenv("GEOSPY_MODAL_EMBED_MAX_RETRIES", "1"))),
        help="Retries for failed Modal embedding calls.",
    )
    parser.add_argument(
        "--modal-environment",
        type=str,
        default=os.getenv("GEOSPY_MODAL_EMBED_ENVIRONMENT", ""),
        help="Modal environment name (falls back to MODAL_ENVIRONMENT then google-map-walkers).",
    )
    parser.add_argument(
        "--no-modal-fallback-local",
        action="store_true",
        help="Disable fallback to local embedding for Modal-failed capture batches.",
    )
    args = parser.parse_args()

    config = CrawlerConfig()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    captures_dir = config.CAPTURES_DIR
    if not os.path.isabs(captures_dir):
        captures_dir = os.path.join(project_root, captures_dir)

    embedders = list(get_retrieval_embedders())
    if not embedders:
        raise SystemExit("No retrieval models configured.")
    primary_embedder = embedders[0]
    db = Database(config.DATABASE_URL)
    if not db.is_vector_ready():
        raise SystemExit(
            "Vector extension is unavailable. Start Postgres with pgvector enabled."
        )

    total_indexed = 0
    total_skipped = 0
    after_capture_id = 0
    max_items = max(0, int(args.max_items))
    batch_size = max(1, int(args.batch_size))
    modal_mode = str(args.mode) == "modal"
    modal_workers = max(1, min(100, int(args.modal_workers)))
    modal_worker_batch_size = max(1, int(args.modal_worker_batch_size))
    modal_max_retries = max(0, int(args.modal_max_retries))
    modal_environment = (args.modal_environment or "").strip()
    fallback_local = not bool(args.no_modal_fallback_local)

    if modal_mode:
        print(
            "embedding mode=modal "
            f"workers={modal_workers} worker_batch_size={modal_worker_batch_size} "
            f"fallback_local={fallback_local}"
        )
    else:
        print("embedding mode=local")

    try:
        while True:
            if max_items and total_indexed >= max_items:
                break

            remaining = max_items - total_indexed if max_items else 0
            if modal_mode:
                fetch_limit = max(batch_size, modal_workers * modal_worker_batch_size)
                if max_items:
                    fetch_limit = min(fetch_limit, remaining)
            else:
                fetch_limit = batch_size if not max_items else min(batch_size, remaining)

            if fetch_limit <= 0:
                break

            rows = db.list_captures_missing_any_embeddings(
                [
                    (embedder.model_name, embedder.model_version)
                    for embedder in embedders
                ],
                limit=fetch_limit,
                after_capture_id=after_capture_id,
            )
            if not rows:
                break

            valid_batch: List[Tuple[int, bytes]] = []
            for row in rows:
                after_capture_id = int(row["capture_id"])
                capture_path = resolve_capture_path(
                    row.get("filepath", ""), captures_dir, project_root
                )
                if not capture_path or not os.path.exists(capture_path):
                    total_skipped += 1
                    print(f"skip capture_id={row['capture_id']} missing file")
                    continue
                try:
                    with open(capture_path, "rb") as f:
                        image_bytes = f.read()
                    valid_batch.append((int(row["capture_id"]), image_bytes))
                except Exception as exc:
                    total_skipped += 1
                    print(f"skip capture_id={row['capture_id']} error={exc}")

            if not valid_batch:
                continue

            if modal_mode:
                indexed, skipped = _process_modal_batch(
                    db=db,
                    embedders=embedders,
                    valid_batch=valid_batch,
                    num_workers=modal_workers,
                    worker_batch_size=modal_worker_batch_size,
                    modal_environment=modal_environment,
                    max_retries=modal_max_retries,
                    fallback_local=fallback_local,
                )
            else:
                indexed, skipped = _process_local_batch(
                    db=db,
                    embedders=embedders,
                    valid_batch=valid_batch,
                )
            total_indexed += int(indexed)
            total_skipped += int(skipped)
            print(f"progress indexed={total_indexed} skipped={total_skipped}")
    finally:
        stats = db.get_capture_embedding_stats(
            primary_embedder.model_name, primary_embedder.model_version
        )
        model_stats = [
            db.get_capture_embedding_stats(embedder.model_name, embedder.model_version)
            for embedder in embedders
        ]
        db.close()
        print(
            "done "
            f"indexed={total_indexed} skipped={total_skipped} "
            f"embedded={stats['embedded_captures']} pending={stats['pending_captures']}"
        )
        print("models:")
        for item in model_stats:
            print(
                f"  {item.get('model_name')} {item.get('model_version')} "
                f"embedded={item.get('embedded_captures')} pending={item.get('pending_captures')}"
            )


if __name__ == "__main__":
    main()
