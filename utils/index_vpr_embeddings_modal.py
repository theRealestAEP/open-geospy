"""Full-density VPR (EigenPlaces) backfill using Modal GPUs.

Workers read images directly from the `geospy-captures` Modal Volume (populated
by utils.modal_volume_upload), embed them on GPU, and return vectors; the local
driver selects one capture id per distinct filepath (skipping duplicate rows),
dispatches jobs, and writes results into the per-base LanceDB table.

Run with:
    GEOSPY_VECTOR_BACKEND=lancedb GEOSPY_VPR_MODEL_ENABLED=1 \
      python -m utils.index_vpr_embeddings_modal --limit 256   # smoke test
    GEOSPY_VECTOR_BACKEND=lancedb GEOSPY_VPR_MODEL_ENABLED=1 \
      python -m utils.index_vpr_embeddings_modal               # everything
"""

import argparse
import os
import time
from collections import deque
from typing import Dict, List, Set, Tuple

try:
    from env_bootstrap import load_project_env
except ModuleNotFoundError:
    load_project_env = None

if load_project_env is not None:
    load_project_env()

import modal

VOLUME_NAME = os.getenv("GEOSPY_MODAL_CAPTURES_VOLUME", "geospy-captures")
VOLUME_MOUNT = "/vol"
GPU_TYPE = os.getenv("GEOSPY_MODAL_VPR_GPU", "A10G")
DEFAULT_ENVIRONMENT = (
    os.getenv("GEOSPY_MODAL_EMBED_ENVIRONMENT")
    or os.getenv("MODAL_ENVIRONMENT")
    or "google-map-walkers"
)

app = modal.App("geospy-vpr-backfill")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
vpr_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.3.0",
    "torchvision>=0.18.0",
    "Pillow>=10.0.0",
    "numpy",
)


@app.function(
    image=vpr_image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: volume},
    timeout=1800,
)
def embed_volume_batch(job: dict) -> dict:
    """Embed images stored on the volume. job = {relpaths, hub_repo, backbone, output_dim}."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    relpaths: List[str] = list(job["relpaths"])
    hub_repo: str = job["hub_repo"]
    backbone: str = job["backbone"]
    output_dim: int = int(job["output_dim"])
    image_size: int = int(job.get("image_size", 512))
    micro_batch: int = int(job.get("micro_batch", 64))

    # EigenPlaces' hubconf internally torch.hub.loads cosplace without
    # trust_repo, so pre-trust both repos to avoid the interactive prompt.
    hub_dir = torch.hub.get_dir()
    os.makedirs(hub_dir, exist_ok=True)
    with open(os.path.join(hub_dir, "trusted_list"), "a") as fh:
        fh.write("gmberton_eigenplaces\ngmberton_cosplace\n")

    model = torch.hub.load(
        hub_repo,
        "get_trained_model",
        backbone=backbone,
        fc_output_dim=output_dim,
        trust_repo=True,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ok_relpaths: List[str] = []
    skipped: List[str] = []
    chunks: List[np.ndarray] = []
    tensors: List[torch.Tensor] = []

    def flush_micro():
        if not tensors:
            return
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            feats = model(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        chunks.append(feats.cpu().float().numpy())
        tensors.clear()

    for relpath in relpaths:
        path = os.path.join(VOLUME_MOUNT, relpath)
        try:
            with Image.open(path) as img:
                tensors.append(preprocess(img.convert("RGB")))
            ok_relpaths.append(relpath)
        except Exception:
            skipped.append(relpath)
            continue
        if len(tensors) >= micro_batch:
            flush_micro()
    flush_micro()

    vectors = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, output_dim), "f4")
    return {"relpaths": ok_relpaths, "vectors": vectors, "skipped": skipped}


def _embedded_filepaths(vector_store, db) -> Set[str]:
    """Filepaths already in the VPR table (via their capture ids)."""
    from backend.app.clip_embeddings import EMBEDDING_BASE_VPR

    try:
        table = vector_store._open_table_or_none(
            vector_store._table_name_for_base(EMBEDDING_BASE_VPR)
        )
        if table is None:
            return set()
        arrow = table.search().select(["capture_id"]).limit(1_000_000_000).to_arrow()
        ids = [int(v.as_py()) for v in arrow.column("capture_id")]
    except Exception as exc:
        print(f"resume_detection_failed error={exc}")
        return set()

    out: Set[str] = set()
    for i in range(0, len(ids), 50000):
        rows = db.conn.execute(
            "SELECT filepath FROM captures WHERE id = ANY(%s)", (ids[i : i + 50000],)
        ).fetchall()
        out.update(str(r["filepath"]) for r in rows if r["filepath"])
    return out


def main() -> None:
    from backend.app.clip_embeddings import EMBEDDING_BASE_VPR, select_retrieval_embedders
    from backend.app.vector_store import build_vector_store
    from config import CrawlerConfig
    from db.postgres_database import Database

    parser = argparse.ArgumentParser(description="Full VPR backfill on Modal GPUs.")
    parser.add_argument("--job-size", type=int, default=512, help="Images per GPU job.")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent GPU containers.")
    parser.add_argument("--limit", type=int, default=0, help="Max images (0 = all).")
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    embedders = select_retrieval_embedders(EMBEDDING_BASE_VPR, allow_fallback=False)
    if not embedders:
        raise SystemExit("No VPR embedder configured. Run with GEOSPY_VPR_MODEL_ENABLED=1.")
    embedder = embedders[0]
    backbone, _, dim = embedder.pretrained.partition(":")

    db = Database(CrawlerConfig().DATABASE_URL)
    vector_store = build_vector_store(db)
    if vector_store.backend_name != "lancedb":
        raise SystemExit("Run with GEOSPY_VECTOR_BACKEND=lancedb.")

    print("selecting distinct filepaths ...", flush=True)
    rows = db.conn.execute(
        """
        SELECT DISTINCT ON (filepath) id, filepath
        FROM captures
        WHERE filepath IS NOT NULL AND filepath <> ''
        ORDER BY filepath, id
        """
    ).fetchall()
    candidates: List[Tuple[int, str]] = [(int(r["id"]), str(r["filepath"])) for r in rows]
    embedded = _embedded_filepaths(vector_store, db)
    pending = [(cid, fp) for cid, fp in candidates if fp not in embedded]
    if args.limit > 0:
        pending = pending[: args.limit]
    print(
        f"distinct_files={len(candidates)} already_embedded={len(embedded)} "
        f"pending={len(pending)} model={embedder.model_name}:{embedder.model_version} "
        f"gpu={GPU_TYPE} workers={args.workers}",
        flush=True,
    )
    if not pending:
        db.close()
        return

    id_by_relpath: Dict[str, int] = {fp: cid for cid, fp in pending}
    from backend.app.clip_embeddings import DEFAULT_VPR_IMAGE_SIZE

    job_payload_base = {
        "hub_repo": embedder.model_name,
        "backbone": backbone,
        "output_dim": int(dim or 2048),
        "image_size": DEFAULT_VPR_IMAGE_SIZE,
    }
    jobs = deque(
        {"relpaths": [fp for _, fp in pending[i : i + args.job_size]], "attempt": 0}
        for i in range(0, len(pending), args.job_size)
    )

    indexed = 0
    skipped = 0
    failed_jobs = 0
    started = time.time()

    def handle_result(payload: dict):
        nonlocal indexed, skipped
        relpaths = payload["relpaths"]
        vectors = payload["vectors"]
        skipped += len(payload["skipped"])
        batch = [
            (id_by_relpath[rp], vectors[i].tolist())
            for i, rp in enumerate(relpaths)
            if rp in id_by_relpath
        ]
        if batch:
            vector_store.upsert_capture_embeddings_batch(
                batch,
                embedder.model_name,
                embedder.model_version,
                embedding_base=EMBEDDING_BASE_VPR,
                assume_new=True,
            )
            indexed += len(batch)
        rate = indexed / max(time.time() - started, 1e-9)
        print(
            f"progress indexed={indexed} skipped={skipped} pending={len(pending)} "
            f"rate_img_s={rate:.1f}",
            flush=True,
        )

    with app.run(environment_name=args.environment):
        while jobs:
            active = []
            while jobs and len(active) < args.workers:
                job = jobs.popleft()
                handle = embed_volume_batch.spawn({**job_payload_base, **job})
                active.append((job, handle))
            for job, handle in active:
                try:
                    handle_result(handle.get())
                except Exception as exc:
                    if job["attempt"] < args.retries:
                        job["attempt"] += 1
                        jobs.append(job)
                        print(f"job_retry error={exc}", flush=True)
                    else:
                        failed_jobs += 1
                        print(f"job_failed count={len(job['relpaths'])} error={exc}", flush=True)

    db.close()
    elapsed = time.time() - started
    print(
        f"done indexed={indexed} skipped={skipped} failed_jobs={failed_jobs} "
        f"elapsed_s={elapsed:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
