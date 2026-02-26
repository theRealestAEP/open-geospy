"""Modal-powered image rerank worker for retrieval accuracy.

Local process loads candidate image bytes, Modal workers compute query/candidate
similarity using a configurable CLIP backbone, and scores are merged locally.
"""

import logging
import os
import threading
from collections import deque
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import modal

LOG_LEVEL = os.getenv("GEOSPY_MODAL_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
log = logging.getLogger(__name__)

app = modal.App("geospy-retrieval-rerank")

DEFAULT_MODAL_ENVIRONMENT = os.getenv(
    "GEOSPY_MODAL_RETRIEVAL_ENVIRONMENT",
    os.getenv("GEOSPY_MODAL_EMBED_ENVIRONMENT", "google-map-walkers"),
)
DEFAULT_MODEL_NAME = os.getenv("GEOSPY_MODAL_RERANK_MODEL", "ViT-L-14")
DEFAULT_PRETRAINED = os.getenv("GEOSPY_MODAL_RERANK_PRETRAINED", "laion2b_s32b_b82k")
DEFAULT_MAX_WORKERS = max(1, int(os.getenv("GEOSPY_MODAL_RERANK_MAX_WORKERS", "64")))
DEFAULT_BATCH_SIZE = max(1, int(os.getenv("GEOSPY_MODAL_RERANK_BATCH_SIZE", "48")))
DEFAULT_MAX_RETRIES = max(0, int(os.getenv("GEOSPY_MODAL_RERANK_MAX_RETRIES", "1")))

_modal_app_context_lock = threading.Lock()
_modal_app_context_manager = None
_modal_app_context_env: Optional[str] = None
_modal_app_context_refcount = 0

rerank_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "open-clip-torch>=2.24.0",
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "kornia>=0.7.3",
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
        "opencv-python-headless>=4.10.0",
        "lightglue @ git+https://github.com/cvg/LightGlue.git",
    )
)


@contextmanager
def _shared_modal_app_context(environment_name: str):
    global _modal_app_context_manager
    global _modal_app_context_env
    global _modal_app_context_refcount

    with _modal_app_context_lock:
        if _modal_app_context_manager is None:
            manager = app.run(environment_name=environment_name)
            manager.__enter__()
            _modal_app_context_manager = manager
            _modal_app_context_env = environment_name
        elif _modal_app_context_env != environment_name:
            raise RuntimeError(
                "Modal app context already active in a different environment "
                f"({_modal_app_context_env} != {environment_name})."
            )
        _modal_app_context_refcount += 1

    try:
        yield
    finally:
        manager_to_close = None
        with _modal_app_context_lock:
            _modal_app_context_refcount = max(0, _modal_app_context_refcount - 1)
            if _modal_app_context_refcount == 0 and _modal_app_context_manager is not None:
                manager_to_close = _modal_app_context_manager
                _modal_app_context_manager = None
                _modal_app_context_env = None
        if manager_to_close is not None:
            manager_to_close.__exit__(None, None, None)


def _chunk_items(items: List[Tuple[int, bytes]], size: int) -> List[List[Tuple[int, bytes]]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


@app.function(
    image=rerank_image,
    timeout=1800,
    cpu=4.0,
    memory=8192,
)
def rerank_capture_batch(job_payload: dict) -> dict:
    import io

    from PIL import Image

    capture_ids_raw = list(job_payload.get("capture_ids") or [])
    image_bytes_raw = list(job_payload.get("image_bytes") or [])
    query_image_bytes = job_payload.get("query_image_bytes") or b""
    model_name = str(job_payload.get("model_name") or DEFAULT_MODEL_NAME).strip()
    pretrained = str(job_payload.get("pretrained") or DEFAULT_PRETRAINED).strip()

    if not capture_ids_raw or not image_bytes_raw or not query_image_bytes:
        return {"scores": [], "skipped": [], "model": {"model_name": model_name, "pretrained": pretrained}}

    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Modal retrieval rerank dependencies missing (open-clip-torch, torch)."
        ) from exc

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    fallback = ("ViT-B-32", "laion2b_s34b_b79k")
    model = None
    preprocess = None
    load_error = None
    for candidate_model, candidate_pretrained in [
        (model_name, pretrained),
        fallback,
    ]:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                candidate_model, pretrained=candidate_pretrained
            )
            model_name = candidate_model
            pretrained = candidate_pretrained
            break
        except Exception as exc:
            load_error = exc
            continue
    if model is None or preprocess is None:
        raise RuntimeError(f"Modal rerank model load failed: {load_error}")

    model.eval()
    model.to(device)

    query_image = Image.open(io.BytesIO(query_image_bytes)).convert("RGB")
    query_tensor = preprocess(query_image).unsqueeze(0).to(device)

    decoded: List[Tuple[int, object]] = []
    skipped: List[dict] = []
    for capture_id, payload in zip(capture_ids_raw, image_bytes_raw):
        try:
            decoded.append((int(capture_id), Image.open(io.BytesIO(payload)).convert("RGB")))
        except Exception as exc:
            skipped.append({"capture_id": int(capture_id), "reason": f"decode-failed:{exc}"})

    if not decoded:
        return {
            "scores": [],
            "skipped": skipped,
            "model": {"model_name": model_name, "pretrained": pretrained},
        }

    tensors = [preprocess(image) for _, image in decoded]
    batch = torch.stack(tensors, dim=0).to(device)

    with torch.no_grad():
        query_feat = model.encode_image(query_tensor)
        query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        cand_feat = model.encode_image(batch)
        cand_feat = cand_feat / cand_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        similarities = (cand_feat @ query_feat.T).squeeze(-1).detach().cpu().float().tolist()

    scores = []
    for (capture_id, _), similarity in zip(decoded, similarities):
        scores.append({"capture_id": capture_id, "similarity": float(similarity)})

    return {
        "scores": scores,
        "skipped": skipped,
        "model": {"model_name": model_name, "pretrained": pretrained},
    }


def dispatch_modal_rerank(
    *,
    query_image_bytes: bytes,
    candidate_items: Sequence[Tuple[int, bytes]],
    num_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    modal_environment: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    max_retries: int = DEFAULT_MAX_RETRIES,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    items = [(int(capture_id), image_bytes) for capture_id, image_bytes in candidate_items]
    if not items:
        return {
            "jobs_total": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "jobs_retried": 0,
            "scores": {},
            "skipped_decode": 0,
        }

    workers = max(1, min(200, int(num_workers)))
    max_retries = max(0, int(max_retries))
    environment_name = modal_environment or os.getenv("MODAL_ENVIRONMENT") or DEFAULT_MODAL_ENVIRONMENT

    jobs = deque(
        {
            "job_id": f"rerank-{idx}",
            "attempt": 0,
            "items": chunk,
        }
        for idx, chunk in enumerate(_chunk_items(items, batch_size))
    )
    jobs_total = len(jobs)
    jobs_completed = 0
    jobs_failed = 0
    jobs_retried = 0
    skipped_decode = 0
    scores: Dict[int, float] = {}

    def emit(event: dict):
        if progress_callback:
            try:
                progress_callback(event)
            except Exception:
                pass

    emit(
        {
            "event": "modal_rerank_dispatch_started",
            "jobs_total": jobs_total,
            "workers": workers,
            "batch_size": int(batch_size),
        }
    )

    with _shared_modal_app_context(environment_name):
        while jobs:
            active = []
            while jobs and len(active) < workers:
                job = jobs.popleft()
                payload = {
                    "capture_ids": [capture_id for capture_id, _ in job["items"]],
                    "image_bytes": [image_bytes for _, image_bytes in job["items"]],
                    "query_image_bytes": query_image_bytes,
                    "model_name": model_name,
                    "pretrained": pretrained,
                }
                handle = rerank_capture_batch.spawn(payload)
                active.append((job, handle))
                emit(
                    {
                        "event": "modal_rerank_job_submitted",
                        "job_id": job["job_id"],
                        "attempt": job["attempt"],
                        "capture_count": len(job["items"]),
                        "call_id": handle.object_id,
                    }
                )

            for job, handle in active:
                try:
                    payload = handle.get()
                    jobs_completed += 1
                    skipped_decode += len(list(payload.get("skipped") or []))
                    for row in list(payload.get("scores") or []):
                        capture_id = int(row.get("capture_id", 0))
                        sim = float(row.get("similarity", 0.0))
                        if capture_id <= 0:
                            continue
                        prev = scores.get(capture_id)
                        if prev is None or sim > prev:
                            scores[capture_id] = sim
                    emit(
                        {
                            "event": "modal_rerank_job_completed",
                            "job_id": job["job_id"],
                            "attempt": job["attempt"],
                            "scored": len(list(payload.get("scores") or [])),
                            "skipped_decode": len(list(payload.get("skipped") or [])),
                            "model": payload.get("model", {}),
                        }
                    )
                except Exception as exc:
                    if job["attempt"] < max_retries:
                        job["attempt"] += 1
                        jobs_retried += 1
                        jobs.append(job)
                        emit(
                            {
                                "event": "modal_rerank_job_retry_enqueued",
                                "job_id": job["job_id"],
                                "attempt": job["attempt"],
                                "error": str(exc),
                            }
                        )
                    else:
                        jobs_failed += 1
                        emit(
                            {
                                "event": "modal_rerank_job_failed",
                                "job_id": job["job_id"],
                                "attempt": job["attempt"],
                                "error": str(exc),
                            }
                        )

    result = {
        "jobs_total": jobs_total,
        "jobs_completed": jobs_completed,
        "jobs_failed": jobs_failed,
        "jobs_retried": jobs_retried,
        "scores": scores,
        "skipped_decode": skipped_decode,
    }
    emit({"event": "modal_rerank_done", **result})
    return result


@app.function(
    image=rerank_image,
    timeout=1800,
    cpu=4.0,
    memory=8192,
)
def verify_capture_batch_lightglue(job_payload: dict) -> dict:
    import io

    import cv2
    import numpy as np
    from PIL import Image

    capture_ids_raw = list(job_payload.get("capture_ids") or [])
    image_bytes_raw = list(job_payload.get("image_bytes") or [])
    query_image_bytes = job_payload.get("query_image_bytes") or b""
    max_keypoints = max(128, int(job_payload.get("max_keypoints", 2048)))
    resize_long_edge = max(256, int(job_payload.get("resize_long_edge", 1280)))
    min_matches = max(4, int(job_payload.get("min_matches", 8)))

    if not capture_ids_raw or not image_bytes_raw or not query_image_bytes:
        return {"metrics": [], "skipped": []}

    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
    except ImportError as exc:
        raise RuntimeError(
            "Modal feature verification dependencies missing (lightglue, torch)."
        ) from exc

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    def _to_tensor(image: Image.Image):
        image = image.convert("RGB")
        w, h = image.size
        long_edge = max(w, h)
        if long_edge > resize_long_edge:
            scale = float(resize_long_edge) / float(long_edge)
            nw = max(8, int(round(w * scale)))
            nh = max(8, int(round(h * scale)))
            image = image.resize((nw, nh), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        return tensor

    with torch.no_grad():
        query_image = Image.open(io.BytesIO(query_image_bytes)).convert("RGB")
        query_tensor = _to_tensor(query_image)
        query_feats = extractor.extract(query_tensor)

    metrics: List[dict] = []
    skipped: List[dict] = []

    for capture_id, payload in zip(capture_ids_raw, image_bytes_raw):
        capture_id_int = int(capture_id)
        try:
            cand_image = Image.open(io.BytesIO(payload)).convert("RGB")
            cand_tensor = _to_tensor(cand_image)
            with torch.no_grad():
                cand_feats = extractor.extract(cand_tensor)
                matches01 = matcher({"image0": query_feats, "image1": cand_feats})
                q_feats, c_feats, matches01 = [rbd(x) for x in (query_feats, cand_feats, matches01)]
            matches = matches01.get("matches")
            if matches is None:
                metrics.append(
                    {
                        "capture_id": capture_id_int,
                        "good_matches": 0.0,
                        "inliers": 0.0,
                        "inlier_ratio": 0.0,
                        "geom_score": 0.0,
                    }
                )
                continue

            good_n = int(matches.shape[0])
            if good_n < min_matches:
                metrics.append(
                    {
                        "capture_id": capture_id_int,
                        "good_matches": float(good_n),
                        "inliers": 0.0,
                        "inlier_ratio": 0.0,
                        "geom_score": 0.0,
                    }
                )
                continue

            points0 = q_feats["keypoints"][matches[:, 0]].detach().cpu().numpy().astype(np.float32)
            points1 = c_feats["keypoints"][matches[:, 1]].detach().cpu().numpy().astype(np.float32)
            inliers = 0.0
            inlier_ratio = 0.0
            if len(points0) >= 4 and len(points1) >= 4:
                _, mask = cv2.findHomography(points0, points1, cv2.RANSAC, 4.0)
                if mask is not None:
                    inliers = float(mask.ravel().sum())
                    inlier_ratio = float(inliers / float(max(1, good_n)))
            geom_score = min(1.0, float(good_n) / 120.0) * float(inlier_ratio)
            metrics.append(
                {
                    "capture_id": capture_id_int,
                    "good_matches": float(good_n),
                    "inliers": float(inliers),
                    "inlier_ratio": float(inlier_ratio),
                    "geom_score": float(max(0.0, geom_score)),
                }
            )
        except Exception as exc:
            skipped.append(
                {"capture_id": capture_id_int, "reason": f"feature-verify-failed:{exc}"}
            )

    return {"metrics": metrics, "skipped": skipped}


def dispatch_modal_feature_verify(
    *,
    query_image_bytes: bytes,
    candidate_items: Sequence[Tuple[int, bytes]],
    num_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    modal_environment: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_keypoints: int = 2048,
    resize_long_edge: int = 1280,
    min_matches: int = 8,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    items = [(int(capture_id), image_bytes) for capture_id, image_bytes in candidate_items]
    if not items:
        return {
            "jobs_total": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "jobs_retried": 0,
            "metrics": {},
            "skipped": 0,
        }

    workers = max(1, min(200, int(num_workers)))
    max_retries = max(0, int(max_retries))
    environment_name = modal_environment or os.getenv("MODAL_ENVIRONMENT") or DEFAULT_MODAL_ENVIRONMENT

    jobs = deque(
        {
            "job_id": f"feature-verify-{idx}",
            "attempt": 0,
            "items": chunk,
        }
        for idx, chunk in enumerate(_chunk_items(items, batch_size))
    )
    jobs_total = len(jobs)
    jobs_completed = 0
    jobs_failed = 0
    jobs_retried = 0
    skipped = 0
    metrics: Dict[int, dict] = {}

    def emit(event: dict):
        if progress_callback:
            try:
                progress_callback(event)
            except Exception:
                pass

    emit(
        {
            "event": "modal_feature_verify_dispatch_started",
            "jobs_total": jobs_total,
            "workers": workers,
            "batch_size": int(batch_size),
        }
    )

    with _shared_modal_app_context(environment_name):
        while jobs:
            active = []
            while jobs and len(active) < workers:
                job = jobs.popleft()
                payload = {
                    "capture_ids": [capture_id for capture_id, _ in job["items"]],
                    "image_bytes": [image_bytes for _, image_bytes in job["items"]],
                    "query_image_bytes": query_image_bytes,
                    "max_keypoints": int(max_keypoints),
                    "resize_long_edge": int(resize_long_edge),
                    "min_matches": int(min_matches),
                }
                handle = verify_capture_batch_lightglue.spawn(payload)
                active.append((job, handle))
                emit(
                    {
                        "event": "modal_feature_verify_job_submitted",
                        "job_id": job["job_id"],
                        "attempt": job["attempt"],
                        "capture_count": len(job["items"]),
                        "call_id": handle.object_id,
                    }
                )

            for job, handle in active:
                try:
                    payload = handle.get()
                    jobs_completed += 1
                    skipped += len(list(payload.get("skipped") or []))
                    for row in list(payload.get("metrics") or []):
                        capture_id = int(row.get("capture_id", 0))
                        if capture_id <= 0:
                            continue
                        existing = metrics.get(capture_id)
                        current_score = float(row.get("geom_score", 0.0))
                        if not existing or current_score >= float(existing.get("geom_score", 0.0)):
                            metrics[capture_id] = {
                                "good_matches": float(row.get("good_matches", 0.0)),
                                "inliers": float(row.get("inliers", 0.0)),
                                "inlier_ratio": float(row.get("inlier_ratio", 0.0)),
                                "geom_score": current_score,
                            }
                    emit(
                        {
                            "event": "modal_feature_verify_job_completed",
                            "job_id": job["job_id"],
                            "attempt": job["attempt"],
                            "verified": len(list(payload.get("metrics") or [])),
                            "skipped": len(list(payload.get("skipped") or [])),
                        }
                    )
                except Exception as exc:
                    if job["attempt"] < max_retries:
                        job["attempt"] += 1
                        jobs_retried += 1
                        jobs.append(job)
                        emit(
                            {
                                "event": "modal_feature_verify_job_retry_enqueued",
                                "job_id": job["job_id"],
                                "attempt": job["attempt"],
                                "error": str(exc),
                            }
                        )
                    else:
                        jobs_failed += 1
                        emit(
                            {
                                "event": "modal_feature_verify_job_failed",
                                "job_id": job["job_id"],
                                "attempt": job["attempt"],
                                "error": str(exc),
                            }
                        )

    result = {
        "jobs_total": jobs_total,
        "jobs_completed": jobs_completed,
        "jobs_failed": jobs_failed,
        "jobs_retried": jobs_retried,
        "metrics": metrics,
        "skipped": skipped,
    }
    emit({"event": "modal_feature_verify_done", **result})
    return result

