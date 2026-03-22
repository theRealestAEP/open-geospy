"""
Modal-based embedding worker for high-parallel backfill jobs.

The local process reads image bytes and writes vectors to Postgres.
Modal workers only handle model inference and return vectors.
"""

import logging
import os
import threading
from collections import deque
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    from env_bootstrap import load_project_env
except ModuleNotFoundError:
    load_project_env = None

if load_project_env is not None:
    load_project_env()

import modal

LOG_LEVEL = os.getenv("GEOSPY_MODAL_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
log = logging.getLogger(__name__)

app = modal.App("geospy-embeddings")
DEFAULT_MODAL_ENVIRONMENT = os.getenv("GEOSPY_MODAL_EMBED_ENVIRONMENT", "google-map-walkers")
DEFAULT_MAX_WORKERS = max(1, int(os.getenv("GEOSPY_MODAL_EMBED_MAX_WORKERS", "64")))
DEFAULT_BATCH_SIZE = max(1, int(os.getenv("GEOSPY_MODAL_EMBED_BATCH_SIZE", "64")))
DEFAULT_MAX_RETRIES = max(0, int(os.getenv("GEOSPY_MODAL_EMBED_MAX_RETRIES", "1")))
MODAL_HF_SECRET_NAME = os.getenv("GEOSPY_MODAL_HF_SECRET_NAME", "").strip()

_modal_app_context_lock = threading.Lock()
_modal_app_context_manager = None
_modal_app_context_env: Optional[str] = None
_modal_app_context_refcount = 0

embedding_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "open-clip-torch>=2.24.0",
        "torch>=2.3.0",
        "Pillow>=10.0.0",
        "transformers>=4.41.0",
    )
)


def _resolve_worker_secrets() -> List[modal.Secret]:
    secrets: List[modal.Secret] = []
    if MODAL_HF_SECRET_NAME:
        secrets.append(modal.Secret.from_name(MODAL_HF_SECRET_NAME))
        return secrets
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        secrets.append(
            modal.Secret.from_dict(
                {
                    "HF_TOKEN": str(hf_token),
                    "HUGGING_FACE_HUB_TOKEN": str(hf_token),
                }
            )
        )
    return secrets


EMBEDDING_WORKER_SECRETS = _resolve_worker_secrets()


def _chunk_items(items: List[Tuple[int, bytes]], size: int) -> List[List[Tuple[int, bytes]]]:
    size = max(1, int(size))
    chunks: List[List[Tuple[int, bytes]]] = []
    for i in range(0, len(items), size):
        chunks.append(items[i : i + size])
    return chunks


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


@app.function(
    image=embedding_image,
    timeout=1800,
    cpu=2.0,
    memory=4096,
    secrets=EMBEDDING_WORKER_SECRETS,
)
def embed_capture_batch(job_payload: dict) -> dict:
    import io

    from PIL import Image

    capture_ids_raw = list(job_payload.get("capture_ids") or [])
    image_bytes_raw = list(job_payload.get("image_bytes") or [])
    models_raw = list(job_payload.get("models") or [])

    if not capture_ids_raw or not image_bytes_raw or not models_raw:
        return {
            "capture_ids": [],
            "model_outputs": [],
            "skipped": [],
            "model_errors": [],
        }

    try:
        import open_clip
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "Modal embedding dependencies missing (open-clip-torch, transformers, torch)."
        ) from exc

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    clip_cache: Dict[Tuple[str, str], Tuple[object, object]] = {}
    hf_cache: Dict[Tuple[str, bool], Tuple[object, object]] = {}
    decoded_images: List[Tuple[int, object]] = []
    skipped: List[dict] = []
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    for capture_id, image_bytes in zip(capture_ids_raw, image_bytes_raw):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            decoded_images.append((int(capture_id), image))
        except Exception as exc:
            skipped.append(
                {
                    "capture_id": int(capture_id),
                    "reason": f"image-decode-failed:{exc}",
                }
            )

    if not decoded_images:
        return {
            "capture_ids": [],
            "model_outputs": [],
            "skipped": skipped,
            "model_errors": [],
        }

    capture_ids = [capture_id for capture_id, _ in decoded_images]
    model_outputs: List[dict] = []
    model_errors: List[dict] = []

    for model in models_raw:
        model_name = str(model.get("model_name") or "").strip()
        pretrained = str(model.get("pretrained") or "").strip()
        model_version = str(model.get("model_version") or "").strip()
        model_id = str(model.get("model_id") or "").strip()
        runtime = str(model.get("runtime") or "open_clip").strip().lower()
        trust_remote_code = bool(model.get("trust_remote_code", False))
        if not model_name:
            model_errors.append(
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "model_id": model_id,
                    "error": "invalid-model-config",
                }
            )
            continue
        try:
            if runtime == "hf_transformers":
                cache_key = (model_name, trust_remote_code)
                if cache_key not in hf_cache:
                    processor = AutoImageProcessor.from_pretrained(
                        model_name,
                        trust_remote_code=trust_remote_code,
                        token=hf_token,
                    )
                    hf_model = AutoModel.from_pretrained(
                        model_name,
                        trust_remote_code=trust_remote_code,
                        token=hf_token,
                    )
                    hf_model.eval()
                    hf_model.to(device)
                    hf_cache[cache_key] = (processor, hf_model)
                processor, hf_model = hf_cache[cache_key]
                images = [image for _, image in decoded_images]
                inputs = processor(images=images, return_tensors="pt")
                for key, value in list(inputs.items()):
                    if hasattr(value, "to"):
                        inputs[key] = value.to(device)
                with torch.no_grad():
                    outputs = hf_model(**inputs)
                    if getattr(outputs, "pooler_output", None) is not None:
                        features = outputs.pooler_output
                    elif getattr(outputs, "last_hidden_state", None) is not None:
                        features = outputs.last_hidden_state.mean(dim=1)
                    else:
                        tensor_out = outputs[0]
                        features = (
                            tensor_out.mean(dim=1)
                            if getattr(tensor_out, "dim", lambda: 0)() == 3
                            else tensor_out
                        )
                    features = features / features.norm(dim=-1, keepdim=True).clamp(
                        min=1e-12
                    )
            else:
                if not pretrained:
                    raise RuntimeError("open_clip models require pretrained")
                cache_key = (model_name, pretrained)
                if cache_key not in clip_cache:
                    clip_model, _, preprocess = open_clip.create_model_and_transforms(
                        model_name, pretrained=pretrained
                    )
                    clip_model.eval()
                    clip_model.to(device)
                    clip_cache[cache_key] = (clip_model, preprocess)
                clip_model, preprocess = clip_cache[cache_key]

                tensors = [preprocess(image) for _, image in decoded_images]
                batch = torch.stack(tensors, dim=0).to(device)
                with torch.no_grad():
                    features = clip_model.encode_image(batch)
                    features = features / features.norm(dim=-1, keepdim=True).clamp(
                        min=1e-12
                    )
            vectors = features.detach().cpu().float().tolist()
            model_outputs.append(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "model_version": model_version,
                    "vectors": vectors,
                }
            )
        except Exception as exc:
            model_errors.append(
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "model_id": model_id,
                    "error": str(exc),
                }
            )

    return {
        "capture_ids": capture_ids,
        "model_outputs": model_outputs,
        "skipped": skipped,
        "model_errors": model_errors,
    }


def dispatch_embedding_jobs(
    capture_items: Sequence[Tuple[int, bytes]],
    model_configs: Sequence[dict],
    num_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    modal_environment: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    progress_callback: Optional[Callable[[dict], None]] = None,
    result_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    items = [(int(capture_id), image_bytes) for capture_id, image_bytes in capture_items]
    if not items:
        return {
            "jobs_total": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "jobs_retried": 0,
            "captures_total": 0,
            "captures_completed": 0,
            "captures_failed": 0,
            "captures_skipped_decode": 0,
            "model_error_events": 0,
            "failed_capture_ids": [],
        }

    jobs = deque(
        {
            "job_id": f"embed-{idx}",
            "attempt": 0,
            "items": chunk,
        }
        for idx, chunk in enumerate(_chunk_items(items, batch_size))
    )
    jobs_total = len(jobs)
    jobs_completed = 0
    jobs_failed = 0
    jobs_retried = 0
    captures_completed = 0
    captures_failed = 0
    captures_skipped_decode = 0
    model_error_events = 0
    failed_capture_ids: List[int] = []

    workers = max(1, min(512, int(num_workers)))
    max_retries = max(0, int(max_retries))
    environment_name = (
        modal_environment
        or os.getenv("MODAL_ENVIRONMENT")
        or DEFAULT_MODAL_ENVIRONMENT
    )

    def emit(event: dict):
        if progress_callback:
            try:
                progress_callback(event)
            except Exception:
                pass

    emit(
        {
            "event": "dispatch_started",
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
                    "models": list(model_configs),
                }
                handle = embed_capture_batch.spawn(payload)
                active.append((job, handle))
                emit(
                    {
                        "event": "job_submitted",
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

                    skipped = list(payload.get("skipped") or [])
                    model_errors = list(payload.get("model_errors") or [])
                    captures_skipped_decode += len(skipped)
                    model_error_events += len(model_errors)
                    capture_ids = [int(v) for v in list(payload.get("capture_ids") or [])]
                    captures_completed += len(capture_ids)

                    emit(
                        {
                            "event": "job_completed",
                            "job_id": job["job_id"],
                            "attempt": job["attempt"],
                            "captures_completed": len(capture_ids),
                            "captures_skipped_decode": len(skipped),
                            "model_error_events": len(model_errors),
                        }
                    )
                    if result_callback:
                        result_callback(payload)
                except Exception as exc:
                    if job["attempt"] < max_retries:
                        job["attempt"] += 1
                        jobs_retried += 1
                        jobs.append(job)
                        emit(
                            {
                                "event": "job_retry_enqueued",
                                "job_id": job["job_id"],
                                "attempt": job["attempt"],
                                "error": str(exc),
                            }
                        )
                    else:
                        jobs_failed += 1
                        failed_ids = [capture_id for capture_id, _ in job["items"]]
                        failed_capture_ids.extend(failed_ids)
                        captures_failed += len(failed_ids)
                        emit(
                            {
                                "event": "job_failed",
                                "job_id": job["job_id"],
                                "attempt": job["attempt"],
                                "capture_count": len(job["items"]),
                                "error": str(exc),
                            }
                        )

    result = {
        "jobs_total": jobs_total,
        "jobs_completed": jobs_completed,
        "jobs_failed": jobs_failed,
        "jobs_retried": jobs_retried,
        "captures_total": len(items),
        "captures_completed": captures_completed,
        "captures_failed": captures_failed,
        "captures_skipped_decode": captures_skipped_decode,
        "model_error_events": model_error_events,
        "failed_capture_ids": failed_capture_ids,
    }
    emit({"event": "all_done", **result})
    return result
