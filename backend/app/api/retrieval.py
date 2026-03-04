import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.app.clip_embeddings import (
    EMBEDDING_BASE_CLIP,
    EMBEDDING_BASE_PIGEON,
    encode_image_batch_for_all_models,
    get_retrieval_embedders,
    select_retrieval_embedders,
)

MAX_RETRIEVAL_UPLOAD_BYTES = (
    int(os.getenv("GEOSPY_MAX_RETRIEVAL_UPLOAD_MB", "15")) * 1024 * 1024
)
RETRIEVAL_MIN_MODEL_COVERAGE = max(
    0.0, min(1.0, float(os.getenv("GEOSPY_RETRIEVAL_MIN_MODEL_COVERAGE", "0.0")))
)
RETRIEVAL_DILIGENT_MODE = (
    os.getenv("GEOSPY_RETRIEVAL_DILIGENT_MODE", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)
RETRIEVAL_SEARCH_CANDIDATE_MULTIPLIER = max(
    2,
    int(
        os.getenv(
            "GEOSPY_RETRIEVAL_SEARCH_CANDIDATE_MULTIPLIER",
            "10" if RETRIEVAL_DILIGENT_MODE else "3",
        )
    ),
)
RETRIEVAL_SEARCH_MAX_CANDIDATES = max(
    100,
    min(
        10000,
        int(
            os.getenv(
                "GEOSPY_RETRIEVAL_SEARCH_MAX_CANDIDATES",
                "5000" if RETRIEVAL_DILIGENT_MODE else "1000",
            )
        ),
    ),
)
RETRIEVAL_IVFFLAT_PROBES = max(
    1, int(os.getenv("GEOSPY_RETRIEVAL_IVFFLAT_PROBES", "120"))
)
RETRIEVAL_EMBEDDING_BASE_DEFAULT = str(
    os.getenv("GEOSPY_RETRIEVAL_EMBEDDING_BASE_DEFAULT", EMBEDDING_BASE_CLIP)
).strip().lower()
log = logging.getLogger(__name__)


def new_retrieval_id() -> str:
    return uuid.uuid4().hex[:12]


def log_retrieval_event(retrieval_id: str, event: str, **fields: Any) -> None:
    payload = {"retrieval_id": retrieval_id, "event": event, **fields}
    try:
        log.info("retrieval %s", json.dumps(payload, default=str, sort_keys=True))
    except Exception:
        log.info("retrieval id=%s event=%s", retrieval_id, event)


class RetrievalIndexRequest(BaseModel):
    limit: int = 100
    embedding_base: str = RETRIEVAL_EMBEDDING_BASE_DEFAULT


def _normalize_embedding_base(value: Optional[str]) -> str:
    base = str(value or RETRIEVAL_EMBEDDING_BASE_DEFAULT).strip().lower()
    if base in {EMBEDDING_BASE_CLIP, EMBEDDING_BASE_PIGEON}:
        return base
    raise HTTPException(
        status_code=400,
        detail=f"embedding_base must be one of: {EMBEDDING_BASE_CLIP}, {EMBEDDING_BASE_PIGEON}",
    )


def create_retrieval_router(
    *,
    get_db: Callable[[], object],
    capture_web_path: Callable[[str], str],
    capture_abs_path: Callable[[str], str],
) -> APIRouter:
    router = APIRouter()

    def _embedders_for_base_or_400(embedding_base: str):
        try:
            embedders = list(
                select_retrieval_embedders(embedding_base, allow_fallback=False)
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if embedders:
            return embedders
        raise HTTPException(
            status_code=400,
            detail=(
                f"No retrieval embedders configured for embedding_base={embedding_base}. "
                "Enable/configure the selected base in env (e.g. GEOSPY_PIGEON_MODEL_ENABLED=1)."
            ),
        )

    def _select_query_embedders(db, embedders, embedding_base: str):
        if not embedders:
            return [], []
        if RETRIEVAL_MIN_MODEL_COVERAGE <= 0.0 or len(embedders) <= 1:
            return embedders, []
        selected = []
        skipped = []
        for idx, embedder in enumerate(embedders):
            stats = db.get_capture_embedding_stats(
                embedder.model_name,
                embedder.model_version,
                embedding_base=embedding_base,
            )
            total = max(0, int(stats.get("total_captures") or 0))
            embedded = max(0, int(stats.get("embedded_captures") or 0))
            coverage = (float(embedded) / float(total)) if total > 0 else 1.0
            if idx == 0 or coverage >= RETRIEVAL_MIN_MODEL_COVERAGE:
                selected.append(embedder)
                continue
            skipped.append(
                {
                    "model_id": getattr(embedder, "model_id", embedder.model_name),
                    "model_name": embedder.model_name,
                    "model_version": embedder.model_version,
                    "coverage": round(coverage, 4),
                    "embedded_captures": embedded,
                    "total_captures": total,
                    "min_required_coverage": RETRIEVAL_MIN_MODEL_COVERAGE,
                    "reason": "coverage-below-threshold",
                }
            )
        if not selected:
            selected = [embedders[0]]
        return selected, skipped

    @router.get("/api/retrieval/index-stats")
    async def retrieval_index_stats():
        retrieval_id = new_retrieval_id()
        log_retrieval_event(retrieval_id, "index_stats_started")
        db = get_db()
        try:
            embedders = list(get_retrieval_embedders())
            model_stats = []
            for embedder in embedders:
                model_stats.append(
                    db.get_capture_embedding_stats(
                        embedder.model_name,
                        embedder.model_version,
                        embedding_base=str(
                            getattr(embedder, "embedding_base", EMBEDDING_BASE_CLIP)
                        ),
                    )
                )
            primary = model_stats[0] if model_stats else {
                "vector_enabled": db.is_vector_ready(),
                "model_name": "",
                "model_version": "",
                "total_captures": 0,
                "embedded_captures": 0,
                "pending_captures": 0,
            }
            stats = {
                **primary,
                "models": model_stats,
            }
            log_retrieval_event(
                retrieval_id,
                "index_stats_completed",
                total_captures=stats.get("total_captures", 0),
                embedded_captures=stats.get("embedded_captures", 0),
                pending_captures=stats.get("pending_captures", 0),
                model_count=len(model_stats),
            )
            return JSONResponse(stats)
        finally:
            db.close()

    @router.post("/api/retrieval/search-by-image")
    async def retrieval_search_by_image(
        image: UploadFile = File(...),
        top_k: int = Form(12),
        min_similarity: Optional[float] = Form(None),
        embedding_base: str = Form(RETRIEVAL_EMBEDDING_BASE_DEFAULT),
    ):
        retrieval_id = new_retrieval_id()
        started = time.perf_counter()
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(image_bytes) > MAX_RETRIEVAL_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")
        if min_similarity is not None and (min_similarity < 0.0 or min_similarity > 1.0):
            raise HTTPException(
                status_code=400, detail="min_similarity must be between 0 and 1"
            )
        embedding_base = _normalize_embedding_base(embedding_base)
        log_retrieval_event(
            retrieval_id,
            "search_started",
            upload_bytes=len(image_bytes),
            top_k=top_k,
            min_similarity=min_similarity,
            diligent_mode=RETRIEVAL_DILIGENT_MODE,
            candidate_multiplier=RETRIEVAL_SEARCH_CANDIDATE_MULTIPLIER,
            max_candidates=RETRIEVAL_SEARCH_MAX_CANDIDATES,
            ivfflat_probes=RETRIEVAL_IVFFLAT_PROBES,
            embedding_base=embedding_base,
        )

        try:
            embedders = _embedders_for_base_or_400(embedding_base)
        except HTTPException:
            raise
        except RuntimeError as exc:
            log_retrieval_event(
                retrieval_id, "search_embedder_unavailable", error=str(exc)
            )
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            log_retrieval_event(retrieval_id, "search_embedding_failed", error=str(exc))
            raise HTTPException(status_code=400, detail=f"Image embedding failed: {exc}")

        db = get_db()
        try:
            if not db.is_vector_ready():
                raise HTTPException(
                    status_code=503,
                    detail="Vector extension is unavailable. Use a pgvector-enabled Postgres image.",
                )
            active_embedders, coverage_skipped_models = _select_query_embedders(
                db, embedders, embedding_base
            )
            if coverage_skipped_models:
                log_retrieval_event(
                    retrieval_id,
                    "search_models_skipped_low_coverage",
                    skipped_models=coverage_skipped_models,
                )
            merged = {}
            failed_models = []
            search_timings_ms: Dict[str, Any] = {}
            model_timings_ms: Dict[str, Dict[str, float]] = {}
            for embedder in active_embedders:
                model_timer_start = time.perf_counter()
                model_key = f"{embedder.model_name}:{embedder.model_version}"
                try:
                    t_encode = time.perf_counter()
                    vector = embedder.encode_image_bytes(image_bytes)
                    encode_elapsed_ms = round((time.perf_counter() - t_encode) * 1000.0, 2)
                except Exception as exc:
                    failed_models.append(
                        {
                            "model_id": getattr(embedder, "model_id", embedder.model_name),
                            "model_name": embedder.model_name,
                            "model_version": embedder.model_version,
                            "error": str(exc),
                        }
                    )
                    continue
                t_query = time.perf_counter()
                rows = db.search_captures_by_embedding(
                    vector,
                    embedder.model_name,
                    embedder.model_version,
                    top_k=max(
                        1,
                        min(
                            RETRIEVAL_SEARCH_MAX_CANDIDATES,
                            int(top_k) * RETRIEVAL_SEARCH_CANDIDATE_MULTIPLIER,
                        ),
                    ),
                    min_similarity=min_similarity,
                    trace_id=retrieval_id,
                    ivfflat_probes=RETRIEVAL_IVFFLAT_PROBES,
                    embedding_base=embedding_base,
                )
                query_elapsed_ms = round((time.perf_counter() - t_query) * 1000.0, 2)
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
                            "model_scores": {model_id: similarity},
                        }
                    else:
                        entry["score"] = float(entry.get("score", 0.0)) + (
                            similarity * model_weight
                        )
                        if model_id not in entry["model_hits"]:
                            entry["model_hits"].append(model_id)
                        entry.setdefault("model_scores", {})[model_id] = similarity
                        if similarity > float(entry.get("similarity", 0.0)):
                            entry["similarity"] = similarity
                model_timings_ms[model_key] = {
                    "encode": encode_elapsed_ms,
                    "search": query_elapsed_ms,
                    "total": round((time.perf_counter() - model_timer_start) * 1000.0, 2),
                }
            rows = sorted(
                merged.values(),
                key=lambda r: float(r.get("score", 0.0)),
                reverse=True,
            )[: max(1, min(200, int(top_k)))]
            if not rows and failed_models:
                log_retrieval_event(
                    retrieval_id,
                    "search_all_models_failed",
                    failed_models=failed_models,
                )
                raise HTTPException(
                    status_code=503,
                    detail="All retrieval models failed to encode. Check model/pretrained settings.",
                )
            for row in rows:
                raw_path = row.get("filepath", "")
                abs_path = capture_abs_path(raw_path)
                row["web_path"] = (
                    capture_web_path(raw_path)
                    if abs_path and os.path.exists(abs_path)
                    else ""
                )
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            search_timings_ms["model_compute_total"] = round(
                sum(float(v.get("total", 0.0)) for v in model_timings_ms.values()), 2
            )
            search_timings_ms["model_timings"] = model_timings_ms
            search_timings_ms["total"] = elapsed_ms
            primary = active_embedders[0]
            log_retrieval_event(
                retrieval_id,
                "search_completed",
                matches=len(rows),
                elapsed_ms=elapsed_ms,
                embedding_base=embedding_base,
                model_count=len(active_embedders),
                failed_models=len(failed_models),
                model_name=primary.model_name,
                model_version=primary.model_version,
                timings_ms=search_timings_ms,
            )
            return JSONResponse(
                {
                    "retrieval_id": retrieval_id,
                    "embedding_base": embedding_base,
                    "model_name": primary.model_name,
                    "model_version": primary.model_version,
                    "models": [
                        {
                            "model_id": getattr(embedder, "model_id", embedder.model_name),
                            "model_name": embedder.model_name,
                            "model_version": embedder.model_version,
                            "embedding_base": str(
                                getattr(embedder, "embedding_base", embedding_base)
                            ),
                            "weight": float(getattr(embedder, "weight", 1.0)),
                        }
                        for embedder in active_embedders
                    ],
                    "skipped_models": coverage_skipped_models,
                    "failed_models": failed_models,
                    "matches": rows,
                    "timings_ms": search_timings_ms,
                }
            )
        finally:
            db.close()

    @router.post("/api/retrieval/index-missing")
    async def retrieval_index_missing(req: RetrievalIndexRequest):
        retrieval_id = new_retrieval_id()
        started = time.perf_counter()
        limit = max(1, min(500, int(req.limit)))
        embedding_base = _normalize_embedding_base(getattr(req, "embedding_base", None))
        log_retrieval_event(
            retrieval_id,
            "index_missing_started",
            limit=limit,
            embedding_base=embedding_base,
        )
        try:
            embedders = _embedders_for_base_or_400(embedding_base)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        db = get_db()
        indexed = 0
        skipped = 0
        failed = []
        model_indexed = {}
        try:
            if not db.is_vector_ready():
                raise HTTPException(
                    status_code=503,
                    detail="Vector extension is unavailable. Use a pgvector-enabled Postgres image.",
                )
            if not embedders:
                raise HTTPException(status_code=503, detail="No retrieval models configured")
            rows = db.list_captures_missing_any_embeddings(
                [
                    (embedder.model_name, embedder.model_version)
                    for embedder in embedders
                ],
                limit=limit,
                embedding_base=embedding_base,
            )
            valid_batch = []
            for row in rows:
                cap_path = capture_abs_path(row.get("filepath", ""))
                if not cap_path or not os.path.exists(cap_path):
                    skipped += 1
                    failed.append(
                        {"capture_id": row["capture_id"], "error": "capture file missing"}
                    )
                    continue
                try:
                    with open(cap_path, "rb") as f:
                        valid_batch.append((int(row["capture_id"]), f.read()))
                except Exception as exc:
                    skipped += 1
                    failed.append({"capture_id": row["capture_id"], "error": str(exc)})
            if valid_batch:
                try:
                    model_batches = encode_image_batch_for_all_models(
                        [payload for _, payload in valid_batch],
                        embedders=embedders,
                    )
                    if not model_batches:
                        raise RuntimeError("No retrieval models encoded successfully")
                    for embedder, vectors in model_batches:
                        if len(vectors) != len(valid_batch):
                            raise RuntimeError(
                                f"batch embedding size mismatch for {embedder.model_id} expected={len(valid_batch)} got={len(vectors)}"
                            )
                        upserted = db.upsert_capture_embeddings_batch(
                            [
                                (capture_id, vector)
                                for (capture_id, _), vector in zip(valid_batch, vectors)
                            ],
                            embedder.model_name,
                            embedder.model_version,
                            embedding_base=embedding_base,
                        )
                        model_indexed[f"{embedder.model_name}:{embedder.model_version}"] = (
                            model_indexed.get(
                                f"{embedder.model_name}:{embedder.model_version}", 0
                            )
                            + int(upserted)
                        )
                    indexed += len(valid_batch)
                except Exception as exc:
                    log_retrieval_event(
                        retrieval_id,
                        "index_missing_batch_failed",
                        error=str(exc),
                        batch_size=len(valid_batch),
                    )
                    for capture_id, img in valid_batch:
                        try:
                            for embedder in embedders:
                                vector = embedder.encode_image_bytes(img)
                                db.upsert_capture_embedding(
                                    capture_id,
                                    embedder.model_name,
                                    embedder.model_version,
                                    vector,
                                    embedding_base=embedding_base,
                                )
                                model_indexed[
                                    f"{embedder.model_name}:{embedder.model_version}"
                                ] = model_indexed.get(
                                    f"{embedder.model_name}:{embedder.model_version}", 0
                                ) + 1
                            indexed += 1
                        except Exception as single_exc:
                            skipped += 1
                            failed.append(
                                {"capture_id": capture_id, "error": str(single_exc)}
                            )
            stats = db.get_capture_embedding_stats(
                embedders[0].model_name,
                embedders[0].model_version,
                embedding_base=embedding_base,
            )
            stats["models"] = [
                db.get_capture_embedding_stats(
                    embedder.model_name,
                    embedder.model_version,
                    embedding_base=embedding_base,
                )
                for embedder in embedders
            ]
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return JSONResponse(
                {
                    "retrieval_id": retrieval_id,
                    "indexed": indexed,
                    "skipped": skipped,
                    "attempted": len(rows),
                    "embedding_base": embedding_base,
                    "indexed_by_model": model_indexed,
                    "stats": stats,
                    "failures": failed[:25],
                    "timings_ms": {"total": elapsed_ms},
                }
            )
        finally:
            log_retrieval_event(
                retrieval_id,
                "index_missing_completed",
                indexed=indexed,
                skipped=skipped,
                attempted=indexed + skipped,
                embedding_base=embedding_base,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
            db.close()

    return router
