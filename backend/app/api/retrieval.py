import os
import time
from typing import Callable, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.app.clip_embeddings import (
    encode_image_batch_for_all_models,
    get_retrieval_embedders,
)
from backend.app.services.retrieval_locator import (
    LOCATOR_DB_MAX_TOP_K,
    LOCATOR_IVFFLAT_PROBES,
    locate_image_bytes,
    log_retrieval_event,
    new_retrieval_id,
)

MAX_RETRIEVAL_UPLOAD_BYTES = (
    int(os.getenv("GEOSPY_MAX_RETRIEVAL_UPLOAD_MB", "15")) * 1024 * 1024
)
RETRIEVAL_INCLUDE_DEBUG_DEFAULT = (
    os.getenv("GEOSPY_RETRIEVAL_INCLUDE_DEBUG_DEFAULT", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
LOCATOR_TOP_K_PER_CROP = max(
    10, int(os.getenv("GEOSPY_LOCATOR_TOP_K_PER_CROP", "220"))
)
LOCATOR_MAX_MERGED_CANDIDATES = max(
    50, int(os.getenv("GEOSPY_LOCATOR_MAX_MERGED_CANDIDATES", "5000"))
)
LOCATOR_PANORAMA_VOTE_CAP = max(
    1, int(os.getenv("GEOSPY_LOCATOR_PANORAMA_VOTE_CAP", "3"))
)
LOCATOR_CLUSTER_RADIUS_M = max(
    5.0, float(os.getenv("GEOSPY_LOCATOR_CLUSTER_RADIUS_M", "45"))
)
LOCATOR_VERIFY_TOP_N = max(5, int(os.getenv("GEOSPY_LOCATOR_VERIFY_TOP_N", "120")))
LOCATOR_MIN_GOOD_MATCHES = max(
    4, int(os.getenv("GEOSPY_LOCATOR_MIN_GOOD_MATCHES", "11"))
)
LOCATOR_MIN_INLIER_RATIO = max(
    0.01, float(os.getenv("GEOSPY_LOCATOR_MIN_INLIER_RATIO", "0.16"))
)
LOCATOR_APPEARANCE_PENALTY_WEIGHT = max(
    0.0, float(os.getenv("GEOSPY_LOCATOR_APPEARANCE_PENALTY_WEIGHT", "0.22"))
)
LOCATOR_DB_MAX_TOP_K_DEFAULT = max(200, int(LOCATOR_DB_MAX_TOP_K))
LOCATOR_IVFFLAT_PROBES_DEFAULT = max(1, int(LOCATOR_IVFFLAT_PROBES))
LOCATE_TUNING_DEFAULTS = {
    "top_k_per_crop": LOCATOR_TOP_K_PER_CROP,
    "max_candidates": LOCATOR_MAX_MERGED_CANDIDATES,
    "panorama_vote_cap": LOCATOR_PANORAMA_VOTE_CAP,
    "cluster_radius_m": LOCATOR_CLUSTER_RADIUS_M,
    "verify_top_n": LOCATOR_VERIFY_TOP_N,
    "min_good_matches": LOCATOR_MIN_GOOD_MATCHES,
    "min_inlier_ratio": LOCATOR_MIN_INLIER_RATIO,
    "appearance_penalty_weight": LOCATOR_APPEARANCE_PENALTY_WEIGHT,
    "db_max_top_k": LOCATOR_DB_MAX_TOP_K_DEFAULT,
    "ivfflat_probes": LOCATOR_IVFFLAT_PROBES_DEFAULT,
}
LOCATE_TUNING_BOUNDS = {
    "top_k_per_crop": {"min": 5, "max": 1200, "type": "int"},
    "max_candidates": {"min": 30, "max": 10000, "type": "int"},
    "panorama_vote_cap": {"min": 1, "max": 8, "type": "int"},
    "cluster_radius_m": {"min": 5.0, "max": 250.0, "type": "float"},
    "verify_top_n": {"min": 5, "max": 400, "type": "int"},
    "min_good_matches": {"min": 4, "max": 80, "type": "int"},
    "min_inlier_ratio": {"min": 0.01, "max": 0.95, "type": "float"},
    "appearance_penalty_weight": {"min": 0.0, "max": 0.95, "type": "float"},
    "db_max_top_k": {"min": 100, "max": 20000, "type": "int"},
    "ivfflat_probes": {"min": 1, "max": 1000, "type": "int"},
}
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


class RetrievalIndexRequest(BaseModel):
    limit: int = 100


def _parse_form_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_form_int(value, field_name: str) -> int:
    try:
        return int(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{field_name} must be an integer")


def _parse_form_float(value, field_name: str) -> float:
    try:
        return float(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a number")


def _clamp_int(value: int, *, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(value)))


def _clamp_float(value: float, *, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, float(value)))


def _normalize_locate_tuning(
    *,
    top_k_per_crop,
    max_candidates,
    panorama_vote_cap,
    cluster_radius_m,
    verify_top_n,
    min_good_matches,
    min_inlier_ratio,
    appearance_penalty_weight,
    db_max_top_k,
    ivfflat_probes,
) -> Dict[str, float]:
    return {
        "top_k_per_crop": _clamp_int(
            _parse_form_int(top_k_per_crop, "top_k_per_crop"), min_value=5, max_value=1200
        ),
        "max_candidates": _clamp_int(
            _parse_form_int(max_candidates, "max_candidates"),
            min_value=30,
            max_value=10000,
        ),
        "panorama_vote_cap": _clamp_int(
            _parse_form_int(panorama_vote_cap, "panorama_vote_cap"),
            min_value=1,
            max_value=8,
        ),
        "cluster_radius_m": _clamp_float(
            _parse_form_float(cluster_radius_m, "cluster_radius_m"),
            min_value=5.0,
            max_value=250.0,
        ),
        "verify_top_n": _clamp_int(
            _parse_form_int(verify_top_n, "verify_top_n"), min_value=5, max_value=400
        ),
        "min_good_matches": _clamp_int(
            _parse_form_int(min_good_matches, "min_good_matches"),
            min_value=4,
            max_value=80,
        ),
        "min_inlier_ratio": _clamp_float(
            _parse_form_float(min_inlier_ratio, "min_inlier_ratio"),
            min_value=0.01,
            max_value=0.95,
        ),
        "appearance_penalty_weight": _clamp_float(
            _parse_form_float(appearance_penalty_weight, "appearance_penalty_weight"),
            min_value=0.0,
            max_value=0.95,
        ),
        "db_max_top_k": _clamp_int(
            _parse_form_int(db_max_top_k, "db_max_top_k"),
            min_value=100,
            max_value=20000,
        ),
        "ivfflat_probes": _clamp_int(
            _parse_form_int(ivfflat_probes, "ivfflat_probes"), min_value=1, max_value=1000
        ),
    }


def create_retrieval_router(
    *,
    get_db: Callable[[], object],
    capture_web_path: Callable[[str], str],
    capture_abs_path: Callable[[str], str],
) -> APIRouter:
    router = APIRouter()

    def _select_query_embedders(db, embedders):
        if not embedders:
            return [], []
        if RETRIEVAL_MIN_MODEL_COVERAGE <= 0.0 or len(embedders) <= 1:
            return embedders, []
        selected = []
        skipped = []
        for idx, embedder in enumerate(embedders):
            stats = db.get_capture_embedding_stats(
                embedder.model_name, embedder.model_version
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
                        embedder.model_name, embedder.model_version
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

    @router.get("/api/retrieval/locate-params")
    async def retrieval_locate_params():
        return JSONResponse(
            {
                "defaults": LOCATE_TUNING_DEFAULTS,
                "bounds": LOCATE_TUNING_BOUNDS,
            }
        )

    @router.post("/api/retrieval/search-by-image")
    async def retrieval_search_by_image(
        image: UploadFile = File(...),
        top_k: int = Form(12),
        min_similarity: Optional[float] = Form(None),
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
        )

        try:
            embedders = list(get_retrieval_embedders())
            if not embedders:
                raise RuntimeError("No retrieval models configured")
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
                db, embedders
            )
            if coverage_skipped_models:
                log_retrieval_event(
                    retrieval_id,
                    "search_models_skipped_low_coverage",
                    skipped_models=coverage_skipped_models,
                )
            merged = {}
            failed_models = []
            for embedder in active_embedders:
                try:
                    vector = embedder.encode_image_bytes(image_bytes)
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
                row["web_path"] = capture_web_path(row.get("filepath", ""))
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            primary = active_embedders[0]
            log_retrieval_event(
                retrieval_id,
                "search_completed",
                matches=len(rows),
                elapsed_ms=elapsed_ms,
                model_count=len(active_embedders),
                failed_models=len(failed_models),
                model_name=primary.model_name,
                model_version=primary.model_version,
            )
            return JSONResponse(
                {
                    "retrieval_id": retrieval_id,
                    "model_name": primary.model_name,
                    "model_version": primary.model_version,
                    "models": [
                        {
                            "model_id": getattr(embedder, "model_id", embedder.model_name),
                            "model_name": embedder.model_name,
                            "model_version": embedder.model_version,
                            "weight": float(getattr(embedder, "weight", 1.0)),
                        }
                        for embedder in active_embedders
                    ],
                    "skipped_models": coverage_skipped_models,
                    "failed_models": failed_models,
                    "matches": rows,
                    "timings_ms": {"total": elapsed_ms},
                }
            )
        finally:
            db.close()

    @router.post("/api/retrieval/locate-by-image")
    async def retrieval_locate_by_image(
        image: UploadFile = File(...),
        top_k_per_crop: int = Form(LOCATOR_TOP_K_PER_CROP),
        max_candidates: int = Form(LOCATOR_MAX_MERGED_CANDIDATES),
        panorama_vote_cap: int = Form(LOCATOR_PANORAMA_VOTE_CAP),
        cluster_radius_m: float = Form(LOCATOR_CLUSTER_RADIUS_M),
        verify_top_n: int = Form(LOCATOR_VERIFY_TOP_N),
        min_good_matches: int = Form(LOCATOR_MIN_GOOD_MATCHES),
        min_inlier_ratio: float = Form(LOCATOR_MIN_INLIER_RATIO),
        appearance_penalty_weight: float = Form(LOCATOR_APPEARANCE_PENALTY_WEIGHT),
        db_max_top_k: int = Form(LOCATOR_DB_MAX_TOP_K_DEFAULT),
        ivfflat_probes: int = Form(LOCATOR_IVFFLAT_PROBES_DEFAULT),
        min_similarity: Optional[float] = Form(None),
        include_debug: Optional[str] = Form(None),
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
        locate_tuning = _normalize_locate_tuning(
            top_k_per_crop=top_k_per_crop,
            max_candidates=max_candidates,
            panorama_vote_cap=panorama_vote_cap,
            cluster_radius_m=cluster_radius_m,
            verify_top_n=verify_top_n,
            min_good_matches=min_good_matches,
            min_inlier_ratio=min_inlier_ratio,
            appearance_penalty_weight=appearance_penalty_weight,
            db_max_top_k=db_max_top_k,
            ivfflat_probes=ivfflat_probes,
        )
        include_debug_bool = _parse_form_bool(
            include_debug, default=RETRIEVAL_INCLUDE_DEBUG_DEFAULT
        )
        log_retrieval_event(
            retrieval_id,
            "locate_started",
            upload_bytes=len(image_bytes),
            **locate_tuning,
            min_similarity=min_similarity,
            include_debug=include_debug_bool,
        )

        db = get_db()
        try:
            embedders = list(get_retrieval_embedders())
            if not db.is_vector_ready():
                raise HTTPException(
                    status_code=503,
                    detail="Vector extension is unavailable. Use a pgvector-enabled Postgres image.",
                )
            active_embedders, coverage_skipped_models = _select_query_embedders(
                db, embedders
            )
            if coverage_skipped_models:
                log_retrieval_event(
                    retrieval_id,
                    "locate_models_skipped_low_coverage",
                    skipped_models=coverage_skipped_models,
                )
            result = locate_image_bytes(
                image_bytes=image_bytes,
                db=db,
                embedders=active_embedders,
                capture_abs_path=capture_abs_path,
                retrieval_id=retrieval_id,
                top_k_per_crop=locate_tuning["top_k_per_crop"],
                max_merged_candidates=locate_tuning["max_candidates"],
                panorama_vote_cap=locate_tuning["panorama_vote_cap"],
                cluster_radius_m=locate_tuning["cluster_radius_m"],
                verify_top_n=locate_tuning["verify_top_n"],
                min_good_matches=locate_tuning["min_good_matches"],
                min_inlier_ratio=locate_tuning["min_inlier_ratio"],
                appearance_penalty_weight=locate_tuning["appearance_penalty_weight"],
                db_max_top_k=locate_tuning["db_max_top_k"],
                ivfflat_probes=locate_tuning["ivfflat_probes"],
                min_similarity=min_similarity,
                model_weights={
                    str(getattr(embedder, "model_id", f"model_{idx}")): float(
                        getattr(embedder, "weight", 1.0)
                    )
                    for idx, embedder in enumerate(active_embedders)
                },
                include_debug=include_debug_bool,
            )
            result["skipped_models"] = coverage_skipped_models
        except RuntimeError as exc:
            log_retrieval_event(
                retrieval_id, "locate_embedder_unavailable", error=str(exc)
            )
            raise HTTPException(status_code=503, detail=str(exc))
        except HTTPException:
            raise
        except Exception as exc:
            log_retrieval_event(retrieval_id, "locate_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=f"Locator failed: {exc}")
        else:
            for row in result.get("supporting_matches", []):
                row["web_path"] = capture_web_path(row.get("filepath", ""))
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            result["timings_ms"] = {
                **(result.get("timings_ms") or {}),
                "api_total": elapsed_ms,
            }
            log_retrieval_event(
                retrieval_id,
                "locate_completed",
                elapsed_ms=elapsed_ms,
                flags=result.get("flags", []),
                has_estimate=bool(result.get("best_estimate")),
            )
            return JSONResponse(result)
        finally:
            db.close()

    @router.post("/api/retrieval/index-missing")
    async def retrieval_index_missing(req: RetrievalIndexRequest):
        retrieval_id = new_retrieval_id()
        started = time.perf_counter()
        limit = max(1, min(500, int(req.limit)))
        log_retrieval_event(retrieval_id, "index_missing_started", limit=limit)
        try:
            embedders = list(get_retrieval_embedders())
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
                        [payload for _, payload in valid_batch]
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
                embedders[0].model_name, embedders[0].model_version
            )
            stats["models"] = [
                db.get_capture_embedding_stats(embedder.model_name, embedder.model_version)
                for embedder in embedders
            ]
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return JSONResponse(
                {
                    "retrieval_id": retrieval_id,
                    "indexed": indexed,
                    "skipped": skipped,
                    "attempted": len(rows),
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
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
            db.close()

    return router
