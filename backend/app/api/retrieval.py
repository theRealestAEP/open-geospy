import json
import logging
import math
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.app.clip_embeddings import (
    EMBEDDING_BASE_CLIP,
    EMBEDDING_BASE_PLACE,
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
LOCATE_SEARCH_CANDIDATE_MULTIPLIER = max(
    4,
    int(
        os.getenv(
            "GEOSPY_LOCATE_SEARCH_CANDIDATE_MULTIPLIER",
            str(max(8, RETRIEVAL_SEARCH_CANDIDATE_MULTIPLIER * 2)),
        )
    ),
)
LOCATE_SEARCH_MAX_CANDIDATES = max(
    200,
    min(
        10000,
        int(
            os.getenv(
                "GEOSPY_LOCATE_SEARCH_MAX_CANDIDATES",
                str(max(1500, RETRIEVAL_SEARCH_MAX_CANDIDATES)),
            )
        ),
    ),
)
LOCATE_PANORAMA_CANDIDATE_LIMIT = max(
    20, int(os.getenv("GEOSPY_LOCATE_PANORAMA_CANDIDATE_LIMIT", "160"))
)
LOCATE_FAMILY_RADIUS_METERS = max(
    5.0, float(os.getenv("GEOSPY_LOCATE_FAMILY_RADIUS_METERS", "35"))
)
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
    if base in {EMBEDDING_BASE_CLIP, EMBEDDING_BASE_PLACE}:
        return base
    raise HTTPException(
        status_code=400,
        detail=(
            f"embedding_base must be one of: "
            f"{EMBEDDING_BASE_CLIP}, {EMBEDDING_BASE_PLACE}"
        ),
    )


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    )
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def create_retrieval_router(
    *,
    get_db: Callable[[], object],
    capture_web_path: Callable[[str], str],
    capture_abs_path: Callable[[str], str],
    get_vector_store: Optional[Callable[[object], object]] = None,
) -> APIRouter:
    router = APIRouter()

    def _resolve_vector_store(db):
        if get_vector_store:
            return get_vector_store(db)
        return db

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
                "Enable/configure the selected base in env."
            ),
        )

    def _select_query_embedders(vector_store, embedders):
        if not embedders:
            return [], []
        if RETRIEVAL_MIN_MODEL_COVERAGE <= 0.0 or len(embedders) <= 1:
            return embedders, []
        selected = []
        skipped = []
        for idx, embedder in enumerate(embedders):
            embedding_base = str(
                getattr(embedder, "embedding_base", EMBEDDING_BASE_CLIP)
            ).strip().lower()
            stats = vector_store.get_capture_embedding_stats(
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
                    "embedding_base": embedding_base,
                    "coverage": round(coverage, 4),
                    "embedded_captures": embedded,
                    "total_captures": total,
                    "min_required_coverage": RETRIEVAL_MIN_MODEL_COVERAGE,
                    "reason": "coverage-below-threshold",
                }
            )
        if not selected and embedders:
            selected = [embedders[0]]
        return selected, skipped

    def _all_active_embedders_or_400(vector_store):
        embedders = list(get_retrieval_embedders())
        if not embedders:
            raise HTTPException(status_code=400, detail="No retrieval embedders configured.")
        return _select_query_embedders(vector_store, embedders)

    def _search_candidates(
        *,
        image_bytes: bytes,
        embedders: Sequence[object],
        vector_store,
        retrieval_id: str,
        min_similarity: Optional[float],
        top_k: int,
        candidate_multiplier: int,
        max_candidates: int,
    ) -> Tuple[List[dict], List[dict], Dict[str, Dict[str, float]]]:
        merged: Dict[int, dict] = {}
        failed_models: List[dict] = []
        model_timings_ms: Dict[str, Dict[str, float]] = {}
        for embedder in embedders:
            model_timer_start = time.perf_counter()
            model_key = f"{embedder.model_name}:{embedder.model_version}"
            embedding_base = str(
                getattr(embedder, "embedding_base", EMBEDDING_BASE_CLIP)
            ).strip().lower()
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
                        "embedding_base": embedding_base,
                        "error": str(exc),
                    }
                )
                continue
            t_query = time.perf_counter()
            rows = vector_store.search_captures_by_embedding(
                vector,
                embedder.model_name,
                embedder.model_version,
                top_k=max(1, min(max_candidates, int(top_k) * candidate_multiplier)),
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
                weighted_score = similarity * model_weight
                entry = merged.get(capture_id)
                if not entry:
                    merged[capture_id] = {
                        **row,
                        "score": weighted_score,
                        "model_hits": [model_id],
                        "model_scores": {model_id: similarity},
                        "embedding_bases": [embedding_base],
                    }
                else:
                    entry["score"] = float(entry.get("score", 0.0)) + weighted_score
                    if model_id not in entry["model_hits"]:
                        entry["model_hits"].append(model_id)
                    entry.setdefault("model_scores", {})[model_id] = similarity
                    if embedding_base not in entry.setdefault("embedding_bases", []):
                        entry["embedding_bases"].append(embedding_base)
                    if similarity > float(entry.get("similarity", 0.0)):
                        entry["similarity"] = similarity
                merged[capture_id]["embedding_base"] = embedding_base
            model_timings_ms[model_key] = {
                "encode": encode_elapsed_ms,
                "search": query_elapsed_ms,
                "total": round((time.perf_counter() - model_timer_start) * 1000.0, 2),
            }
        rows = sorted(
            merged.values(),
            key=lambda r: float(r.get("score", 0.0)),
            reverse=True,
        )
        return rows, failed_models, model_timings_ms

    def _attach_web_paths(rows: Sequence[dict]) -> None:
        for row in rows:
            raw_path = row.get("filepath", "")
            abs_path = capture_abs_path(raw_path)
            row["web_path"] = (
                capture_web_path(raw_path)
                if abs_path and os.path.exists(abs_path)
                else ""
            )

    def _aggregate_panorama_candidates(
        rows: Sequence[dict], model_count: int
    ) -> List[dict]:
        panoramas: Dict[int, dict] = {}
        for row in rows:
            panorama_id = int(row.get("panorama_id") or 0)
            if panorama_id <= 0:
                continue
            entry = panoramas.get(panorama_id)
            if not entry:
                entry = {
                    "panorama_id": panorama_id,
                    "pano_id": row.get("pano_id", ""),
                    "lat": float(row.get("lat", 0.0)),
                    "lon": float(row.get("lon", 0.0)),
                    "capture_rows": [],
                    "capture_scores": [],
                    "model_hits": set(),
                    "heading_bins": set(),
                    "pitch_bins": set(),
                    "best_capture": row,
                    "best_similarity": float(row.get("similarity", 0.0)),
                }
                panoramas[panorama_id] = entry
            entry["capture_rows"].append(row)
            entry["capture_scores"].append(float(row.get("score", 0.0)))
            entry["model_hits"].update(str(item) for item in list(row.get("model_hits") or []))
            entry["heading_bins"].add(round(float(row.get("heading", 0.0)) / 15.0) * 15)
            entry["pitch_bins"].add(round(float(row.get("pitch", 75.0))))
            similarity = float(row.get("similarity", 0.0))
            if similarity >= float(entry.get("best_similarity", 0.0)):
                entry["best_similarity"] = similarity
            if float(row.get("score", 0.0)) >= float(
                entry["best_capture"].get("score", 0.0)
            ):
                entry["best_capture"] = row

        ranked: List[dict] = []
        for entry in panoramas.values():
            top_capture_scores = sorted(entry["capture_scores"], reverse=True)[:5]
            best_capture = dict(entry["best_capture"])
            capture_count = len(entry["capture_rows"])
            heading_support = len(entry["heading_bins"])
            pitch_support = len(entry["pitch_bins"])
            model_support = len(entry["model_hits"])
            panorama_score = sum(top_capture_scores)
            panorama_score += 0.12 * float(entry["best_similarity"])
            panorama_score += 0.035 * float(min(heading_support, 6))
            panorama_score += 0.025 * float(min(pitch_support, 4))
            if model_count > 0:
                panorama_score += 0.06 * (float(model_support) / float(model_count))
            ranked.append(
                {
                    **best_capture,
                    "panorama_score": round(panorama_score, 6),
                    "capture_hits": capture_count,
                    "heading_support": heading_support,
                    "pitch_support": pitch_support,
                    "model_support": model_support,
                    "top_capture_scores": [round(float(v), 6) for v in top_capture_scores],
                }
            )
        ranked.sort(key=lambda r: float(r.get("panorama_score", 0.0)), reverse=True)
        return ranked

    def _cluster_panorama_families(
        panoramas: Sequence[dict], family_radius_meters: float
    ) -> List[dict]:
        families: List[dict] = []
        for panorama in panoramas:
            lat = float(panorama.get("lat", 0.0))
            lon = float(panorama.get("lon", 0.0))
            pano_score = float(panorama.get("panorama_score", 0.0))
            target_family = None
            best_distance = None
            for family in families:
                family_lat = float(family["center_lat"])
                family_lon = float(family["center_lon"])
                distance_m = _haversine_m(lat, lon, family_lat, family_lon)
                if distance_m > family_radius_meters:
                    continue
                if best_distance is None or distance_m < best_distance:
                    best_distance = distance_m
                    target_family = family
            if target_family is None:
                target_family = {
                    "family_id": f"family-{len(families) + 1}",
                    "center_lat": lat,
                    "center_lon": lon,
                    "weight_sum": max(0.001, pano_score),
                    "panoramas": [],
                    "model_hits": set(),
                    "capture_hits": 0,
                    "heading_support": 0,
                    "best_panorama": panorama,
                }
                families.append(target_family)
            else:
                weight = max(0.001, pano_score)
                total_weight = float(target_family["weight_sum"]) + weight
                target_family["center_lat"] = (
                    (float(target_family["center_lat"]) * float(target_family["weight_sum"]))
                    + (lat * weight)
                ) / total_weight
                target_family["center_lon"] = (
                    (float(target_family["center_lon"]) * float(target_family["weight_sum"]))
                    + (lon * weight)
                ) / total_weight
                target_family["weight_sum"] = total_weight
            target_family["panoramas"].append(panorama)
            target_family["capture_hits"] += int(panorama.get("capture_hits", 0))
            target_family["heading_support"] = max(
                int(target_family["heading_support"]),
                int(panorama.get("heading_support", 0)),
            )
            target_family["model_hits"].update(
                str(item) for item in list(panorama.get("model_hits") or [])
            )
            if float(panorama.get("panorama_score", 0.0)) >= float(
                target_family["best_panorama"].get("panorama_score", 0.0)
            ):
                target_family["best_panorama"] = panorama

        ranked_families: List[dict] = []
        for family in families:
            panoramas_in_family = sorted(
                family["panoramas"],
                key=lambda row: float(row.get("panorama_score", 0.0)),
                reverse=True,
            )
            top_panorama_scores = [
                float(row.get("panorama_score", 0.0))
                for row in panoramas_in_family[:4]
            ]
            family_score = sum(top_panorama_scores)
            family_score += 0.08 * float(min(len(panoramas_in_family), 5))
            family_score += 0.03 * float(min(int(family["heading_support"]), 6))
            family_score += 0.05 * float(min(len(family["model_hits"]), 3))
            best_panorama = dict(family["best_panorama"])
            ranked_families.append(
                {
                    **best_panorama,
                    "family_id": family["family_id"],
                    "family_score": round(family_score, 6),
                    "family_center_lat": round(float(family["center_lat"]), 7),
                    "family_center_lon": round(float(family["center_lon"]), 7),
                    "family_radius_meters": float(family_radius_meters),
                    "family_panorama_count": len(panoramas_in_family),
                    "family_capture_hits": int(family["capture_hits"]),
                    "family_model_support": len(family["model_hits"]),
                    "family_panorama_ids": [
                        int(item.get("panorama_id", 0)) for item in panoramas_in_family[:10]
                    ],
                }
            )
        ranked_families.sort(
            key=lambda row: float(row.get("family_score", 0.0)), reverse=True
        )
        return ranked_families

    @router.get("/api/retrieval/index-stats")
    async def retrieval_index_stats():
        retrieval_id = new_retrieval_id()
        log_retrieval_event(retrieval_id, "index_stats_started")
        db = get_db()
        vector_store = _resolve_vector_store(db)
        try:
            embedders = list(get_retrieval_embedders())
            model_stats = []
            for embedder in embedders:
                model_stats.append(
                    vector_store.get_capture_embedding_stats(
                        embedder.model_name,
                        embedder.model_version,
                        embedding_base=str(
                            getattr(embedder, "embedding_base", EMBEDDING_BASE_CLIP)
                        ),
                    )
                )
            primary = model_stats[0] if model_stats else {
                "vector_enabled": vector_store.is_vector_ready(),
                "model_name": "",
                "model_version": "",
                "total_captures": 0,
                "embedded_captures": 0,
                "pending_captures": 0,
            }
            stats = {**primary, "models": model_stats}
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

        db = get_db()
        vector_store = _resolve_vector_store(db)
        try:
            if not vector_store.is_vector_ready():
                raise HTTPException(
                    status_code=503,
                    detail="Vector search backend is unavailable. Check vector backend configuration.",
                )
            embedders = _embedders_for_base_or_400(embedding_base)
            active_embedders, coverage_skipped_models = _select_query_embedders(
                vector_store, embedders
            )
            if coverage_skipped_models:
                log_retrieval_event(
                    retrieval_id,
                    "search_models_skipped_low_coverage",
                    skipped_models=coverage_skipped_models,
                )
            rows, failed_models, model_timings_ms = _search_candidates(
                image_bytes=image_bytes,
                embedders=active_embedders,
                vector_store=vector_store,
                retrieval_id=retrieval_id,
                min_similarity=min_similarity,
                top_k=max(1, min(200, int(top_k))),
                candidate_multiplier=RETRIEVAL_SEARCH_CANDIDATE_MULTIPLIER,
                max_candidates=RETRIEVAL_SEARCH_MAX_CANDIDATES,
            )
            rows = rows[: max(1, min(200, int(top_k)))]
            if not rows and failed_models:
                log_retrieval_event(
                    retrieval_id,
                    "search_all_models_failed",
                    failed_models=failed_models,
                )
                raise HTTPException(
                    status_code=503,
                    detail="All retrieval models failed to encode. Check model settings.",
                )
            _attach_web_paths(rows)
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            search_timings_ms: Dict[str, Any] = {
                "model_compute_total": round(
                    sum(float(v.get("total", 0.0)) for v in model_timings_ms.values()), 2
                ),
                "model_timings": model_timings_ms,
                "total": elapsed_ms,
            }
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

    @router.post("/api/retrieval/locate-by-image")
    async def retrieval_locate_by_image(
        image: UploadFile = File(...),
        top_k: int = Form(8),
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
            "locate_started",
            upload_bytes=len(image_bytes),
            top_k=top_k,
            min_similarity=min_similarity,
            candidate_multiplier=LOCATE_SEARCH_CANDIDATE_MULTIPLIER,
            max_candidates=LOCATE_SEARCH_MAX_CANDIDATES,
            family_radius_meters=LOCATE_FAMILY_RADIUS_METERS,
            ivfflat_probes=RETRIEVAL_IVFFLAT_PROBES,
        )

        db = get_db()
        vector_store = _resolve_vector_store(db)
        try:
            if not vector_store.is_vector_ready():
                raise HTTPException(
                    status_code=503,
                    detail="Vector search backend is unavailable. Check vector backend configuration.",
                )
            active_embedders, coverage_skipped_models = _all_active_embedders_or_400(
                vector_store
            )
            if coverage_skipped_models:
                log_retrieval_event(
                    retrieval_id,
                    "locate_models_skipped_low_coverage",
                    skipped_models=coverage_skipped_models,
                )

            stage_vector_started = time.perf_counter()
            vector_rows, failed_models, model_timings_ms = _search_candidates(
                image_bytes=image_bytes,
                embedders=active_embedders,
                vector_store=vector_store,
                retrieval_id=retrieval_id,
                min_similarity=min_similarity,
                top_k=max(8, int(top_k) * 8),
                candidate_multiplier=LOCATE_SEARCH_CANDIDATE_MULTIPLIER,
                max_candidates=LOCATE_SEARCH_MAX_CANDIDATES,
            )
            if not vector_rows and failed_models:
                raise HTTPException(
                    status_code=503,
                    detail="All retrieval models failed to encode. Check model settings.",
                )
            _attach_web_paths(vector_rows)
            vector_stage_ms = round(
                (time.perf_counter() - stage_vector_started) * 1000.0, 2
            )

            stage_panorama_started = time.perf_counter()
            panorama_rows = _aggregate_panorama_candidates(
                vector_rows[:LOCATE_SEARCH_MAX_CANDIDATES],
                model_count=len(active_embedders),
            )[:LOCATE_PANORAMA_CANDIDATE_LIMIT]
            panorama_stage_ms = round(
                (time.perf_counter() - stage_panorama_started) * 1000.0, 2
            )

            stage_family_started = time.perf_counter()
            family_rows = _cluster_panorama_families(
                panorama_rows, family_radius_meters=LOCATE_FAMILY_RADIUS_METERS
            )[: max(1, min(50, int(top_k)))]
            family_stage_ms = round(
                (time.perf_counter() - stage_family_started) * 1000.0, 2
            )

            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            pipeline_stages = [
                {
                    "key": "vector_search",
                    "title": "Vector search",
                    "status": "completed",
                    "detail": (
                        f"Searched {len(active_embedders)} active models across "
                        f"{len({str(getattr(e, 'embedding_base', 'clip')) for e in active_embedders})} "
                        "embedding families."
                    ),
                    "timings_ms": {
                        "stage": vector_stage_ms,
                        "models": model_timings_ms,
                    },
                },
                {
                    "key": "panorama_rerank",
                    "title": "Panorama aggregation",
                    "status": "completed",
                    "detail": (
                        f"Collapsed {len(vector_rows)} capture hits into "
                        f"{len(panorama_rows)} panorama candidates."
                    ),
                    "timings_ms": {"stage": panorama_stage_ms},
                },
                {
                    "key": "family_rank",
                    "title": "Panorama-family ranking",
                    "status": "completed",
                    "detail": (
                        f"Clustered panoramas within {int(LOCATE_FAMILY_RADIUS_METERS)}m "
                        f"into {len(family_rows)} location families."
                    ),
                    "timings_ms": {"stage": family_stage_ms},
                },
            ]
            primary = active_embedders[0]
            log_retrieval_event(
                retrieval_id,
                "locate_completed",
                matches=len(family_rows),
                capture_candidates=len(vector_rows),
                panorama_candidates=len(panorama_rows),
                elapsed_ms=elapsed_ms,
                model_count=len(active_embedders),
                failed_models=len(failed_models),
                model_name=primary.model_name,
                model_version=primary.model_version,
            )
            return JSONResponse(
                {
                    "retrieval_id": retrieval_id,
                    "mode": "locate",
                    "model_name": primary.model_name,
                    "model_version": primary.model_version,
                    "models": [
                        {
                            "model_id": getattr(embedder, "model_id", embedder.model_name),
                            "model_name": embedder.model_name,
                            "model_version": embedder.model_version,
                            "embedding_base": str(
                                getattr(embedder, "embedding_base", EMBEDDING_BASE_CLIP)
                            ),
                            "weight": float(getattr(embedder, "weight", 1.0)),
                        }
                        for embedder in active_embedders
                    ],
                    "skipped_models": coverage_skipped_models,
                    "failed_models": failed_models,
                    "capture_candidates": len(vector_rows),
                    "panorama_candidates": len(panorama_rows),
                    "matches": family_rows,
                    "pipeline": {"stages": pipeline_stages},
                    "timings_ms": {
                        "vector_search": vector_stage_ms,
                        "panorama_rerank": panorama_stage_ms,
                        "family_rank": family_stage_ms,
                        "model_timings": model_timings_ms,
                        "total": elapsed_ms,
                    },
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
        vector_store = _resolve_vector_store(db)
        indexed = 0
        skipped = 0
        failed = []
        model_indexed = {}
        try:
            if not vector_store.is_vector_ready():
                raise HTTPException(
                    status_code=503,
                    detail="Vector search backend is unavailable. Check vector backend configuration.",
                )
            if (
                hasattr(vector_store, "supports_missing_embedding_backfill")
                and not vector_store.supports_missing_embedding_backfill()
            ):
                backend_name = str(getattr(vector_store, "backend_name", "vector"))
                raise HTTPException(
                    status_code=501,
                    detail=(
                        f"Vector backend '{backend_name}' does not support "
                        "missing-embedding backfill indexing."
                    ),
                )
            if not embedders:
                raise HTTPException(status_code=503, detail="No retrieval models configured")
            rows = vector_store.list_captures_missing_any_embeddings(
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
                        upserted = vector_store.upsert_capture_embeddings_batch(
                            [
                                (capture_id, vector)
                                for (capture_id, _), vector in zip(valid_batch, vectors)
                            ],
                            embedder.model_name,
                            embedder.model_version,
                            embedding_base=embedding_base,
                        )
                        key = f"{embedder.model_name}:{embedder.model_version}"
                        model_indexed[key] = model_indexed.get(key, 0) + int(upserted)
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
                                vector_store.upsert_capture_embedding(
                                    capture_id,
                                    embedder.model_name,
                                    embedder.model_version,
                                    vector,
                                    embedding_base=embedding_base,
                                )
                                key = f"{embedder.model_name}:{embedder.model_version}"
                                model_indexed[key] = model_indexed.get(key, 0) + 1
                            indexed += 1
                        except Exception as single_exc:
                            skipped += 1
                            failed.append(
                                {"capture_id": capture_id, "error": str(single_exc)}
                            )
            stats = vector_store.get_capture_embedding_stats(
                embedders[0].model_name,
                embedders[0].model_version,
                embedding_base=embedding_base,
            )
            stats["models"] = [
                vector_store.get_capture_embedding_stats(
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
