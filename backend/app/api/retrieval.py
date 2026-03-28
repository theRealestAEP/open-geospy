import base64
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
LOCATE_ORB_ENABLED_DEFAULT = (
    os.getenv("GEOSPY_LOCATE_ORB_ENABLED", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
LOCATE_ORB_TOP_N_DEFAULT = max(
    1, min(5000, int(os.getenv("GEOSPY_LOCATE_ORB_TOP_N", "100")))
)
LOCATE_ORB_WEIGHT_DEFAULT = max(
    0.0, min(5.0, float(os.getenv("GEOSPY_LOCATE_ORB_WEIGHT", "0.75")))
)
LOCATE_ORB_FEATURES = max(
    100, min(2000, int(os.getenv("GEOSPY_LOCATE_ORB_FEATURES", "500")))
)
LOCATE_ORB_RANSAC_TOP_K_DEFAULT = max(
    0, min(500, int(os.getenv("GEOSPY_LOCATE_ORB_RANSAC_TOP_K", "10")))
)
LOCATE_ORB_RATIO_TEST = max(
    0.5, min(0.95, float(os.getenv("GEOSPY_LOCATE_ORB_RATIO_TEST", "0.75")))
)
LOCATE_ORB_RANSAC_REPROJ_THRESHOLD = max(
    1.0,
    min(
        25.0,
        float(os.getenv("GEOSPY_LOCATE_ORB_RANSAC_REPROJ_THRESHOLD", "5.0")),
    ),
)
LOCATE_ORB_VISUALIZATION_LIMIT = max(
    1, min(12, int(os.getenv("GEOSPY_LOCATE_ORB_VISUALIZATION_LIMIT", "6")))
)
LOCATE_ORB_VISUALIZATION_MATCH_LIMIT = max(
    8, min(80, int(os.getenv("GEOSPY_LOCATE_ORB_VISUALIZATION_MATCH_LIMIT", "28")))
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


def _parse_boolish(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return bool(default)
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_result_limit(value: int, *, default: int = 1, maximum: int = 200) -> int:
    return max(int(default), min(int(maximum), int(value)))


def _import_orb_runtime():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "ORB rerank dependencies are missing. Install opencv-python-headless."
        ) from exc
    return cv2, np


def _encode_cv_image_data_url(cv2, image, *, quality: int = 76) -> str:
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(95, int(quality)))],
    )
    if not success:
        return ""
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


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

    def _run_vector_search_stage(
        *,
        image_bytes: bytes,
        vector_store,
        retrieval_id: str,
        embedding_base: str,
        min_similarity: Optional[float],
        vector_top_k: int,
        low_coverage_event: str,
    ) -> Tuple[
        List[dict],
        List[object],
        List[dict],
        List[dict],
        Dict[str, Dict[str, float]],
        float,
    ]:
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
                low_coverage_event,
                skipped_models=coverage_skipped_models,
            )

        stage_vector_started = time.perf_counter()
        rows, failed_models, model_timings_ms = _search_candidates(
            image_bytes=image_bytes,
            embedders=active_embedders,
            vector_store=vector_store,
            retrieval_id=retrieval_id,
            min_similarity=min_similarity,
            top_k=vector_top_k,
            candidate_multiplier=RETRIEVAL_SEARCH_CANDIDATE_MULTIPLIER,
            max_candidates=RETRIEVAL_SEARCH_MAX_CANDIDATES,
        )
        rows = rows[:vector_top_k]
        if not rows and failed_models:
            raise HTTPException(
                status_code=503,
                detail="All retrieval models failed to encode. Check model settings.",
            )
        _attach_web_paths(rows)
        vector_stage_ms = round((time.perf_counter() - stage_vector_started) * 1000.0, 2)
        return (
            rows,
            active_embedders,
            coverage_skipped_models,
            failed_models,
            model_timings_ms,
            vector_stage_ms,
        )

    def _rerank_capture_rows_with_orb(
        rows: Sequence[dict],
        *,
        image_bytes: bytes,
        enabled: bool,
        top_n: int,
        orb_weight: float,
        ransac_top_k: int,
        visualization_limit: int,
    ) -> Tuple[List[dict], Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "enabled": bool(enabled),
            "status": "skipped",
            "reason": "disabled",
            "candidate_count": len(rows),
            "top_n": 0,
            "feature_count": LOCATE_ORB_FEATURES,
            "weight": float(orb_weight),
            "ratio_test": float(LOCATE_ORB_RATIO_TEST),
            "ransac_top_k": int(ransac_top_k),
            "query_keypoints": 0,
            "candidates_extracted": 0,
            "candidates_scored": 0,
            "ransac_checked": 0,
            "missing_files": 0,
            "decode_errors": 0,
            "visualization_limit": int(max(1, visualization_limit)),
            "ignore_bottom_ratio": float(LOCATE_ORB_IGNORE_BOTTOM_RATIO),
            "comparisons": [],
            "query_fingerprint_data_url": "",
            "timings_ms": {
                "query_extract": 0.0,
                "candidate_read": 0.0,
                "candidate_extract": 0.0,
                "match": 0.0,
                "ransac": 0.0,
                "stage": 0.0,
            },
        }
        reranked = list(rows)
        if not enabled:
            return reranked, stats
        if not reranked:
            stats["reason"] = "no-candidates"
            return reranked, stats

        cv2, np = _import_orb_runtime()
        stage_started = time.perf_counter()
        query_array = np.frombuffer(image_bytes, dtype=np.uint8)
        query_color = cv2.imdecode(query_array, cv2.IMREAD_COLOR)
        if query_color is None:
            raise RuntimeError("Failed to decode query image for ORB rerank")
        query_image = cv2.cvtColor(query_color, cv2.COLOR_BGR2GRAY)
        query_mask, query_ignored_bottom_pixels = _build_orb_feature_mask(
            np, query_image.shape
        )

        orb = cv2.ORB_create(nfeatures=LOCATE_ORB_FEATURES)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        t_query = time.perf_counter()
        query_keypoints, query_descriptors = orb.detectAndCompute(
            query_image, query_mask
        )
        stats["timings_ms"]["query_extract"] = round(
            (time.perf_counter() - t_query) * 1000.0, 2
        )
        stats["query_keypoints"] = len(query_keypoints or [])
        limit = min(len(reranked), max(1, int(top_n)))
        stats["top_n"] = limit
        comparison_limit = min(limit, max(1, int(visualization_limit)))
        query_visual = _annotate_orb_focus_mask(
            cv2,
            query_color,
            ignored_bottom_pixels=query_ignored_bottom_pixels,
        )
        query_fingerprint = cv2.drawKeypoints(
            query_visual,
            query_keypoints or [],
            None,
            color=(87, 214, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        stats["query_fingerprint_data_url"] = _encode_cv_image_data_url(
            cv2,
            query_fingerprint,
            quality=72,
        )
        if not query_keypoints or query_descriptors is None:
            stats["reason"] = "query-has-no-features"
            return reranked, stats

        candidate_read_ms = 0.0
        candidate_extract_ms = 0.0
        match_ms = 0.0
        ransac_ms = 0.0
        comparisons_by_capture_id: Dict[int, dict] = {}
        for idx, row in enumerate(reranked[:limit]):
            raw_path = row.get("filepath", "")
            abs_path = capture_abs_path(raw_path)
            if not abs_path or not os.path.exists(abs_path):
                stats["missing_files"] += 1
                continue

            t_read = time.perf_counter()
            candidate_color = cv2.imread(abs_path, cv2.IMREAD_COLOR)
            candidate_read_ms += (time.perf_counter() - t_read) * 1000.0
            if candidate_color is None:
                stats["decode_errors"] += 1
                continue
            candidate_image = cv2.cvtColor(candidate_color, cv2.COLOR_BGR2GRAY)
            candidate_mask, candidate_ignored_bottom_pixels = _build_orb_feature_mask(
                np, candidate_image.shape
            )

            t_extract = time.perf_counter()
            candidate_keypoints, candidate_descriptors = orb.detectAndCompute(
                candidate_image, candidate_mask
            )
            candidate_extract_ms += (time.perf_counter() - t_extract) * 1000.0
            if not candidate_keypoints or candidate_descriptors is None:
                continue
            stats["candidates_extracted"] += 1

            t_match = time.perf_counter()
            pairs = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)
            good_matches = []
            for pair in pairs:
                if len(pair) < 2:
                    continue
                first, second = pair
                if first.distance < LOCATE_ORB_RATIO_TEST * second.distance:
                    good_matches.append(first)
            good_matches.sort(key=lambda match: float(match.distance))
            match_ms += (time.perf_counter() - t_match) * 1000.0

            denom = float(
                max(1, min(len(query_keypoints), len(candidate_keypoints)))
            )
            good_match_count = len(good_matches)
            good_ratio = float(good_match_count) / denom
            mask = None
            inlier_count = 0
            inlier_ratio = 0.0
            if (
                ransac_top_k > 0
                and idx < int(ransac_top_k)
                and good_match_count >= 4
            ):
                src_points = np.float32(
                    [query_keypoints[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                dst_points = np.float32(
                    [candidate_keypoints[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                t_ransac = time.perf_counter()
                _, mask = cv2.findHomography(
                    src_points,
                    dst_points,
                    cv2.RANSAC,
                    LOCATE_ORB_RANSAC_REPROJ_THRESHOLD,
                )
                ransac_ms += (time.perf_counter() - t_ransac) * 1000.0
                stats["ransac_checked"] += 1
                if mask is not None:
                    inlier_count = int(mask.ravel().sum())
                    inlier_ratio = float(inlier_count) / denom

            capture_id = int(row.get("capture_id") or 0)
            score_before = float(row.get("score", 0.0))
            orb_score = max(good_ratio, inlier_ratio)
            row["orb_score"] = round(orb_score, 6)
            row["orb_good_matches"] = int(good_match_count)
            row["orb_inliers"] = int(inlier_count)
            row["orb_reranked"] = True
            row["score"] = score_before + (float(orb_weight) * orb_score)
            stats["candidates_scored"] += 1

            if capture_id > 0 and len(comparisons_by_capture_id) < comparison_limit:
                visual_matches = good_matches[: int(LOCATE_ORB_VISUALIZATION_MATCH_LIMIT)]
                match_visual = cv2.drawMatches(
                    query_visual,
                    query_keypoints,
                    _annotate_orb_focus_mask(
                        cv2,
                        candidate_color,
                        ignored_bottom_pixels=candidate_ignored_bottom_pixels,
                    ),
                    candidate_keypoints,
                    visual_matches,
                    None,
                    matchColor=(255, 209, 102),
                    singlePointColor=(91, 123, 177),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                comparisons_by_capture_id[capture_id] = {
                    "capture_id": capture_id,
                    "panorama_id": int(row.get("panorama_id") or 0),
                    "pano_id": str(row.get("pano_id") or ""),
                    "web_path": str(row.get("web_path") or ""),
                    "lat": float(row.get("lat") or 0.0),
                    "lon": float(row.get("lon") or 0.0),
                    "heading": float(row.get("heading") or 0.0),
                    "similarity": round(float(row.get("similarity") or 0.0), 6),
                    "score_before": round(score_before, 6),
                    "score_after": round(float(row.get("score") or 0.0), 6),
                    "orb_score": round(orb_score, 6),
                    "orb_good_matches": int(good_match_count),
                    "orb_inliers": int(inlier_count),
                    "visual_match_count": int(len(visual_matches)),
                    "query_keypoints": int(len(query_keypoints or [])),
                    "candidate_keypoints": int(len(candidate_keypoints or [])),
                    "rank_before": int(idx + 1),
                    "visualization_data_url": _encode_cv_image_data_url(
                        cv2,
                        match_visual,
                        quality=70,
                    ),
                }

        reranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        ranked_comparisons: List[dict] = []
        for rank_after, row in enumerate(reranked, start=1):
            capture_id = int(row.get("capture_id") or 0)
            comparison = comparisons_by_capture_id.get(capture_id)
            if not comparison:
                continue
            ranked_comparisons.append(
                {
                    **comparison,
                    "rank_after": int(rank_after),
                    "score_after": round(float(row.get("score") or 0.0), 6),
                    "orb_score": round(float(row.get("orb_score") or comparison["orb_score"]), 6),
                    "orb_good_matches": int(
                        row.get("orb_good_matches") or comparison["orb_good_matches"]
                    ),
                    "orb_inliers": int(row.get("orb_inliers") or comparison["orb_inliers"]),
                    "visual_match_count": int(
                        comparison.get("visual_match_count") or 0
                    ),
                }
            )
            if len(ranked_comparisons) >= comparison_limit:
                break
        stats["comparisons"] = ranked_comparisons
        stats["status"] = "completed"
        stats["reason"] = ""
        stats["timings_ms"] = {
            "query_extract": round(float(stats["timings_ms"]["query_extract"]), 2),
            "candidate_read": round(candidate_read_ms, 2),
            "candidate_extract": round(candidate_extract_ms, 2),
            "match": round(match_ms, 2),
            "ransac": round(ransac_ms, 2),
            "stage": round((time.perf_counter() - stage_started) * 1000.0, 2),
        }
        return reranked, stats

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
            best_capture_score = float(best_capture.get("score", 0.0))
            panorama_score = best_capture_score
            ranked.append(
                {
                    **best_capture,
                    "panorama_score": round(panorama_score, 6),
                    "best_capture_score": round(best_capture_score, 6),
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
            best_panorama = dict(family["best_panorama"])
            family_score = float(best_panorama.get("panorama_score", 0.0))
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
        vector_top_k = _normalize_result_limit(top_k)
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
            vector_top_k=vector_top_k,
        )

        db = get_db()
        vector_store = _resolve_vector_store(db)
        try:
            (
                rows,
                active_embedders,
                coverage_skipped_models,
                failed_models,
                model_timings_ms,
                _vector_stage_ms,
            ) = _run_vector_search_stage(
                image_bytes=image_bytes,
                vector_store=vector_store,
                retrieval_id=retrieval_id,
                embedding_base=embedding_base,
                min_similarity=min_similarity,
                vector_top_k=vector_top_k,
                low_coverage_event="search_models_skipped_low_coverage",
            )
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
        embedding_base: str = Form(RETRIEVAL_EMBEDDING_BASE_DEFAULT),
        orb_enabled: Optional[str] = Form(None),
        orb_top_n: Optional[int] = Form(None),
        orb_weight: Optional[float] = Form(None),
        orb_ransac_top_k: Optional[int] = Form(None),
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
        vector_top_k = _normalize_result_limit(top_k)
        resolved_orb_enabled = _parse_boolish(
            orb_enabled, default=LOCATE_ORB_ENABLED_DEFAULT
        )
        resolved_orb_top_n = max(
            1,
            min(
                LOCATE_SEARCH_MAX_CANDIDATES,
                int(
                    LOCATE_ORB_TOP_N_DEFAULT if orb_top_n is None else orb_top_n
                ),
            ),
        )
        resolved_orb_weight = max(
            0.0,
            min(
                5.0,
                float(
                    LOCATE_ORB_WEIGHT_DEFAULT if orb_weight is None else orb_weight
                ),
            ),
        )
        resolved_orb_ransac_top_k = max(
            0,
            min(
                resolved_orb_top_n,
                int(
                    LOCATE_ORB_RANSAC_TOP_K_DEFAULT
                    if orb_ransac_top_k is None
                    else orb_ransac_top_k
                ),
            ),
        )
        log_retrieval_event(
            retrieval_id,
            "locate_started",
            upload_bytes=len(image_bytes),
            top_k=top_k,
            min_similarity=min_similarity,
            candidate_multiplier=RETRIEVAL_SEARCH_CANDIDATE_MULTIPLIER,
            max_candidates=RETRIEVAL_SEARCH_MAX_CANDIDATES,
            family_radius_meters=LOCATE_FAMILY_RADIUS_METERS,
            ivfflat_probes=RETRIEVAL_IVFFLAT_PROBES,
            embedding_base=embedding_base,
            vector_top_k=vector_top_k,
            orb_enabled=resolved_orb_enabled,
            orb_top_n=resolved_orb_top_n,
            orb_weight=resolved_orb_weight,
            orb_ransac_top_k=resolved_orb_ransac_top_k,
        )

        db = get_db()
        vector_store = _resolve_vector_store(db)
        try:
            (
                vector_rows,
                active_embedders,
                coverage_skipped_models,
                failed_models,
                model_timings_ms,
                vector_stage_ms,
            ) = _run_vector_search_stage(
                image_bytes=image_bytes,
                vector_store=vector_store,
                retrieval_id=retrieval_id,
                embedding_base=embedding_base,
                min_similarity=min_similarity,
                vector_top_k=vector_top_k,
                low_coverage_event="locate_models_skipped_low_coverage",
            )

            orb_rows = vector_rows
            orb_stage_ms = 0.0
            orb_stats: Dict[str, Any] = {
                "enabled": bool(resolved_orb_enabled),
                "status": "skipped",
                "reason": "disabled",
                "top_n": resolved_orb_top_n,
                "weight": resolved_orb_weight,
                "ransac_top_k": resolved_orb_ransac_top_k,
                "timings_ms": {"stage": 0.0},
            }
            if resolved_orb_enabled:
                stage_orb_started = time.perf_counter()
                try:
                    orb_rows, orb_stats = _rerank_capture_rows_with_orb(
                        vector_rows,
                        image_bytes=image_bytes,
                        enabled=resolved_orb_enabled,
                        top_n=resolved_orb_top_n,
                        orb_weight=resolved_orb_weight,
                        ransac_top_k=resolved_orb_ransac_top_k,
                        visualization_limit=vector_top_k,
                    )
                except RuntimeError as exc:
                    raise HTTPException(status_code=503, detail=str(exc))
                orb_stage_ms = round(
                    (time.perf_counter() - stage_orb_started) * 1000.0, 2
                )

            stage_panorama_started = time.perf_counter()
            panorama_rows = _aggregate_panorama_candidates(
                orb_rows,
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
                        f"Collapsed {len(orb_rows)} capture hits into "
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
            orb_stage = {
                "key": "orb_rerank",
                "title": "ORB rerank",
                "status": str(orb_stats.get("status", "skipped")),
                "detail": "",
                "timings_ms": dict(orb_stats.get("timings_ms") or {"stage": orb_stage_ms}),
            }
            if resolved_orb_enabled:
                if orb_stage["status"] == "completed":
                    orb_stage["detail"] = (
                        f"Reranked top {int(orb_stats.get('top_n', 0))} vector candidates "
                        f"with ORB, scored {int(orb_stats.get('candidates_scored', 0))} "
                        f"images and RANSAC-checked {int(orb_stats.get('ransac_checked', 0))}."
                    )
                else:
                    orb_stage["detail"] = str(
                        orb_stats.get("reason") or "ORB rerank did not run."
                    )
            else:
                orb_stage["detail"] = "ORB rerank disabled for this request."
            pipeline_stages.insert(1, orb_stage)
            primary = active_embedders[0]
            log_retrieval_event(
                retrieval_id,
                "locate_completed",
                matches=len(family_rows),
                capture_candidates=len(orb_rows),
                panorama_candidates=len(panorama_rows),
                elapsed_ms=elapsed_ms,
                embedding_base=embedding_base,
                model_count=len(active_embedders),
                failed_models=len(failed_models),
                model_name=primary.model_name,
                model_version=primary.model_version,
                orb_enabled=resolved_orb_enabled,
                orb_top_n=resolved_orb_top_n,
                orb_weight=resolved_orb_weight,
                orb_ransac_top_k=resolved_orb_ransac_top_k,
                orb_stage_status=orb_stage["status"],
            )
            return JSONResponse(
                {
                    "retrieval_id": retrieval_id,
                    "mode": "locate",
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
                    "capture_candidates": len(orb_rows),
                    "panorama_candidates": len(panorama_rows),
                    "orb": {
                        "enabled": resolved_orb_enabled,
                        "top_n": resolved_orb_top_n,
                        "weight": resolved_orb_weight,
                        "ransac_top_k": resolved_orb_ransac_top_k,
                        "stats": orb_stats,
                    },
                    "matches": family_rows,
                    "pipeline": {"stages": pipeline_stages},
                    "timings_ms": {
                        "vector_search": vector_stage_ms,
                        "orb_rerank": round(
                            float(
                                (orb_stats.get("timings_ms") or {}).get(
                                    "stage", orb_stage_ms
                                )
                            ),
                            2,
                        ),
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
