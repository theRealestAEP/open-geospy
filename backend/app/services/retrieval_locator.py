import io
import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None


log = logging.getLogger(__name__)
LOCATOR_DB_MAX_TOP_K = max(200, int(os.getenv("GEOSPY_LOCATOR_DB_MAX_TOP_K", "5000")))
LOCATOR_IVFFLAT_PROBES = max(1, int(os.getenv("GEOSPY_RETRIEVAL_IVFFLAT_PROBES", "120")))
LOCATOR_MODAL_RERANK_ENABLED = (
    os.getenv("GEOSPY_LOCATOR_MODAL_RERANK_ENABLED", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)
LOCATOR_MODAL_RERANK_REQUIRE_SUCCESS = (
    os.getenv("GEOSPY_LOCATOR_MODAL_RERANK_REQUIRE_SUCCESS", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
LOCATOR_MODAL_RERANK_TOP_N = max(
    1, int(os.getenv("GEOSPY_LOCATOR_MODAL_RERANK_TOP_N", "5000"))
)
LOCATOR_MODAL_RERANK_WEIGHT = max(
    0.0, min(0.95, float(os.getenv("GEOSPY_LOCATOR_MODAL_RERANK_WEIGHT", "0.35")))
)
LOCATOR_MODAL_RERANK_MAX_WORKERS = max(
    1, int(os.getenv("GEOSPY_MODAL_RERANK_MAX_WORKERS", "64"))
)
LOCATOR_MODAL_RERANK_BATCH_SIZE = max(
    1, int(os.getenv("GEOSPY_MODAL_RERANK_BATCH_SIZE", "48"))
)
LOCATOR_MODAL_RERANK_MAX_RETRIES = max(
    0, int(os.getenv("GEOSPY_MODAL_RERANK_MAX_RETRIES", "1"))
)
LOCATOR_MODAL_RERANK_ENVIRONMENT = os.getenv(
    "GEOSPY_MODAL_RETRIEVAL_ENVIRONMENT",
    os.getenv("GEOSPY_MODAL_EMBED_ENVIRONMENT", "google-map-walkers"),
)
LOCATOR_MODAL_RERANK_MODEL = os.getenv("GEOSPY_MODAL_RERANK_MODEL", "ViT-L-14")
LOCATOR_MODAL_RERANK_PRETRAINED = os.getenv(
    "GEOSPY_MODAL_RERANK_PRETRAINED", "laion2b_s32b_b82k"
)
LOCATOR_MODAL_FEATURE_VERIFY_ENABLED = (
    os.getenv("GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_ENABLED", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)
LOCATOR_MODAL_FEATURE_VERIFY_REQUIRE_SUCCESS = (
    os.getenv("GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_REQUIRE_SUCCESS", "0")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
LOCATOR_MODAL_FEATURE_VERIFY_TOP_N = max(
    1, int(os.getenv("GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_TOP_N", "220"))
)
LOCATOR_MODAL_FEATURE_VERIFY_MAX_WORKERS = max(
    1, int(os.getenv("GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_MAX_WORKERS", "64"))
)
LOCATOR_MODAL_FEATURE_VERIFY_BATCH_SIZE = max(
    1, int(os.getenv("GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_BATCH_SIZE", "24"))
)
LOCATOR_MODAL_FEATURE_VERIFY_MAX_RETRIES = max(
    0, int(os.getenv("GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_MAX_RETRIES", "1"))
)
LOCATOR_MODAL_FEATURE_VERIFY_MAX_KEYPOINTS = max(
    128, int(os.getenv("GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_MAX_KEYPOINTS", "2048"))
)
LOCATOR_MODAL_FEATURE_VERIFY_RESIZE_LONG_EDGE = max(
    256, int(os.getenv("GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_RESIZE_LONG_EDGE", "1280"))
)
LOCATOR_MODAL_FEATURE_VERIFY_MIN_MATCHES = max(
    4, int(os.getenv("GEOSPY_LOCATOR_MODAL_FEATURE_VERIFY_MIN_MATCHES", "8"))
)


def new_retrieval_id() -> str:
    return uuid.uuid4().hex[:12]


def log_retrieval_event(retrieval_id: str, event: str, **fields):
    payload = {"retrieval_id": retrieval_id, "event": event, **fields}
    try:
        log.info("retrieval %s", json.dumps(payload, default=str, sort_keys=True))
    except Exception:
        log.info("retrieval id=%s event=%s", retrieval_id, event)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _pil_to_jpeg_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=92)
    return out.getvalue()


def _build_query_crops(image_bytes: bytes) -> List[Tuple[str, bytes]]:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        w, h = img.size
        crops: List[Tuple[str, bytes]] = [("full", _pil_to_jpeg_bytes(img))]

        def add_center(name: str, frac: float):
            cw = int(w * frac)
            ch = int(h * frac)
            if cw < 32 or ch < 32:
                return
            x0 = (w - cw) // 2
            y0 = (h - ch) // 2
            crops.append((name, _pil_to_jpeg_bytes(img.crop((x0, y0, x0 + cw, y0 + ch)))))

        def add_box(name: str, box: Tuple[int, int, int, int]):
            x0, y0, x1, y1 = box
            if x1 - x0 < 32 or y1 - y0 < 32:
                return
            crops.append((name, _pil_to_jpeg_bytes(img.crop(box))))

        add_center("center80", 0.8)
        add_center("center60", 0.6)
        add_center("center40", 0.4)
        add_box("left", (0, 0, w // 2, h))
        add_box("right", (w // 2, 0, w, h))
        add_box("top", (0, 0, w, h // 2))
        add_box("bottom", (0, h // 2, w, h))
        add_box("q1", (0, 0, w // 2, h // 2))
        add_box("q2", (w // 2, 0, w, h // 2))
        add_box("q3", (0, h // 2, w // 2, h))
        add_box("q4", (w // 2, h // 2, w, h))
        add_box("upper_center", (w // 4, 0, (3 * w) // 4, (2 * h) // 3))
        add_box("lower_center", (w // 4, h // 3, (3 * w) // 4, h))
        return crops


def _crop_weight(crop_name: str) -> float:
    weights = {
        "full": 0.95,
        "center80": 1.05,
        "center60": 1.15,
        "center40": 1.2,
        "left": 1.0,
        "right": 1.0,
        "top": 0.95,
        "bottom": 1.0,
        "q1": 1.08,
        "q2": 1.08,
        "q3": 1.08,
        "q4": 1.08,
        "upper_center": 1.1,
        "lower_center": 1.1,
    }
    return float(weights.get(crop_name, 1.0))


def _mean_rgb(image_bytes: bytes) -> Optional[Tuple[float, float, float]]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB").resize((64, 64))
            stat = rgb.getbbox()
            if stat is None:
                return None
            hist = rgb.histogram()
            if not hist or len(hist) < 768:
                return None
            total = float(64 * 64)
            channels = []
            for i in range(3):
                channel_hist = hist[i * 256 : (i + 1) * 256]
                mean = sum(v * idx for idx, v in enumerate(channel_hist)) / total
                channels.append(float(mean))
            return (channels[0], channels[1], channels[2])
    except Exception:
        return None


def _appearance_penalty(
    query_mean_rgb: Optional[Tuple[float, float, float]],
    candidate_image_bytes: bytes,
) -> float:
    if query_mean_rgb is None:
        return 0.0
    cand_mean = _mean_rgb(candidate_image_bytes)
    if cand_mean is None:
        return 0.0
    dr = query_mean_rgb[0] - cand_mean[0]
    dg = query_mean_rgb[1] - cand_mean[1]
    db = query_mean_rgb[2] - cand_mean[2]
    distance = math.sqrt((dr * dr) + (dg * dg) + (db * db))
    max_distance = math.sqrt(3.0 * (255.0**2))
    return max(0.0, min(1.0, distance / max_distance))


def _weighted_centroid(items: List[dict], score_key: str) -> Tuple[float, float]:
    sum_w = 0.0
    sum_lat = 0.0
    sum_lon = 0.0
    for item in items:
        w = max(1e-6, float(item.get(score_key, 0.0)))
        lat = float(item.get("lat", 0.0))
        lon = float(item.get("lon", 0.0))
        sum_w += w
        sum_lat += lat * w
        sum_lon += lon * w
    if sum_w <= 0:
        return (0.0, 0.0)
    return (sum_lat / sum_w, sum_lon / sum_w)


def _cluster_candidates(candidates: List[dict], cluster_radius_m: float) -> List[dict]:
    clusters: List[dict] = []
    for item in candidates:
        item_score = float(item.get("score", item.get("vector_score", 0.0)))
        lat = float(item["lat"])
        lon = float(item["lon"])
        chosen_idx = -1
        best_d = float("inf")
        for idx, cluster in enumerate(clusters):
            d = _haversine_m(lat, lon, cluster["lat"], cluster["lon"])
            if d <= cluster_radius_m and d < best_d:
                chosen_idx = idx
                best_d = d
        if chosen_idx < 0:
            clusters.append(
                {
                    "cluster_id": len(clusters),
                    "items": [item],
                    "lat": lat,
                    "lon": lon,
                    "score": item_score,
                }
            )
        else:
            cluster = clusters[chosen_idx]
            cluster["items"].append(item)
            cluster["score"] += item_score
            cluster["lat"], cluster["lon"] = _weighted_centroid(
                cluster["items"], "score" if "score" in item else "vector_score"
            )
    for cluster in clusters:
        centroid_lat = float(cluster["lat"])
        centroid_lon = float(cluster["lon"])
        distances = [
            _haversine_m(float(i["lat"]), float(i["lon"]), centroid_lat, centroid_lon)
            for i in cluster["items"]
        ]
        cluster["radius_m"] = max(distances) if distances else 0.0
        cluster["size"] = len(cluster["items"])
    clusters.sort(key=lambda c: (-float(c["score"]), int(c["cluster_id"])))
    return clusters


def _decode_gray_image(image_bytes: bytes) -> Optional[Any]:
    if cv2 is None or np is None:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def _compute_orb_score(
    query_kp: Any,
    query_desc: Any,
    cand_gray: Any,
    min_good_matches: int,
) -> Dict[str, float]:
    if cv2 is None or np is None:
        return {"geom_score": 0.0, "good_matches": 0.0, "inliers": 0.0, "inlier_ratio": 0.0}
    orb = cv2.ORB_create(nfeatures=1800)
    cand_kp, cand_desc = orb.detectAndCompute(cand_gray, None)
    if query_kp is None or query_desc is None or cand_desc is None:
        return {"geom_score": 0.0, "good_matches": 0.0, "inliers": 0.0, "inlier_ratio": 0.0}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(query_desc, cand_desc, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good_n = len(good)
    if good_n < int(min_good_matches):
        return {
            "geom_score": 0.0,
            "good_matches": float(good_n),
            "inliers": 0.0,
            "inlier_ratio": 0.0,
        }

    src_pts = [query_kp[m.queryIdx].pt for m in good]
    dst_pts = [cand_kp[m.trainIdx].pt for m in good]
    src = np.float32(src_pts).reshape(-1, 1, 2)
    dst = np.float32(dst_pts).reshape(-1, 1, 2)
    h_matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if h_matrix is None or mask is None:
        return {
            "geom_score": 0.0,
            "good_matches": float(good_n),
            "inliers": 0.0,
            "inlier_ratio": 0.0,
        }
    inliers = float(mask.ravel().sum())
    inlier_ratio = inliers / float(max(1, good_n))
    geom_score = min(1.0, (good_n / 80.0)) * inlier_ratio
    return {
        "geom_score": float(max(0.0, geom_score)),
        "good_matches": float(good_n),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
    }


def _apply_modal_rerank(
    *,
    retrieval_id: str,
    query_image_bytes: bytes,
    candidates: List[dict],
    capture_abs_path: Callable[[str], str],
    rerank_top_n: int,
    rerank_weight: float,
    rerank_workers: int,
    rerank_batch_size: int,
    rerank_max_retries: int,
    rerank_environment: str,
    rerank_model_name: str,
    rerank_pretrained: str,
) -> Dict[str, Any]:
    from worker.modal_retrieval_worker import dispatch_modal_rerank

    if not candidates:
        return {
            "enabled": True,
            "considered": 0,
            "loaded_images": 0,
            "missing_images": 0,
            "applied": 0,
            "scores_received": 0,
        }

    pool = sorted(
        candidates,
        key=lambda item: float(item.get("vector_score", 0.0)),
        reverse=True,
    )[: max(1, int(rerank_top_n))]

    payload: List[Tuple[int, bytes]] = []
    missing_images = 0
    for item in pool:
        capture_id = int(item.get("capture_id", 0))
        if capture_id <= 0:
            continue
        abs_path = capture_abs_path(str(item.get("filepath", "")))
        if not abs_path or not os.path.exists(abs_path):
            missing_images += 1
            continue
        try:
            with open(abs_path, "rb") as f:
                payload.append((capture_id, f.read()))
        except Exception:
            missing_images += 1

    if not payload:
        return {
            "enabled": True,
            "considered": len(pool),
            "loaded_images": 0,
            "missing_images": missing_images,
            "applied": 0,
            "scores_received": 0,
        }

    rerank_result = dispatch_modal_rerank(
        query_image_bytes=query_image_bytes,
        candidate_items=payload,
        num_workers=max(1, int(rerank_workers)),
        batch_size=max(1, int(rerank_batch_size)),
        modal_environment=rerank_environment,
        model_name=rerank_model_name,
        pretrained=rerank_pretrained,
        max_retries=max(0, int(rerank_max_retries)),
    )
    scores = dict(rerank_result.get("scores") or {})

    applied = 0
    w = max(0.0, min(0.95, float(rerank_weight)))
    for item in candidates:
        capture_id = int(item.get("capture_id", 0))
        modal_similarity = float(scores.get(capture_id, 0.0))
        item["modal_similarity"] = modal_similarity
        if capture_id in scores:
            base = float(item.get("vector_score", 0.0))
            item["vector_score"] = ((1.0 - w) * base) + (w * modal_similarity)
            applied += 1

    candidates.sort(key=lambda item: float(item.get("vector_score", 0.0)), reverse=True)
    log_retrieval_event(
        retrieval_id,
        "locator_modal_rerank_completed",
        considered=len(pool),
        loaded_images=len(payload),
        missing_images=missing_images,
        applied=applied,
        scores_received=len(scores),
        workers=max(1, int(rerank_workers)),
        batch_size=max(1, int(rerank_batch_size)),
        top_n=max(1, int(rerank_top_n)),
        weight=round(w, 4),
        jobs_total=rerank_result.get("jobs_total", 0),
        jobs_completed=rerank_result.get("jobs_completed", 0),
        jobs_failed=rerank_result.get("jobs_failed", 0),
    )
    return {
        "enabled": True,
        "considered": len(pool),
        "loaded_images": len(payload),
        "missing_images": missing_images,
        "applied": applied,
        "scores_received": len(scores),
        "jobs_total": rerank_result.get("jobs_total", 0),
        "jobs_completed": rerank_result.get("jobs_completed", 0),
        "jobs_failed": rerank_result.get("jobs_failed", 0),
        "model_name": rerank_model_name,
        "pretrained": rerank_pretrained,
    }


def _apply_modal_feature_verify(
    *,
    retrieval_id: str,
    query_image_bytes: bytes,
    verify_candidates: List[dict],
    capture_abs_path: Callable[[str], str],
    query_mean_rgb: Optional[Tuple[float, float, float]],
    verify_top_n: int,
    verify_workers: int,
    verify_batch_size: int,
    verify_max_retries: int,
    verify_environment: str,
    max_keypoints: int,
    resize_long_edge: int,
    min_matches: int,
) -> Dict[str, Any]:
    from worker.modal_retrieval_worker import dispatch_modal_feature_verify

    if not verify_candidates:
        return {
            "enabled": True,
            "considered": 0,
            "loaded_images": 0,
            "missing_images": 0,
            "applied": 0,
            "metrics_received": 0,
        }

    pool = sorted(
        verify_candidates,
        key=lambda item: float(item.get("vector_score", 0.0)),
        reverse=True,
    )[: max(1, int(verify_top_n))]

    payload: List[Tuple[int, bytes]] = []
    appearance_by_capture: Dict[int, float] = {}
    missing_images = 0
    for item in pool:
        capture_id = int(item.get("capture_id", 0))
        if capture_id <= 0:
            continue
        abs_path = capture_abs_path(str(item.get("filepath", "")))
        if not abs_path or not os.path.exists(abs_path):
            missing_images += 1
            continue
        try:
            with open(abs_path, "rb") as f:
                image_bytes = f.read()
            payload.append((capture_id, image_bytes))
            appearance_by_capture[capture_id] = _appearance_penalty(
                query_mean_rgb, image_bytes
            )
        except Exception:
            missing_images += 1

    if not payload:
        return {
            "enabled": True,
            "considered": len(pool),
            "loaded_images": 0,
            "missing_images": missing_images,
            "applied": 0,
            "metrics_received": 0,
        }

    verify_result = dispatch_modal_feature_verify(
        query_image_bytes=query_image_bytes,
        candidate_items=payload,
        num_workers=max(1, int(verify_workers)),
        batch_size=max(1, int(verify_batch_size)),
        modal_environment=str(verify_environment),
        max_retries=max(0, int(verify_max_retries)),
        max_keypoints=max(128, int(max_keypoints)),
        resize_long_edge=max(256, int(resize_long_edge)),
        min_matches=max(4, int(min_matches)),
    )
    metrics = dict(verify_result.get("metrics") or {})

    applied = 0
    for item in verify_candidates:
        capture_id = int(item.get("capture_id", 0))
        item["appearance_penalty"] = float(
            appearance_by_capture.get(capture_id, item.get("appearance_penalty", 0.0))
        )
        metric = metrics.get(capture_id)
        if not metric:
            continue
        geom_score = float(metric.get("geom_score", 0.0))
        item["good_matches"] = max(
            float(item.get("good_matches", 0.0)),
            float(metric.get("good_matches", 0.0)),
        )
        item["inliers"] = max(
            float(item.get("inliers", 0.0)),
            float(metric.get("inliers", 0.0)),
        )
        item["inlier_ratio"] = max(
            float(item.get("inlier_ratio", 0.0)),
            float(metric.get("inlier_ratio", 0.0)),
        )
        item["geom_score"] = max(float(item.get("geom_score", 0.0)), geom_score)
        if geom_score > 0:
            item["geom_source"] = "lightglue"
            applied += 1

    log_retrieval_event(
        retrieval_id,
        "locator_modal_feature_verify_completed",
        considered=len(pool),
        loaded_images=len(payload),
        missing_images=missing_images,
        applied=applied,
        metrics_received=len(metrics),
        workers=max(1, int(verify_workers)),
        batch_size=max(1, int(verify_batch_size)),
        top_n=max(1, int(verify_top_n)),
        max_keypoints=max(128, int(max_keypoints)),
        resize_long_edge=max(256, int(resize_long_edge)),
        min_matches=max(4, int(min_matches)),
        jobs_total=verify_result.get("jobs_total", 0),
        jobs_completed=verify_result.get("jobs_completed", 0),
        jobs_failed=verify_result.get("jobs_failed", 0),
    )
    return {
        "enabled": True,
        "considered": len(pool),
        "loaded_images": len(payload),
        "missing_images": missing_images,
        "applied": applied,
        "metrics_received": len(metrics),
        "jobs_total": verify_result.get("jobs_total", 0),
        "jobs_completed": verify_result.get("jobs_completed", 0),
        "jobs_failed": verify_result.get("jobs_failed", 0),
    }


def locate_image_bytes(
    *,
    image_bytes: bytes,
    db: Any,
    embedders: Sequence[Any],
    capture_abs_path: Callable[[str], str],
    retrieval_id: Optional[str] = None,
    top_k_per_crop: int = 80,
    max_merged_candidates: int = 300,
    panorama_vote_cap: int = 3,
    cluster_radius_m: float = 45.0,
    verify_top_n: int = 20,
    min_similarity: Optional[float] = None,
    model_weights: Optional[Dict[str, float]] = None,
    min_good_matches: int = 14,
    min_inlier_ratio: float = 0.18,
    appearance_penalty_weight: float = 0.22,
    db_max_top_k: int = LOCATOR_DB_MAX_TOP_K,
    ivfflat_probes: int = LOCATOR_IVFFLAT_PROBES,
    modal_rerank_enabled: bool = LOCATOR_MODAL_RERANK_ENABLED,
    modal_rerank_require_success: bool = LOCATOR_MODAL_RERANK_REQUIRE_SUCCESS,
    modal_rerank_top_n: int = LOCATOR_MODAL_RERANK_TOP_N,
    modal_rerank_weight: float = LOCATOR_MODAL_RERANK_WEIGHT,
    modal_rerank_workers: int = LOCATOR_MODAL_RERANK_MAX_WORKERS,
    modal_rerank_batch_size: int = LOCATOR_MODAL_RERANK_BATCH_SIZE,
    modal_rerank_max_retries: int = LOCATOR_MODAL_RERANK_MAX_RETRIES,
    modal_rerank_environment: str = LOCATOR_MODAL_RERANK_ENVIRONMENT,
    modal_rerank_model_name: str = LOCATOR_MODAL_RERANK_MODEL,
    modal_rerank_pretrained: str = LOCATOR_MODAL_RERANK_PRETRAINED,
    modal_feature_verify_enabled: bool = LOCATOR_MODAL_FEATURE_VERIFY_ENABLED,
    modal_feature_verify_require_success: bool = LOCATOR_MODAL_FEATURE_VERIFY_REQUIRE_SUCCESS,
    modal_feature_verify_top_n: int = LOCATOR_MODAL_FEATURE_VERIFY_TOP_N,
    modal_feature_verify_workers: int = LOCATOR_MODAL_FEATURE_VERIFY_MAX_WORKERS,
    modal_feature_verify_batch_size: int = LOCATOR_MODAL_FEATURE_VERIFY_BATCH_SIZE,
    modal_feature_verify_max_retries: int = LOCATOR_MODAL_FEATURE_VERIFY_MAX_RETRIES,
    modal_feature_verify_max_keypoints: int = LOCATOR_MODAL_FEATURE_VERIFY_MAX_KEYPOINTS,
    modal_feature_verify_resize_long_edge: int = LOCATOR_MODAL_FEATURE_VERIFY_RESIZE_LONG_EDGE,
    modal_feature_verify_min_matches: int = LOCATOR_MODAL_FEATURE_VERIFY_MIN_MATCHES,
    include_debug: bool = False,
) -> Dict[str, Any]:
    retrieval_id = retrieval_id or new_retrieval_id()
    timings_ms: Dict[str, float] = {}
    stage_start = time.perf_counter()
    flags: List[str] = []
    debug: Dict[str, Any] = {}
    reason_counts: Dict[str, int] = {}

    log_retrieval_event(
        retrieval_id,
        "locator_started",
        model_count=len(embedders),
        top_k_per_crop=top_k_per_crop,
        max_merged_candidates=max_merged_candidates,
        panorama_vote_cap=panorama_vote_cap,
        cluster_radius_m=cluster_radius_m,
        verify_top_n=verify_top_n,
        min_similarity=min_similarity,
        min_good_matches=min_good_matches,
        min_inlier_ratio=min_inlier_ratio,
        appearance_penalty_weight=appearance_penalty_weight,
        db_max_top_k=db_max_top_k,
        ivfflat_probes=ivfflat_probes,
        modal_rerank_enabled=bool(modal_rerank_enabled),
        modal_rerank_top_n=max(1, int(modal_rerank_top_n)),
        modal_rerank_weight=round(float(modal_rerank_weight), 4),
        modal_rerank_workers=max(1, int(modal_rerank_workers)),
        modal_rerank_batch_size=max(1, int(modal_rerank_batch_size)),
        modal_rerank_max_retries=max(0, int(modal_rerank_max_retries)),
        modal_rerank_environment=modal_rerank_environment,
        modal_rerank_model_name=modal_rerank_model_name,
        modal_rerank_pretrained=modal_rerank_pretrained,
        modal_feature_verify_enabled=bool(modal_feature_verify_enabled),
        modal_feature_verify_top_n=max(1, int(modal_feature_verify_top_n)),
        modal_feature_verify_workers=max(1, int(modal_feature_verify_workers)),
        modal_feature_verify_batch_size=max(1, int(modal_feature_verify_batch_size)),
        modal_feature_verify_max_retries=max(0, int(modal_feature_verify_max_retries)),
        modal_feature_verify_max_keypoints=max(128, int(modal_feature_verify_max_keypoints)),
        modal_feature_verify_resize_long_edge=max(
            256, int(modal_feature_verify_resize_long_edge)
        ),
        modal_feature_verify_min_matches=max(4, int(modal_feature_verify_min_matches)),
    )

    if not embedders:
        raise RuntimeError("No retrieval embedders are configured")

    model_specs = []
    for idx, embedder in enumerate(embedders):
        model_id = str(getattr(embedder, "model_id", f"model_{idx}"))
        model_specs.append(
            {
                "model_id": model_id,
                "model_name": str(getattr(embedder, "model_name", model_id)),
                "model_version": str(getattr(embedder, "model_version", "")),
                "weight": max(
                    0.0,
                    float(
                        (model_weights or {}).get(
                            model_id,
                            getattr(embedder, "weight", 1.0),
                        )
                    ),
                ),
                "embedder": embedder,
            }
        )
    model_specs = [m for m in model_specs if float(m["weight"]) > 0.0]
    if not model_specs:
        raise RuntimeError("All retrieval model weights are zero")
    total_model_weight = max(1e-9, sum(float(m["weight"]) for m in model_specs))

    crops = _build_query_crops(image_bytes)
    timings_ms["crop_build"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    debug["crop_count"] = len(crops)

    merged: Dict[int, dict] = {}
    crop_result_counts: Dict[str, int] = {}
    model_crop_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    failed_models: List[dict] = []

    effective_db_max_top_k = max(int(top_k_per_crop), min(20000, int(db_max_top_k)))
    effective_ivfflat_probes = max(1, min(1000, int(ivfflat_probes)))

    for model in model_specs:
        embedder = model["embedder"]
        model_id = str(model["model_id"])
        model_weight = float(model["weight"])
        try:
            model_vectors = embedder.encode_image_bytes_batch([c[1] for c in crops])
        except Exception as exc:
            failed_models.append(
                {
                    "model_id": model_id,
                    "model_name": str(model["model_name"]),
                    "model_version": str(model["model_version"]),
                    "error": str(exc),
                }
            )
            reason_counts["model_encode_failed"] = reason_counts.get(
                "model_encode_failed", 0
            ) + 1
            log_retrieval_event(
                retrieval_id,
                "locator_model_skipped",
                model_id=model_id,
                model_name=str(model["model_name"]),
                model_version=str(model["model_version"]),
                error=str(exc),
            )
            continue
        if len(model_vectors) != len(crops):
            failed_models.append(
                {
                    "model_id": model_id,
                    "model_name": str(model["model_name"]),
                    "model_version": str(model["model_version"]),
                    "error": "batch-size-mismatch",
                }
            )
            reason_counts["model_encode_failed"] = reason_counts.get(
                "model_encode_failed", 0
            ) + 1
            continue
        for (crop_name, _), vector in zip(crops, model_vectors):
            t0 = time.perf_counter()
            rows = db.search_captures_by_embedding(
                vector,
                str(model["model_name"]),
                str(model["model_version"]),
                top_k=max(1, int(top_k_per_crop)),
                min_similarity=min_similarity,
                max_top_k=effective_db_max_top_k,
                trace_id=retrieval_id,
                ivfflat_probes=effective_ivfflat_probes,
            )
            crop_key = f"{model_id}:{crop_name}"
            crop_result_counts[crop_key] = len(rows)
            model_crop_counts[model_id][crop_name] = len(rows)
            crop_weight = _crop_weight(crop_name)
            for row in rows:
                cid = int(row["capture_id"])
                sim = float(row.get("similarity", 0.0))
                weighted_sim = sim * crop_weight * model_weight
                existing = merged.get(cid)
                if not existing:
                    merged[cid] = {
                        **row,
                        "best_similarity": sim,
                        "sum_similarity": sim,
                        "crop_hits": {crop_name},
                        "crop_count": 1,
                        "model_hits": {model_id},
                        "model_best_similarity": {model_id: sim},
                        "model_weighted_similarity": {model_id: weighted_sim},
                        "vector_score": weighted_sim / total_model_weight,
                        "appearance_penalty": 0.0,
                        "geom_score": 0.0,
                        "score": weighted_sim / total_model_weight,
                        "good_matches": 0.0,
                        "inliers": 0.0,
                        "inlier_ratio": 0.0,
                    }
                else:
                    existing["best_similarity"] = max(float(existing["best_similarity"]), sim)
                    existing["sum_similarity"] = float(existing["sum_similarity"]) + sim
                    existing["crop_hits"].add(crop_name)
                    existing["crop_count"] = len(existing["crop_hits"])
                    existing.setdefault("model_hits", set()).add(model_id)
                    model_best = existing.setdefault("model_best_similarity", {})
                    model_best[model_id] = max(float(model_best.get(model_id, 0.0)), sim)
                    model_weighted = existing.setdefault("model_weighted_similarity", {})
                    model_weighted[model_id] = max(
                        float(model_weighted.get(model_id, 0.0)),
                        weighted_sim,
                    )
                    weighted_sum = sum(float(v) for v in model_weighted.values())
                    crop_bonus = min(0.12, 0.02 * max(0, int(existing["crop_count"]) - 1))
                    existing["vector_score"] = (weighted_sum / total_model_weight) + crop_bonus
                    if sim > float(existing.get("similarity", 0.0)):
                        existing["similarity"] = sim
            timings_ms[f"embed_search_{crop_key}"] = round(
                (time.perf_counter() - t0) * 1000.0, 2
            )
    if failed_models:
        debug["failed_models"] = failed_models
        flags.append("model_encode_failed")

    candidates = list(merged.values())
    candidates.sort(key=lambda x: float(x.get("vector_score", 0.0)), reverse=True)
    if len(candidates) > int(max_merged_candidates):
        candidates = candidates[: int(max_merged_candidates)]
        reason_counts["trimmed_after_merge"] = len(merged) - len(candidates)
    debug["crop_result_counts"] = crop_result_counts
    debug["model_crop_result_counts"] = {
        model_id: dict(crop_counts) for model_id, crop_counts in model_crop_counts.items()
    }
    debug["merged_unique_candidates"] = len(merged)
    debug["merged_kept_candidates"] = len(candidates)
    timings_ms["retrieve_merge"] = round((time.perf_counter() - stage_start) * 1000.0, 2)

    if not candidates:
        flags.append("no_candidates")
        log_retrieval_event(retrieval_id, "locator_no_candidates")
        return {
            "retrieval_id": retrieval_id,
            "models": [
                {
                    "model_id": m["model_id"],
                    "model_name": m["model_name"],
                    "model_version": m["model_version"],
                    "weight": m["weight"],
                }
                for m in model_specs
            ],
            "best_estimate": None,
            "clusters": [],
            "supporting_matches": [],
            "flags": flags,
            "reason_counts": reason_counts,
            "timings_ms": timings_ms,
            "debug": debug if include_debug else {},
        }

    by_pano: Dict[int, List[dict]] = {}
    for row in candidates:
        by_pano.setdefault(int(row["panorama_id"]), []).append(row)
    pruned: List[dict] = []
    pano_rows_dropped = 0
    for rows in by_pano.values():
        rows.sort(key=lambda x: float(x.get("vector_score", 0.0)), reverse=True)
        keep = rows[: max(1, int(panorama_vote_cap))]
        pruned.extend(keep)
        pano_rows_dropped += max(0, len(rows) - len(keep))
    pruned.sort(key=lambda x: float(x.get("vector_score", 0.0)), reverse=True)
    debug["panorama_vote_cap_dropped"] = pano_rows_dropped
    debug["post_vote_cap_candidates"] = len(pruned)

    modal_rerank_info: Dict[str, Any] = {"enabled": bool(modal_rerank_enabled)}
    if modal_rerank_enabled and pruned:
        t_modal = time.perf_counter()
        try:
            modal_rerank_info = _apply_modal_rerank(
                retrieval_id=retrieval_id,
                query_image_bytes=image_bytes,
                candidates=pruned,
                capture_abs_path=capture_abs_path,
                rerank_top_n=max(1, int(modal_rerank_top_n)),
                rerank_weight=max(0.0, min(0.95, float(modal_rerank_weight))),
                rerank_workers=max(1, int(modal_rerank_workers)),
                rerank_batch_size=max(1, int(modal_rerank_batch_size)),
                rerank_max_retries=max(0, int(modal_rerank_max_retries)),
                rerank_environment=str(modal_rerank_environment),
                rerank_model_name=str(modal_rerank_model_name),
                rerank_pretrained=str(modal_rerank_pretrained),
            )
            if int(modal_rerank_info.get("applied", 0)) > 0:
                flags.append("modal_rerank_applied")
            else:
                reason_counts["modal_rerank_no_scores"] = reason_counts.get(
                    "modal_rerank_no_scores", 0
                ) + 1
        except Exception as exc:
            modal_rerank_info = {"enabled": True, "error": str(exc)}
            reason_counts["modal_rerank_failed"] = reason_counts.get(
                "modal_rerank_failed", 0
            ) + 1
            flags.append("modal_rerank_failed")
            log_retrieval_event(
                retrieval_id,
                "locator_modal_rerank_failed",
                error=str(exc),
                require_success=bool(modal_rerank_require_success),
            )
            if modal_rerank_require_success:
                raise
        timings_ms["modal_rerank"] = round((time.perf_counter() - t_modal) * 1000.0, 2)
    debug["modal_rerank"] = modal_rerank_info

    clusters = _cluster_candidates(pruned, float(cluster_radius_m))
    debug["cluster_count"] = len(clusters)
    timings_ms["cluster"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    if not clusters:
        flags.append("no_clusters")
        return {
            "retrieval_id": retrieval_id,
            "models": [
                {
                    "model_id": m["model_id"],
                    "model_name": m["model_name"],
                    "model_version": m["model_version"],
                    "weight": m["weight"],
                }
                for m in model_specs
            ],
            "best_estimate": None,
            "clusters": [],
            "supporting_matches": [],
            "flags": flags,
            "reason_counts": reason_counts,
            "timings_ms": timings_ms,
            "debug": debug if include_debug else {},
        }

    query_mean_rgb = _mean_rgb(image_bytes)
    geom_enabled = cv2 is not None and np is not None
    if not geom_enabled:
        flags.append("geom_unavailable")
    query_gray = None
    query_kp = None
    query_desc = None
    if geom_enabled:
        query_gray = _decode_gray_image(image_bytes)
        if query_gray is None:
            geom_enabled = False
            flags.append("geom_query_decode_failed")
        else:
            orb = cv2.ORB_create(nfeatures=1800)
            query_kp, query_desc = orb.detectAndCompute(query_gray, None)
            if query_desc is None:
                geom_enabled = False
                flags.append("low_feature_support")

    verify_pool: List[dict] = []
    for cluster in clusters[:3]:
        verify_pool.extend(cluster["items"][: max(1, int(verify_top_n))])
    seen_capture_ids = set()
    verify_candidates: List[dict] = []
    for item in verify_pool:
        cid = int(item["capture_id"])
        if cid in seen_capture_ids:
            continue
        seen_capture_ids.add(cid)
        verify_candidates.append(item)
        if len(verify_candidates) >= int(verify_top_n):
            break

    modal_feature_verify_info: Dict[str, Any] = {"enabled": bool(modal_feature_verify_enabled)}
    modal_feature_geom_applied = 0
    if modal_feature_verify_enabled and verify_candidates:
        t_modal_verify = time.perf_counter()
        try:
            modal_feature_verify_info = _apply_modal_feature_verify(
                retrieval_id=retrieval_id,
                query_image_bytes=image_bytes,
                verify_candidates=verify_candidates,
                capture_abs_path=capture_abs_path,
                query_mean_rgb=query_mean_rgb,
                verify_top_n=max(1, int(modal_feature_verify_top_n)),
                verify_workers=max(1, int(modal_feature_verify_workers)),
                verify_batch_size=max(1, int(modal_feature_verify_batch_size)),
                verify_max_retries=max(0, int(modal_feature_verify_max_retries)),
                verify_environment=str(modal_rerank_environment),
                max_keypoints=max(128, int(modal_feature_verify_max_keypoints)),
                resize_long_edge=max(256, int(modal_feature_verify_resize_long_edge)),
                min_matches=max(4, int(modal_feature_verify_min_matches)),
            )
            modal_feature_geom_applied = int(modal_feature_verify_info.get("applied", 0))
            if modal_feature_geom_applied > 0:
                flags.append("modal_feature_verify_applied")
            else:
                reason_counts["modal_feature_verify_no_scores"] = reason_counts.get(
                    "modal_feature_verify_no_scores", 0
                ) + 1
        except Exception as exc:
            modal_feature_verify_info = {"enabled": True, "error": str(exc)}
            reason_counts["modal_feature_verify_failed"] = reason_counts.get(
                "modal_feature_verify_failed", 0
            ) + 1
            flags.append("modal_feature_verify_failed")
            log_retrieval_event(
                retrieval_id,
                "locator_modal_feature_verify_failed",
                error=str(exc),
                require_success=bool(modal_feature_verify_require_success),
            )
            if modal_feature_verify_require_success:
                raise
        timings_ms["modal_feature_verify"] = round(
            (time.perf_counter() - t_modal_verify) * 1000.0, 2
        )
    debug["modal_feature_verify"] = modal_feature_verify_info

    geom_success = modal_feature_geom_applied
    geom_attempts = 0
    t_verify = time.perf_counter()
    if geom_enabled:
        for item in verify_candidates:
            if (
                str(item.get("geom_source", "")).lower() == "lightglue"
                and float(item.get("geom_score", 0.0)) > 0.0
            ):
                reason_counts["verify_skipped_modal_scored"] = reason_counts.get(
                    "verify_skipped_modal_scored", 0
                ) + 1
                continue
            abs_path = capture_abs_path(str(item.get("filepath", "")))
            if not abs_path or not os.path.exists(abs_path):
                reason_counts["verify_missing_file"] = reason_counts.get(
                    "verify_missing_file", 0
                ) + 1
                continue
            try:
                with open(abs_path, "rb") as f:
                    cand_bytes = f.read()
                cand_gray = _decode_gray_image(cand_bytes)
                if cand_gray is None:
                    reason_counts["verify_decode_failed"] = reason_counts.get(
                        "verify_decode_failed", 0
                    ) + 1
                    continue
                geom_attempts += 1
                geom = _compute_orb_score(
                    query_kp,
                    query_desc,
                    cand_gray,
                    min_good_matches=min_good_matches,
                )
                item["geom_score"] = max(
                    float(item.get("geom_score", 0.0)),
                    float(geom["geom_score"]),
                )
                item["good_matches"] = max(
                    float(item.get("good_matches", 0.0)),
                    float(geom["good_matches"]),
                )
                item["inliers"] = max(
                    float(item.get("inliers", 0.0)),
                    float(geom["inliers"]),
                )
                item["inlier_ratio"] = max(
                    float(item.get("inlier_ratio", 0.0)),
                    float(geom["inlier_ratio"]),
                )
                item["appearance_penalty"] = _appearance_penalty(
                    query_mean_rgb, cand_bytes
                )
                if item["geom_score"] > 0:
                    geom_success += 1
            except Exception:
                reason_counts["verify_error"] = reason_counts.get("verify_error", 0) + 1
    else:
        reason_counts["verify_skipped"] = len(verify_candidates)
    timings_ms["verify"] = round((time.perf_counter() - t_verify) * 1000.0, 2)
    debug["geom_attempts"] = geom_attempts
    debug["geom_success"] = geom_success

    if geom_attempts > 0 and geom_success == 0 and modal_feature_geom_applied <= 0:
        flags.append("low_feature_support")

    verify_set = {int(item["capture_id"]): item for item in verify_candidates}
    for item in pruned:
        verified_item = verify_set.get(int(item["capture_id"]))
        if verified_item:
            item.update(verified_item)
        vector_score = float(item.get("vector_score", 0.0))
        geom_score = float(item.get("geom_score", 0.0))
        good_matches = float(item.get("good_matches", 0.0))
        inlier_ratio = float(item.get("inlier_ratio", 0.0))
        appearance_penalty = float(item.get("appearance_penalty", 0.0))
        if verified_item and (
            good_matches < float(min_good_matches)
            or inlier_ratio < float(min_inlier_ratio)
        ):
            reason_counts["verify_gated_low_geom"] = reason_counts.get(
                "verify_gated_low_geom", 0
            ) + 1
            item["score"] = vector_score * 0.25
            item["geom_gated"] = True
        else:
            has_geom = geom_score > 0
            geom_weight = 0.4 if has_geom else 0.0
            base_score = vector_score * (1.0 - geom_weight) + geom_score * geom_weight
            penalty = max(
                0.0,
                min(0.9, float(appearance_penalty_weight) * max(0.0, appearance_penalty)),
            )
            item["score"] = max(0.0, base_score * (1.0 - penalty))
            item["geom_gated"] = False

    rescored_clusters = _cluster_candidates(pruned, float(cluster_radius_m))
    best_cluster = rescored_clusters[0]
    best_items = sorted(
        best_cluster["items"], key=lambda x: float(x.get("score", 0.0)), reverse=True
    )
    support = best_items[:20]
    centroid_lat, centroid_lon = _weighted_centroid(best_items, "score")
    distances = [
        _haversine_m(float(item["lat"]), float(item["lon"]), centroid_lat, centroid_lon)
        for item in support
    ]
    radius_m = max(20.0, max(distances) if distances else 20.0)
    top_item = support[0] if support else None
    second_score = float(support[1].get("score", 0.0)) if len(support) > 1 else 0.0
    if top_item:
        top_score = float(top_item.get("score", 0.0))
        top_geom = float(top_item.get("geom_score", 0.0))
        score_gap = top_score - second_score
        if (top_geom >= 0.22 and top_score >= 0.45) or (
            top_score >= 0.78 and score_gap >= 0.08
        ):
            centroid_lat = float(top_item["lat"])
            centroid_lon = float(top_item["lon"])
            radius_m = max(12.0, min(radius_m, 30.0))
            flags.append("exact_match_anchor")
            debug["exact_match_anchor"] = {
                "capture_id": int(top_item["capture_id"]),
                "top_score": round(top_score, 4),
                "top_geom": round(top_geom, 4),
                "score_gap": round(score_gap, 4),
            }
    best_score = float(support[0].get("score", 0.0)) if support else 0.0
    support_factor = min(1.0, len(support) / 8.0)
    compactness = max(0.0, 1.0 - min(1.0, radius_m / 220.0))
    score_norm = max(0.0, min(1.0, (best_score - 0.2) / 0.6))
    confidence = (0.5 * score_norm) + (0.3 * compactness) + (0.2 * support_factor)
    if "low_feature_support" in flags:
        confidence *= 0.85
    if len(rescored_clusters) < 2:
        flags.append("sparse_coverage")
        confidence *= 0.9

    best_estimate = {
        "lat": round(float(centroid_lat), 7),
        "lon": round(float(centroid_lon), 7),
        "confidence": round(float(max(0.0, min(1.0, confidence))), 4),
        "radius_m": round(float(radius_m), 2),
        "cluster_id": int(best_cluster["cluster_id"]),
        "cluster_size": int(best_cluster["size"]),
    }

    for item in support:
        item["crop_hits"] = sorted(list(item.get("crop_hits", [])))
        item["model_hits"] = sorted(list(item.get("model_hits", [])))
        item["model_best_similarity"] = {
            str(k): float(v) for k, v in (item.get("model_best_similarity") or {}).items()
        }
        item["similarity"] = float(item.get("similarity", 0.0))
        item["vector_score"] = float(item.get("vector_score", 0.0))
        item["geom_score"] = float(item.get("geom_score", 0.0))
        item["appearance_penalty"] = float(item.get("appearance_penalty", 0.0))
        item["score"] = float(item.get("score", 0.0))
        item["lat"] = float(item.get("lat", 0.0))
        item["lon"] = float(item.get("lon", 0.0))

    cluster_payload = [
        {
            "cluster_id": int(c["cluster_id"]),
            "lat": round(float(c["lat"]), 7),
            "lon": round(float(c["lon"]), 7),
            "score": round(float(c["score"]), 6),
            "size": int(c["size"]),
            "radius_m": round(float(c["radius_m"]), 2),
        }
        for c in rescored_clusters[:8]
    ]

    timings_ms["total"] = round((time.perf_counter() - stage_start) * 1000.0, 2)
    log_retrieval_event(
        retrieval_id,
        "locator_completed",
        flags=flags,
        clusters=len(rescored_clusters),
        geom_attempts=geom_attempts,
        geom_success=geom_success,
        confidence=best_estimate["confidence"],
        radius_m=best_estimate["radius_m"],
        timings_ms=timings_ms,
    )

    return {
        "retrieval_id": retrieval_id,
        "models": [
            {
                "model_id": m["model_id"],
                "model_name": m["model_name"],
                "model_version": m["model_version"],
                "weight": m["weight"],
            }
            for m in model_specs
        ],
        "best_estimate": best_estimate,
        "clusters": cluster_payload,
        "supporting_matches": support,
        "flags": sorted(set(flags)),
        "reason_counts": reason_counts,
        "timings_ms": timings_ms,
        "debug": debug if include_debug else {},
    }
