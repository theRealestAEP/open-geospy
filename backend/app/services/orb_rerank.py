import base64
import copy
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from backend.app.services.runtime import env_bool, env_float, env_int
from backend.app.services.scene_masking import build_sam2_scene_mask, merge_sam2_mask_stats

LOCATE_ORB_ENABLED_DEFAULT = env_bool("GEOSPY_LOCATE_ORB_ENABLED", False)
LOCATE_ORB_TOP_N_DEFAULT = env_int("GEOSPY_LOCATE_ORB_TOP_N", 100, minimum=1, maximum=5000)
LOCATE_ORB_WEIGHT_DEFAULT = env_float("GEOSPY_LOCATE_ORB_WEIGHT", 0.75, minimum=0.0, maximum=5.0)
LOCATE_ORB_FEATURES = env_int("GEOSPY_LOCATE_ORB_FEATURES", 500, minimum=100, maximum=2000)
LOCATE_ORB_RANSAC_TOP_K_DEFAULT = env_int("GEOSPY_LOCATE_ORB_RANSAC_TOP_K", 10, minimum=0, maximum=500)
LOCATE_ORB_RATIO_TEST = env_float("GEOSPY_LOCATE_ORB_RATIO_TEST", 0.75, minimum=0.5, maximum=0.95)
LOCATE_ORB_RANSAC_REPROJ_THRESHOLD = env_float(
    "GEOSPY_LOCATE_ORB_RANSAC_REPROJ_THRESHOLD",
    5.0,
    minimum=1.0,
    maximum=25.0,
)
LOCATE_ORB_VISUALIZATION_LIMIT = env_int(
    "GEOSPY_LOCATE_ORB_VISUALIZATION_LIMIT",
    6,
    minimum=1,
    maximum=12,
)
LOCATE_ORB_VISUALIZATION_MATCH_LIMIT = env_int(
    "GEOSPY_LOCATE_ORB_VISUALIZATION_MATCH_LIMIT",
    28,
    minimum=8,
    maximum=80,
)
LOCATE_ORB_IGNORE_BOTTOM_RATIO_DEFAULT = env_float(
    "GEOSPY_LOCATE_ORB_IGNORE_BOTTOM_RATIO",
    0.28,
    minimum=0.0,
    maximum=0.6,
)


@dataclass(frozen=True)
class OrbRerankConfig:
    enabled: bool
    top_n: int
    feature_count: int
    weight: float
    ransac_top_k: int
    visualization_limit: int
    ignore_bottom_ratio: float
    sam2_mask_cars: bool
    sam2_mask_trees: bool


def normalize_orb_ignore_bottom_ratio(value: Optional[float]) -> float:
    if value is None:
        return float(LOCATE_ORB_IGNORE_BOTTOM_RATIO_DEFAULT)
    return max(0.0, min(0.6, float(value)))


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


def _build_orb_feature_mask(
    np,
    image_shape: Sequence[int],
    *,
    ignore_bottom_ratio: float,
    excluded_mask=None,
):
    height = int(image_shape[0]) if image_shape else 0
    width = int(image_shape[1]) if len(image_shape) > 1 else 0
    if height <= 0 or width <= 0:
        return None, 0
    mask = np.full((height, width), 255, dtype=np.uint8)
    changed = False
    ignored_bottom_pixels = 0
    if ignore_bottom_ratio > 0.0:
        ignored_bottom_pixels = min(
            max(0, int(round(height * float(ignore_bottom_ratio)))),
            max(0, height - 8),
        )
        if ignored_bottom_pixels > 0:
            mask[height - ignored_bottom_pixels :, :] = 0
            changed = True
    if excluded_mask is not None:
        excluded = np.asarray(excluded_mask, dtype=bool)
        if excluded.shape[:2] == (height, width) and excluded.any():
            mask[excluded] = 0
            changed = True
    if not changed:
        return None, 0
    return mask, ignored_bottom_pixels


def _annotate_orb_focus_mask(
    cv2,
    image,
    *,
    ignored_bottom_pixels: int = 0,
    excluded_mask=None,
):
    annotated = image.copy()
    if excluded_mask is not None:
        overlay_mask = excluded_mask.astype(bool)
        if overlay_mask.shape[:2] == annotated.shape[:2] and overlay_mask.any():
            overlay = annotated.copy()
            overlay[overlay_mask] = (56, 72, 214)
            annotated = cv2.addWeighted(overlay, 0.42, annotated, 0.58, 0.0)
            contours, _ = cv2.findContours(
                overlay_mask.astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                cv2.drawContours(annotated, contours, -1, (130, 170, 255), 2)
    if ignored_bottom_pixels > 0:
        height, width = annotated.shape[:2]
        cutoff = max(0, height - int(ignored_bottom_pixels))
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (0, cutoff),
            (width, height),
            (18, 30, 64),
            thickness=-1,
        )
        annotated = cv2.addWeighted(overlay, 0.38, annotated, 0.62, 0.0)
        cv2.line(annotated, (0, cutoff), (width, cutoff), (255, 209, 102), thickness=2)
    return annotated


def rerank_capture_rows_with_orb(
    rows: Sequence[dict],
    *,
    image_bytes: bytes,
    capture_abs_path: Callable[[str], str],
    config: OrbRerankConfig,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "enabled": bool(config.enabled),
        "status": "skipped",
        "reason": "disabled",
        "candidate_count": len(rows),
        "top_n": 0,
        "feature_count": int(config.feature_count),
        "weight": float(config.weight),
        "ratio_test": float(LOCATE_ORB_RATIO_TEST),
        "ransac_top_k": int(config.ransac_top_k),
        "query_keypoints": 0,
        "candidates_extracted": 0,
        "candidates_scored": 0,
        "ransac_checked": 0,
        "missing_files": 0,
        "decode_errors": 0,
        "visualization_limit": int(max(1, config.visualization_limit)),
        "ignore_bottom_ratio": float(config.ignore_bottom_ratio),
        "sam2_enabled": bool(config.sam2_mask_cars or config.sam2_mask_trees),
        "sam2_mask_cars": bool(config.sam2_mask_cars),
        "sam2_mask_trees": bool(config.sam2_mask_trees),
        "sam2_vehicle_boxes": 0,
        "sam2_tree_boxes": 0,
        "sam2_masked_pixels": 0,
        "sam2_candidate_images_masked": 0,
        "sam2_candidate_mask_limit": 0,
        "sam2_model_id": "",
        "sam2_device": "",
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
    if not config.enabled:
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
    sam2_query_mask = None
    sam2_enabled = bool(config.sam2_mask_cars or config.sam2_mask_trees)
    if sam2_enabled:
        sam2_query_mask, sam2_stats = build_sam2_scene_mask(
            image_bytes=image_bytes,
            mask_cars=config.sam2_mask_cars,
            mask_trees=config.sam2_mask_trees,
        )
        merge_sam2_mask_stats(stats, sam2_stats)
    query_mask, query_ignored_bottom_pixels = _build_orb_feature_mask(
        np,
        query_image.shape,
        ignore_bottom_ratio=config.ignore_bottom_ratio,
        excluded_mask=sam2_query_mask,
    )

    orb = cv2.ORB_create(nfeatures=max(100, int(config.feature_count)))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    t_query = time.perf_counter()
    query_keypoints, query_descriptors = orb.detectAndCompute(query_image, query_mask)
    stats["timings_ms"]["query_extract"] = round(
        (time.perf_counter() - t_query) * 1000.0, 2
    )
    stats["query_keypoints"] = len(query_keypoints or [])
    limit = min(len(reranked), max(1, int(config.top_n)))
    stats["top_n"] = limit
    comparison_limit = min(limit, max(1, int(config.visualization_limit)))
    query_visual = _annotate_orb_focus_mask(
        cv2,
        query_color,
        ignored_bottom_pixels=query_ignored_bottom_pixels,
        excluded_mask=sam2_query_mask,
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
    if progress_callback:
        progress_callback(
            {
                "query_keypoints": int(stats["query_keypoints"]),
                "query_fingerprint_data_url": str(
                    stats["query_fingerprint_data_url"] or ""
                ),
                "processed_candidates": 0,
                "candidate_count": int(limit),
            }
        )
    if not query_keypoints or query_descriptors is None:
        stats["reason"] = "query-has-no-features"
        return reranked, stats

    candidate_read_ms = 0.0
    candidate_extract_ms = 0.0
    match_ms = 0.0
    ransac_ms = 0.0
    comparisons_by_capture_id: Dict[int, dict] = {}
    sam2_candidate_mask_limit = (
        min(limit, max(comparison_limit, int(config.ransac_top_k), 1))
        if sam2_enabled
        else 0
    )
    stats["sam2_candidate_mask_limit"] = int(sam2_candidate_mask_limit)
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
        candidate_sam2_mask = None
        if sam2_enabled and idx < sam2_candidate_mask_limit:
            candidate_sam2_mask, candidate_sam2_stats = build_sam2_scene_mask(
                image_bgr=candidate_color,
                mask_cars=config.sam2_mask_cars,
                mask_trees=config.sam2_mask_trees,
            )
            merge_sam2_mask_stats(stats, candidate_sam2_stats, candidate_image=True)
        candidate_mask, candidate_ignored_bottom_pixels = _build_orb_feature_mask(
            np,
            candidate_image.shape,
            ignore_bottom_ratio=config.ignore_bottom_ratio,
            excluded_mask=candidate_sam2_mask,
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

        denom = float(max(1, min(len(query_keypoints), len(candidate_keypoints))))
        good_match_count = len(good_matches)
        good_ratio = float(good_match_count) / denom
        mean_match_distance = (
            float(
                sum(float(m.distance) for m in good_matches[:8])
                / max(1, min(8, good_match_count))
            )
            if good_match_count > 0
            else 0.0
        )
        inlier_count = 0
        inlier_ratio = 0.0
        if (
            config.ransac_top_k > 0
            and idx < int(config.ransac_top_k)
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
        distance_quality = (
            max(0.0, min(1.0, 1.0 - (mean_match_distance / 64.0)))
            if good_match_count > 1
            else 0.0
        )
        # ORB is a tie-breaker over vector similarity, not a replacement. These
        # weights keep feature support bounded while still rewarding geometry.
        match_count_bonus = 0.028 * math.sqrt(max(0, good_match_count - 1))
        inlier_bonus = 0.05 * math.sqrt(max(0, inlier_count))
        ratio_bonus = 0.30 * max(good_ratio, inlier_ratio)
        distance_bonus = 0.015 * distance_quality
        orb_score = min(
            0.35,
            match_count_bonus + inlier_bonus + ratio_bonus + distance_bonus,
        )
        row["orb_score"] = round(orb_score, 6)
        row["orb_good_matches"] = int(good_match_count)
        row["orb_inliers"] = int(inlier_count)
        row["orb_mean_match_distance"] = round(mean_match_distance, 4)
        row["orb_reranked"] = True
        row["score"] = score_before + (float(config.weight) * orb_score)
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
                    excluded_mask=candidate_sam2_mask,
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
                "orb_mean_match_distance": round(mean_match_distance, 4),
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
            if progress_callback:
                progress_callback(
                    {
                        "processed_candidates": int(idx + 1),
                        "candidates_scored": int(stats["candidates_scored"]),
                        "latest_comparison": copy.deepcopy(
                            comparisons_by_capture_id[capture_id]
                        ),
                    }
                )
        elif progress_callback:
            progress_callback(
                {
                    "processed_candidates": int(idx + 1),
                    "candidates_scored": int(stats["candidates_scored"]),
                }
            )

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
                "orb_mean_match_distance": round(
                    float(
                        row.get("orb_mean_match_distance")
                        or comparison.get("orb_mean_match_distance")
                        or 0.0
                    ),
                    4,
                ),
                "visual_match_count": int(comparison.get("visual_match_count") or 0),
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
