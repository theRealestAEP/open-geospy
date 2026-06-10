"""Pure scoring/aggregation logic for the locate pipeline.

Extracted verbatim from backend/app/api/retrieval.py so the
correctness-critical math is unit-testable without FastAPI, the
vector store, or embedding models.
"""

import math
from typing import Dict, List, Sequence


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def merge_capture_row(
    merged: Dict[int, dict],
    row: dict,
    *,
    model_id: str,
    model_weight: float,
    embedding_base: str,
) -> None:
    """Merge one vector-search result row into the per-capture accumulator.

    Scores from multiple embedding models are summed (similarity x weight);
    the max per-model similarity is kept on the entry.
    """
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


def aggregate_panorama_candidates(rows: Sequence[dict]) -> List[dict]:
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
        top_score_mean = (
            sum(top_capture_scores) / len(top_capture_scores)
            if top_capture_scores
            else 0.0
        )
        score_consistency = (
            max(0.0, min(1.0, top_score_mean / best_capture_score))
            if best_capture_score > 0.0
            else 0.0
        )
        support_bonus = min(
            0.05,
            (0.010 * math.log1p(max(0, capture_count - 1)))
            + (0.006 * math.log1p(max(0, heading_support - 1)))
            + (0.004 * math.log1p(max(0, pitch_support - 1)))
            + (0.010 * math.log1p(max(0, model_support - 1)))
            + (
                0.012
                * score_consistency
                * math.log1p(max(0, len(top_capture_scores) - 1))
            ),
        )
        # Keep the best capture dominant, but let repeated model/view support
        # break ties instead of collecting decorative metadata.
        panorama_score = best_capture_score + support_bonus
        ranked.append(
            {
                **best_capture,
                "panorama_score": round(panorama_score, 6),
                "best_capture_score": round(best_capture_score, 6),
                "panorama_support_bonus": round(support_bonus, 6),
                "capture_hits": capture_count,
                "heading_support": heading_support,
                "pitch_support": pitch_support,
                "model_support": model_support,
                "model_hits": sorted(str(item) for item in entry["model_hits"]),
                "top_capture_scores": [round(float(v), 6) for v in top_capture_scores],
            }
        )
    ranked.sort(key=lambda r: float(r.get("panorama_score", 0.0)), reverse=True)
    return ranked


def cluster_panorama_families(
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
            distance_m = haversine_m(lat, lon, family_lat, family_lon)
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
        family_model_support = len(family["model_hits"])
        family_support_bonus = min(
            0.035,
            (0.008 * math.log1p(max(0, len(panoramas_in_family) - 1)))
            + (0.004 * math.log1p(max(0, int(family["capture_hits"]) - 1)))
            + (0.006 * math.log1p(max(0, int(family["heading_support"]) - 1)))
            + (0.008 * math.log1p(max(0, family_model_support - 1))),
        )
        family_score = (
            float(best_panorama.get("panorama_score", 0.0))
            + family_support_bonus
        )
        ranked_families.append(
            {
                **best_panorama,
                "family_id": family["family_id"],
                "family_score": round(family_score, 6),
                "family_support_bonus": round(family_support_bonus, 6),
                "family_center_lat": round(float(family["center_lat"]), 7),
                "family_center_lon": round(float(family["center_lon"]), 7),
                "family_radius_meters": float(family_radius_meters),
                "family_panorama_count": len(panoramas_in_family),
                "family_capture_hits": int(family["capture_hits"]),
                "family_model_support": family_model_support,
                "family_panorama_ids": [
                    int(item.get("panorama_id", 0)) for item in panoramas_in_family[:10]
                ],
                "family_weight_sum": round(float(family["weight_sum"]), 6),
            }
        )
    ranked_families.sort(
        key=lambda row: float(row.get("family_score", 0.0)), reverse=True
    )
    return ranked_families
