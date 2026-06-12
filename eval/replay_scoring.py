"""Replay captured locate candidate pools through scoring variants offline.

Loads raw_candidates.jsonl written by eval.run_locator (one line per case with
the full vector candidate pool) and scores each case with the production
aggregation plus experimental variants, so family-scoring changes can be
compared across all cases in seconds without re-running the backend.

Run with:
    python -m eval.replay_scoring eval/results/<run>/raw_candidates.jsonl
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from backend.app.services.locate_scoring import (
    aggregate_panorama_candidates,
    cluster_panorama_families,
    haversine_m,
)

FAMILY_RADIUS_METERS = 75.0


def _load_pools(path: str) -> List[dict]:
    pools = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pools.append(json.loads(line))
    return pools


def predict_production(candidates: Sequence[dict]) -> Optional[Tuple[float, float]]:
    """Mirror the live pipeline: panorama aggregation then family clustering."""
    rows = [
        {
            "capture_id": c.get("capture_id"),
            "panorama_id": c.get("panorama_id"),
            "lat": c.get("lat"),
            "lon": c.get("lon"),
            "score": float(c.get("score") or 0.0),
            "similarity": float(c.get("score") or 0.0),
            "model_hits": ["replay"],
        }
        for c in candidates
        if c.get("lat") is not None and c.get("lon") is not None
    ]
    if not rows:
        return None
    panoramas = aggregate_panorama_candidates(rows)
    families = cluster_panorama_families(
        panoramas, family_radius_meters=FAMILY_RADIUS_METERS
    )
    if not families:
        return None
    top = families[0]
    return float(top.get("lat", 0.0)), float(top.get("lon", 0.0))


def predict_density_vote(
    candidates: Sequence[dict],
    *,
    radius_m: float = FAMILY_RADIUS_METERS,
    score_power: float = 4.0,
) -> Optional[Tuple[float, float]]:
    """Each candidate votes score^power at its location; a family's vote is the
    sum over member candidates, so spatial consensus can outrank one hot hit."""
    points = [
        (float(c["lat"]), float(c["lon"]), float(c.get("score") or 0.0))
        for c in candidates
        if c.get("lat") is not None and c.get("lon") is not None
    ]
    if not points:
        return None
    clusters: List[dict] = []
    for lat, lon, score in points:
        vote = max(0.0, score) ** score_power
        target = None
        best_d = None
        for cl in clusters:
            d = haversine_m(lat, lon, cl["lat"], cl["lon"])
            if d <= radius_m and (best_d is None or d < best_d):
                best_d = d
                target = cl
        if target is None:
            clusters.append({"lat": lat, "lon": lon, "vote": vote, "best": score})
        else:
            total = target["vote"] + vote
            if total > 0:
                target["lat"] = (target["lat"] * target["vote"] + lat * vote) / total
                target["lon"] = (target["lon"] * target["vote"] + lon * vote) / total
            target["vote"] = total
            target["best"] = max(target["best"], score)
    top = max(clusters, key=lambda cl: cl["vote"])
    return top["lat"], top["lon"]


def predict_topk_sum(
    candidates: Sequence[dict],
    *,
    radius_m: float = FAMILY_RADIUS_METERS,
    k: int = 3,
) -> Optional[Tuple[float, float]]:
    """Family score = sum of its top-k member scores (caps runaway support
    from dozens of weak hits while still rewarding multi-capture agreement)."""
    points = [
        (float(c["lat"]), float(c["lon"]), float(c.get("score") or 0.0))
        for c in candidates
        if c.get("lat") is not None and c.get("lon") is not None
    ]
    if not points:
        return None
    clusters: List[dict] = []
    for lat, lon, score in points:
        target = None
        best_d = None
        for cl in clusters:
            d = haversine_m(lat, lon, cl["lat"], cl["lon"])
            if d <= radius_m and (best_d is None or d < best_d):
                best_d = d
                target = cl
        if target is None:
            clusters.append({"lat": lat, "lon": lon, "scores": [score]})
        else:
            target["scores"].append(score)
    for cl in clusters:
        cl["family_score"] = sum(sorted(cl["scores"], reverse=True)[:k])
    top = max(clusters, key=lambda cl: cl["family_score"])
    return top["lat"], top["lon"]


VARIANTS: Dict[str, Callable[[Sequence[dict]], Optional[Tuple[float, float]]]] = {
    "production": predict_production,
    "density-p2": lambda c: predict_density_vote(c, score_power=2.0),
    "density-p4": lambda c: predict_density_vote(c, score_power=4.0),
    "density-p8": lambda c: predict_density_vote(c, score_power=8.0),
    "topk2-sum": lambda c: predict_topk_sum(c, k=2),
    "topk3-sum": lambda c: predict_topk_sum(c, k=3),
    "topk5-sum": lambda c: predict_topk_sum(c, k=5),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pools", help="Path to raw_candidates.jsonl")
    parser.add_argument("--pool-depth", type=int, default=0,
                        help="Optional cap on candidates per case (0 = all).")
    args = parser.parse_args()

    pools = _load_pools(args.pools)
    print(f"cases: {len(pools)}")
    results: Dict[str, List[Optional[float]]] = defaultdict(list)
    for pool in pools:
        expected_lat = pool.get("expected_lat")
        expected_lon = pool.get("expected_lon")
        if expected_lat is None or expected_lon is None:
            continue
        candidates = pool.get("candidates") or []
        if args.pool_depth > 0:
            candidates = candidates[: args.pool_depth]
        for name, fn in VARIANTS.items():
            pred = fn(candidates)
            if pred is None:
                results[name].append(None)
            else:
                results[name].append(
                    haversine_m(
                        float(expected_lat), float(expected_lon), pred[0], pred[1]
                    )
                )

    print(f"{'variant':<14} {'@25m':>6} {'@50m':>6} {'@100m':>6} {'median':>9}")
    for name, errors in results.items():
        n = len(errors)
        within = lambda t: 100.0 * sum(1 for e in errors if e is not None and e <= t) / n
        valid = sorted(e for e in errors if e is not None)
        med = valid[len(valid) // 2] if valid else float("nan")
        print(f"{name:<14} {within(25):>5.1f}% {within(50):>5.1f}% {within(100):>5.1f}% {med:>8.1f}m")


if __name__ == "__main__":
    main()
