"""Characterization tests for the locate scoring pipeline.

These pin the current behavior of capture merging, panorama aggregation,
and family clustering so algorithm changes are deliberate, not accidental.
"""

import pytest

from backend.app.services.locate_scoring import (
    aggregate_panorama_candidates,
    cluster_panorama_families,
    haversine_m,
    merge_capture_row,
)

# ~degrees of latitude per meter
DEG_PER_METER_LAT = 1.0 / 111195.0


def make_row(
    capture_id=1,
    panorama_id=10,
    score=0.5,
    similarity=0.5,
    heading=0.0,
    pitch=75.0,
    lat=37.7749,
    lon=-122.4194,
    model_hits=("clip",),
):
    return {
        "capture_id": capture_id,
        "panorama_id": panorama_id,
        "pano_id": f"pano-{panorama_id}",
        "score": score,
        "similarity": similarity,
        "heading": heading,
        "pitch": pitch,
        "lat": lat,
        "lon": lon,
        "model_hits": list(model_hits),
    }


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_m(37.0, -122.0, 37.0, -122.0) < 0.01

    def test_one_degree_latitude(self):
        # 1 degree of latitude is ~111.195 km regardless of longitude.
        assert haversine_m(0.0, 0.0, 1.0, 0.0) == pytest.approx(111195, rel=0.001)

    def test_symmetric(self):
        a = haversine_m(37.7749, -122.4194, 37.8044, -122.2712)
        b = haversine_m(37.8044, -122.2712, 37.7749, -122.4194)
        assert a == pytest.approx(b)


class TestMergeCaptureRow:
    def test_new_capture_creates_weighted_entry(self):
        merged = {}
        merge_capture_row(
            merged,
            make_row(capture_id=1, similarity=0.8),
            model_id="clip",
            model_weight=0.5,
            embedding_base="clip",
        )
        entry = merged[1]
        assert entry["score"] == pytest.approx(0.4)  # similarity * weight
        assert entry["model_hits"] == ["clip"]
        assert entry["model_scores"] == {"clip": 0.8}
        assert entry["embedding_bases"] == ["clip"]

    def test_second_model_sums_scores_and_keeps_max_similarity(self):
        merged = {}
        merge_capture_row(
            merged,
            make_row(capture_id=1, similarity=0.8),
            model_id="clip",
            model_weight=1.0,
            embedding_base="clip",
        )
        merge_capture_row(
            merged,
            make_row(capture_id=1, similarity=0.6),
            model_id="place",
            model_weight=1.0,
            embedding_base="place",
        )
        entry = merged[1]
        assert entry["score"] == pytest.approx(0.8 + 0.6)
        assert entry["model_hits"] == ["clip", "place"]
        assert entry["model_scores"] == {"clip": 0.8, "place": 0.6}
        assert entry["similarity"] == pytest.approx(0.8)  # max kept
        assert entry["embedding_bases"] == ["clip", "place"]

    def test_duplicate_model_does_not_duplicate_hits(self):
        merged = {}
        for _ in range(2):
            merge_capture_row(
                merged,
                make_row(capture_id=1, similarity=0.5),
                model_id="clip",
                model_weight=1.0,
                embedding_base="clip",
            )
        entry = merged[1]
        assert entry["model_hits"] == ["clip"]
        # Scores still sum (this is current behavior: same-model re-merge accumulates).
        assert entry["score"] == pytest.approx(1.0)


class TestAggregatePanoramaCandidates:
    def test_invalid_panorama_id_skipped(self):
        assert aggregate_panorama_candidates([make_row(panorama_id=0)]) == []
        assert aggregate_panorama_candidates([make_row(panorama_id=-3)]) == []

    def test_single_capture_no_support_bonus(self):
        ranked = aggregate_panorama_candidates([make_row(score=0.7)])
        assert len(ranked) == 1
        entry = ranked[0]
        assert entry["panorama_support_bonus"] == 0.0
        assert entry["panorama_score"] == pytest.approx(0.7)
        assert entry["capture_hits"] == 1

    def test_multi_capture_bonus_bounded_and_best_dominates(self):
        rows = [
            make_row(capture_id=1, score=0.7, similarity=0.7, heading=0.0),
            make_row(capture_id=2, score=0.65, similarity=0.65, heading=90.0),
            make_row(capture_id=3, score=0.6, similarity=0.6, heading=180.0, pitch=90.0),
        ]
        ranked = aggregate_panorama_candidates(rows)
        assert len(ranked) == 1
        entry = ranked[0]
        assert entry["capture_hits"] == 3
        assert entry["heading_support"] == 3
        assert entry["pitch_support"] == 2
        assert 0.0 < entry["panorama_support_bonus"] <= 0.05
        # Best capture dominates: score equals best capture's score + bonus.
        assert entry["panorama_score"] == pytest.approx(0.7 + entry["panorama_support_bonus"])
        assert entry["capture_id"] == 1

    def test_ranking_is_descending_by_panorama_score(self):
        rows = [
            make_row(capture_id=1, panorama_id=10, score=0.5),
            make_row(capture_id=2, panorama_id=20, score=0.9),
            make_row(capture_id=3, panorama_id=30, score=0.7),
        ]
        ranked = aggregate_panorama_candidates(rows)
        assert [r["panorama_id"] for r in ranked] == [20, 30, 10]

    def test_heading_binned_to_15_degrees(self):
        rows = [
            make_row(capture_id=1, heading=0.0),
            make_row(capture_id=2, heading=7.0),  # rounds to bin 0 -> same bin
        ]
        ranked = aggregate_panorama_candidates(rows)
        assert ranked[0]["heading_support"] == 1


class TestClusterPanoramaFamilies:
    def test_single_panorama_single_family_no_bonus(self):
        panos = aggregate_panorama_candidates([make_row(score=0.7)])
        families = cluster_panorama_families(panos, 35.0)
        assert len(families) == 1
        fam = families[0]
        assert fam["family_support_bonus"] == 0.0
        assert fam["family_score"] == pytest.approx(fam["panorama_score"])
        assert fam["family_panorama_count"] == 1
        assert fam["family_center_lat"] == pytest.approx(37.7749)

    def test_nearby_panoramas_merge_into_one_family(self):
        lat = 37.7749
        rows = [
            make_row(capture_id=1, panorama_id=10, score=0.8, lat=lat),
            make_row(
                capture_id=2,
                panorama_id=20,
                score=0.6,
                lat=lat + 10.0 * DEG_PER_METER_LAT,  # ~10m away
            ),
        ]
        panos = aggregate_panorama_candidates(rows)
        families = cluster_panorama_families(panos, 35.0)
        assert len(families) == 1
        fam = families[0]
        assert fam["family_panorama_count"] == 2
        # Best panorama wins the family identity.
        assert fam["panorama_id"] == 10
        assert 0.0 < fam["family_support_bonus"] <= 0.035
        assert fam["family_score"] == pytest.approx(
            fam["panorama_score"] + fam["family_support_bonus"]
        )
        # Weighted center sits between the two panoramas, closer to the heavier one.
        assert lat < fam["family_center_lat"] < lat + 10.0 * DEG_PER_METER_LAT

    def test_distant_panoramas_stay_separate(self):
        lat = 37.7749
        rows = [
            make_row(capture_id=1, panorama_id=10, score=0.6, lat=lat),
            make_row(
                capture_id=2,
                panorama_id=20,
                score=0.9,
                lat=lat + 1000.0 * DEG_PER_METER_LAT,  # ~1km away
            ),
        ]
        panos = aggregate_panorama_candidates(rows)
        families = cluster_panorama_families(panos, 35.0)
        assert len(families) == 2
        # Sorted descending by family score.
        assert families[0]["panorama_id"] == 20
        assert families[1]["panorama_id"] == 10

    def test_family_ids_and_panorama_ids_exposed(self):
        panos = aggregate_panorama_candidates(
            [make_row(capture_id=1, panorama_id=10, score=0.7)]
        )
        families = cluster_panorama_families(panos, 35.0)
        assert families[0]["family_id"] == "family-1"
        assert families[0]["family_panorama_ids"] == [10]
