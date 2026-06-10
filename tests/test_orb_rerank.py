"""Tests for the ORB rerank stage.

Includes a synthetic end-to-end check: a candidate that is pixel-identical
to the query must overtake a higher-scored noise candidate after reranking.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from backend.app.services.orb_rerank import (  # noqa: E402
    LOCATE_ORB_IGNORE_BOTTOM_RATIO_DEFAULT,
    OrbRerankConfig,
    normalize_orb_ignore_bottom_ratio,
    rerank_capture_rows_with_orb,
)


def make_config(**overrides):
    base = dict(
        enabled=True,
        top_n=10,
        feature_count=500,
        weight=0.75,
        ransac_top_k=10,
        visualization_limit=1,
        ignore_bottom_ratio=0.0,
        sam2_mask_cars=False,
        sam2_mask_trees=False,
    )
    base.update(overrides)
    return OrbRerankConfig(**base)


def textured_image(seed, size=(240, 320)):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    # Blur slightly so ORB finds stable corners rather than pure noise.
    return cv2.GaussianBlur(img, (3, 3), 0)


def encode_jpg(image):
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


class TestNormalizeIgnoreBottomRatio:
    def test_none_uses_default(self):
        assert normalize_orb_ignore_bottom_ratio(None) == pytest.approx(
            LOCATE_ORB_IGNORE_BOTTOM_RATIO_DEFAULT
        )

    def test_clamped_to_valid_range(self):
        assert normalize_orb_ignore_bottom_ratio(-1.0) == 0.0
        assert normalize_orb_ignore_bottom_ratio(0.9) == 0.6
        assert normalize_orb_ignore_bottom_ratio(0.3) == pytest.approx(0.3)


class TestRerankBehavior:
    def test_disabled_returns_rows_unchanged(self):
        rows = [{"capture_id": 1, "score": 0.5, "filepath": "missing.jpg"}]
        reranked, stats = rerank_capture_rows_with_orb(
            rows,
            image_bytes=b"not-an-image",
            capture_abs_path=lambda p: p,
            config=make_config(enabled=False),
        )
        assert reranked == rows
        assert stats["status"] == "skipped"
        assert stats["reason"] == "disabled"

    def test_empty_candidates(self):
        reranked, stats = rerank_capture_rows_with_orb(
            [],
            image_bytes=b"x",
            capture_abs_path=lambda p: p,
            config=make_config(),
        )
        assert reranked == []
        assert stats["reason"] == "no-candidates"

    def test_missing_files_counted_and_scores_untouched(self, tmp_path):
        query = encode_jpg(textured_image(seed=1))
        rows = [{"capture_id": 1, "score": 0.5, "filepath": "does-not-exist.jpg"}]
        reranked, stats = rerank_capture_rows_with_orb(
            rows,
            image_bytes=query,
            capture_abs_path=lambda p: str(tmp_path / p),
            config=make_config(),
        )
        assert stats["missing_files"] == 1
        assert reranked[0]["score"] == pytest.approx(0.5)

    def test_identical_candidate_overtakes_noise(self, tmp_path):
        query_img = textured_image(seed=1)
        match_path = tmp_path / "match.jpg"
        noise_path = tmp_path / "noise.jpg"
        cv2.imwrite(str(match_path), query_img)
        cv2.imwrite(str(noise_path), textured_image(seed=2))

        rows = [
            # Noise candidate starts AHEAD on vector score.
            {"capture_id": 2, "score": 0.55, "similarity": 0.55, "filepath": "noise.jpg"},
            {"capture_id": 1, "score": 0.50, "similarity": 0.50, "filepath": "match.jpg"},
        ]
        reranked, stats = rerank_capture_rows_with_orb(
            rows,
            image_bytes=encode_jpg(query_img),
            capture_abs_path=lambda p: str(tmp_path / p),
            config=make_config(),
        )
        assert stats["status"] == "completed"
        assert stats["query_keypoints"] > 0
        assert stats["candidates_scored"] >= 1
        # The pixel-identical candidate must rank first after ORB rerank.
        assert reranked[0]["capture_id"] == 1
        match_row = reranked[0]
        assert match_row["orb_reranked"] is True
        assert match_row["orb_good_matches"] > 0
        # ORB bonus is bounded: capped orb_score (0.35) times weight.
        assert match_row["score"] <= 0.50 + 0.75 * 0.35 + 1e-9

    def test_orb_score_capped(self, tmp_path):
        query_img = textured_image(seed=1)
        match_path = tmp_path / "match.jpg"
        cv2.imwrite(str(match_path), query_img)
        rows = [{"capture_id": 1, "score": 0.5, "similarity": 0.5, "filepath": "match.jpg"}]
        reranked, _ = rerank_capture_rows_with_orb(
            rows,
            image_bytes=encode_jpg(query_img),
            capture_abs_path=lambda p: str(tmp_path / p),
            config=make_config(),
        )
        assert reranked[0]["orb_score"] <= 0.35
