"""Tests for env parsing helpers in backend/app/services/runtime.py."""

import pytest

from backend.app.services.runtime import env_bool, env_float, env_int, parse_boolish


class TestParseBoolish:
    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", " yes ", "on"])
    def test_truthy(self, raw):
        assert parse_boolish(raw) is True

    @pytest.mark.parametrize("raw", ["0", "false", "NO", " off "])
    def test_falsy(self, raw):
        assert parse_boolish(raw, default=True) is False

    def test_garbage_falls_back_to_default(self):
        assert parse_boolish("maybe", default=True) is True
        assert parse_boolish("maybe", default=False) is False

    def test_none_falls_back_to_default(self):
        assert parse_boolish(None, default=True) is True


class TestEnvInt:
    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("GEOSPY_TEST_INT", "42")
        assert env_int("GEOSPY_TEST_INT", 7) == 42

    def test_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("GEOSPY_TEST_INT", raising=False)
        assert env_int("GEOSPY_TEST_INT", 7) == 7

    def test_clamped_to_bounds(self, monkeypatch):
        monkeypatch.setenv("GEOSPY_TEST_INT", "999")
        assert env_int("GEOSPY_TEST_INT", 7, minimum=1, maximum=100) == 100
        monkeypatch.setenv("GEOSPY_TEST_INT", "-5")
        assert env_int("GEOSPY_TEST_INT", 7, minimum=1, maximum=100) == 1


class TestEnvFloat:
    def test_reads_and_clamps(self, monkeypatch):
        monkeypatch.setenv("GEOSPY_TEST_FLOAT", "0.9")
        assert env_float("GEOSPY_TEST_FLOAT", 0.5) == pytest.approx(0.9)
        assert env_float("GEOSPY_TEST_FLOAT", 0.5, maximum=0.6) == pytest.approx(0.6)
        monkeypatch.setenv("GEOSPY_TEST_FLOAT", "-1.0")
        assert env_float("GEOSPY_TEST_FLOAT", 0.5, minimum=0.0) == pytest.approx(0.0)


class TestEnvBool:
    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("GEOSPY_TEST_BOOL", "true")
        assert env_bool("GEOSPY_TEST_BOOL") is True
        monkeypatch.setenv("GEOSPY_TEST_BOOL", "off")
        assert env_bool("GEOSPY_TEST_BOOL", default=True) is False
