"""Smoke tests of the Streamlit UI with Streamlit's headless AppTest runner."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_explains_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error)


def test_app_explains_missing_index(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "missing-index"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("build-index" in w.value for w in at.warning)
