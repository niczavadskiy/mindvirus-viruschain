from __future__ import annotations

import os

from steerchain.config import ExperimentConfig
from steerchain.hf_auth import resolve_hf_token


def test_resolve_hf_token_prefers_config(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    config = ExperimentConfig(hf_token="hf_from_config")
    assert resolve_hf_token(config) == "hf_from_config"


def test_resolve_hf_token_uses_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    config = ExperimentConfig(hf_token=None)
    assert resolve_hf_token(config) == "hf_from_env"
