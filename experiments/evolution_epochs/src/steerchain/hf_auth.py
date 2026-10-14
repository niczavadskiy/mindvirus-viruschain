"""Resolve and apply Hugging Face Hub credentials before model downloads."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from steerchain.config import ExperimentConfig, repo_root


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_hf_token(config: ExperimentConfig | None = None) -> str | None:
    """Prefer config.hf_token, then HF_TOKEN / HUGGING_FACE_HUB_TOKEN, then .env files."""
    for path in (repo_root() / ".env", repo_root() / "configs" / ".env"):
        _load_dotenv(path)
    if config is not None:
        token = (config.hf_token or "").strip()
        if token:
            return token
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def apply_hf_token(config: ExperimentConfig | None = None) -> str | None:
    """Set Hub env vars (and optional login) so from_pretrained uses auth."""
    try:
        from steerchain.hub_patch import prepare_hub_downloads

        prepare_hub_downloads()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"hub_patch skipped: {exc}", stacklevel=2)
    token = resolve_hf_token(config)
    if not token:
        warnings.warn(
            "HF_TOKEN not set; Hub downloads may be rate-limited or fail.",
            stacklevel=2,
        )
        return None
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception:
        # Env vars alone are enough for transformers / hub downloads.
        pass
    return token
