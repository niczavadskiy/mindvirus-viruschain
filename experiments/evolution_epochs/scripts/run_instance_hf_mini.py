"""Instance / notebook entrypoint for the locked HF mini smoke.

Notebook:
  %cd /workspace/mind-viruses/experiments/evolution_epochs
  %run scripts/run_instance_hf_mini.py

CLI:
  python scripts/run_instance_hf_mini.py
  python scripts/run_instance_hf_mini.py --config configs/smoke_hf_mini.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap + run HF mini smoke on an instance.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "smoke_hf_mini_instruct.yaml",
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Write/update .env HF_TOKEN then run (preferred over putting secrets in yaml).",
    )
    args = parser.parse_args()

    if args.hf_token:
        env_path = ROOT / ".env"
        env_path.write_text(f"HF_TOKEN={args.hf_token.strip()}\n", encoding="utf-8")
        print(f"wrote {env_path}")

    if not args.skip_install:
        _pip("install", "-q", "-r", str(ROOT / "requirements-hf.txt"))
        _pip("install", "-q", "-e", str(ROOT))
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "torchaudio",
                "torchvision",
                "sentence-transformers",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Ensure repo .env is visible before steerchain import path runs.
    sys.path.insert(0, str(ROOT / "src"))
    from steerchain.hf_auth import apply_hf_token
    from steerchain.config import load_config

    config = load_config(args.config)
    token = apply_hf_token(config)
    print(f"hf_token_set={bool(token)}")
    try:
        import torch

        print(f"cuda={torch.cuda.is_available()} torch={torch.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch check failed: {exc}")

    cmd = [sys.executable, "-m", "steerchain", "--config", str(args.config)]
    print("running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
