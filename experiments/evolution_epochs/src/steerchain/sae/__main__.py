"""CLI: python -m steerchain.sae --config configs/smoke.yaml --run-dir runs/smoke"""

from __future__ import annotations

import argparse
from pathlib import Path

from steerchain.config import load_config
from steerchain.sae.analyze import run_sae_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAE analysis on a finished hop-chain run.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    config.sae.enabled = True
    summary = run_sae_analysis(run_dir=args.run_dir, config=config)
    print(f"sae status={summary.get('status')}")


if __name__ == "__main__":
    main()
