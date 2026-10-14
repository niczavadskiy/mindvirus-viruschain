"""CLI: python -m steerchain --config configs/smoke.yaml"""

from __future__ import annotations

import argparse
from pathlib import Path

from steerchain.config import load_config
from steerchain.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hop-chain steering experiment.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--sae-only",
        action="store_true",
        help="Skip simulation; run SAE analysis on an existing --run-dir.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.sae_only:
        if args.run_dir is None:
            raise SystemExit("--sae-only requires --run-dir")
        from steerchain.sae.analyze import run_sae_analysis

        config.sae.enabled = True
        summary = run_sae_analysis(run_dir=args.run_dir, config=config)
        print(f"sae status={summary.get('status')}")
        return
    rows = run_experiment(config, run_dir=args.run_dir)
    print(f"wrote {len(rows)} epoch rows")


if __name__ == "__main__":
    main()
