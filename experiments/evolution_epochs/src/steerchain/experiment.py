"""Run conditions A–D (and E, when configured) and write the report."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from steerchain.backend import build_backend
from steerchain.chain import run_condition
from steerchain.config import ExperimentConfig, repo_root
from steerchain.detector import build_detector
from steerchain.ideologies import ideologies_for, load_default_soul
from steerchain.log import RunLogger
from steerchain.report import write_report


def run_experiment(config: ExperimentConfig, run_dir: Path | None = None) -> list[dict]:
    destination = Path(run_dir) if run_dir is not None else Path(config.run_dir)
    if not destination.is_absolute():
        destination = repo_root() / destination
    data_dir = config.resolved_data_dir()
    default_soul = load_default_soul(data_dir)
    ideologies = ideologies_for(config)
    detector = build_detector(config, ideologies, default_soul)
    rows: list[dict] = []
    for ideology in ideologies:
        for condition in config.conditions:
            for seed in config.seeds:
                backend = build_backend(config, default_soul)
                sub = destination / ideology.ideology_id / condition / f"seed_{seed}"
                logger = RunLogger(sub)
                logger.write_config(
                    {
                        **asdict(config),
                        "ideology_id": ideology.ideology_id,
                        "condition": condition,
                        "seed": seed,
                        "tau": detector.tau(ideology),
                        "transformer_layer_k": backend.transformer_layer_k,
                        "hidden_size": backend.hidden_size,
                        "model_id_resolved": backend.model_id,
                    }
                )
                rows.extend(
                    run_condition(
                        config=config,
                        ideology=ideology,
                        condition=condition,
                        seed=seed,
                        default_soul=default_soul,
                        backend=backend,
                        detector=detector,
                        logger=logger,
                    )
                )
    write_report(destination, rows)
    if config.sae.enabled:
        from steerchain.sae.analyze import run_sae_analysis

        run_sae_analysis(run_dir=destination, config=config)
    return rows
