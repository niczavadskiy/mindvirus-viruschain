"""Run conditions A–D (and E, when configured) and write the report."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from steerchain.backend import build_backend
from steerchain.chain import run_condition
from steerchain.config import ExperimentConfig, repo_root
from steerchain.detector import build_detector
from steerchain.hf_auth import apply_hf_token
from steerchain.ideologies import ideologies_for, load_default_soul
from steerchain.log import RunLogger
from steerchain.report import write_report


def stamped_run_dir(base: Path | str, when: datetime | None = None) -> Path:
    """Append YYYYMMDD_HHMMSS under the configured run base (e.g. runs/smoke/…)."""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(base) / stamp


def run_experiment(
    config: ExperimentConfig,
    run_dir: Path | None = None,
    *,
    stamp_run: bool = True,
) -> list[dict]:
    destination = Path(run_dir) if run_dir is not None else Path(config.run_dir)
    if not destination.is_absolute():
        destination = repo_root() / destination
    if stamp_run:
        destination = stamped_run_dir(destination)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={destination}", flush=True)
    token = apply_hf_token(config)
    data_dir = config.resolved_data_dir()
    default_soul = load_default_soul(data_dir)
    ideologies = ideologies_for(config)
    n_jobs = len(ideologies) * len(config.conditions) * len(config.seeds)
    print(
        f"plan ideologies={len(ideologies)} conditions={config.conditions} "
        f"seeds={config.seeds} epochs={config.n_epochs} hops={config.n_hops} "
        f"agents_per_hop={config.agents_per_hop} jobs={n_jobs}",
        flush=True,
    )
    # One HF host for dialogues, judge, and probes (avoids loading the model twice).
    print("loading backend/detector…", flush=True)
    shared_backend = build_backend(config, default_soul) if config.backend == "hf" else None
    detector = build_detector(
        config,
        ideologies,
        default_soul,
        complete_fn=None if shared_backend is None else shared_backend.complete,
    )
    print("backend ready", flush=True)
    rows: list[dict] = []
    job = 0
    for ideology in ideologies:
        for condition in config.conditions:
            for seed in config.seeds:
                job += 1
                print(
                    f"job {job}/{n_jobs}: {ideology.ideology_id}/{condition}/seed_{seed}",
                    flush=True,
                )
                backend = shared_backend if shared_backend is not None else build_backend(config, default_soul)
                backend.release()
                if hasattr(backend, "steer_log"):
                    backend.steer_log.clear()
                if hasattr(backend, "illegal_steering"):
                    backend.illegal_steering.clear()
                sub = destination / ideology.ideology_id / condition / f"seed_{seed}"
                logger = RunLogger(sub)
                payload = {
                    **asdict(config),
                    "ideology_id": ideology.ideology_id,
                    "condition": condition,
                    "seed": seed,
                    "tau": detector.tau(ideology),
                    "transformer_layer_k": backend.transformer_layer_k,
                    "n_layers": getattr(backend, "n_layers", None),
                    "hidden_size": backend.hidden_size,
                    "model_id_resolved": backend.model_id,
                    "backend": config.backend,
                    "run_dir_stamped": str(destination),
                    "run_started_at": destination.name if stamp_run else None,
                    "hf_token_set": bool(token),
                }
                payload["hf_token"] = None  # never persist secrets into run artifacts
                logger.write_config(payload)
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
    print(f"run_dir={destination}", flush=True)
    return rows
