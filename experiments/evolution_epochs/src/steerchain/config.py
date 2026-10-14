"""Experiment configuration. Hop counts and transformer layer k are separate fields.

transformer_layer_k=None means: capture every residual layer, then lock contrastive
steering via layer_select_mode after the first successful epoch.
An explicit integer forces that layer instead of auto-select.

layer_select_mode:
  - max_delta: argmax_l ||mean Δ_l|| on successful pairs
  - probe_ba: argmax_l balanced accuracy of a ridge linear probe on Δ_l
    predicting whether the new-hop agent accepted the idea (Y)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_LAYER_SELECT_MODES = ("max_delta", "probe_ba")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class SAEConfig:
    enabled: bool = False
    repo_id: str = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100"
    architecture: str = "topk"
    top_k: int = 100
    d_sae: int = 32768
    d_model: int = 2048
    activation_site: str = "resid_post_layer"
    transformer_layer_k: int | None = 12
    decoder_topk: int = 50
    consensus_threshold: float = 0.5
    cluster_method: str = "agglomerative"
    run_mode_a: bool = True
    run_mode_b: bool = True
    run_causal: bool = False
    local_dir: str | None = None
    revision: str | None = None


@dataclass
class ExperimentConfig:
    model_id: str = "Qwen/Qwen3.5-2B-Base"
    judge_model_id: str | None = None
    embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    backend: str = "stub"
    embedder: str = "lexical"
    n_hops: int = 10
    agents_per_hop: int = 20
    turns_per_hop: int = 10
    max_tool_loops: int = 2
    n_epochs: int = 2
    seeds: list[int] = field(default_factory=lambda: [0])
    n_seed_agents: int = 1
    ideologies: list[str] = field(default_factory=lambda: ["whale_welfare", "us_dominance"])
    conditions: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    norm_mode: str = "unit"
    beta: float = 1.0
    fixed_norm: float = 1.0
    tau: float | None = None
    pair_spreader_hop: int = 9
    pair_target_hop: int = 10
    probe_retention_threshold: float = 1.0
    transformer_layer_k: int | None = None
    layer_select_mode: str = "max_delta"
    hidden_size: int = 32
    max_new_tokens: int = 256
    run_probes: bool = True
    data_dir: str | None = None
    run_dir: str = "runs/smoke"
    sae: SAEConfig = field(default_factory=SAEConfig)

    def resolved_data_dir(self) -> Path:
        if self.data_dir:
            return Path(self.data_dir)
        return repo_root() / "data" / "ideologies"

    def resolved_judge_model(self) -> str:
        return self.judge_model_id or self.model_id


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in ExperimentConfig.__dataclass_fields__.values()}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    sae_raw = raw.pop("sae", None)
    config = ExperimentConfig(**raw)
    if config.layer_select_mode not in _LAYER_SELECT_MODES:
        raise ValueError(
            f"Unknown layer_select_mode {config.layer_select_mode!r}. "
            f"Expected one of {_LAYER_SELECT_MODES}."
        )
    if sae_raw is not None:
        sae_known = {f.name for f in SAEConfig.__dataclass_fields__.values()}
        sae_unknown = sorted(set(sae_raw) - sae_known)
        if sae_unknown:
            raise ValueError(f"Unknown sae config keys: {sae_unknown}")
        config.sae = SAEConfig(**sae_raw)
    return config
