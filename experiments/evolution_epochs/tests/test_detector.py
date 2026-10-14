"""Detector, tau lock, blinded judge text, and separate drift metrics."""

from pathlib import Path

import numpy as np

from steerchain.config import load_config, repo_root
from steerchain.detector import (
    JUDGE_SYSTEM,
    Detector,
    OverlapJudge,
    OverlapProbeJudge,
    blinded,
    files_text,
    judge_user_message,
    lock_tau,
)
from steerchain.embeddings import LexicalEmbedder
from steerchain.episode import system_prompt
from steerchain.ideologies import load_default_soul, load_ideology, load_manifest
from steerchain.injector import add_vector, attach_additive_hook, resolve_layers
from steerchain.metrics import epoch_metrics
from steerchain.records import EpisodeRecord
from types import SimpleNamespace


def test_published_ideologies_and_probes_load():
    data_dir = repo_root() / "data" / "ideologies"
    manifest = load_manifest(data_dir)
    assert manifest["ideologies"] == [
        "whale_welfare",
        "us_dominance",
        "ai_welfare",
        "ai_supremacy",
        "chinese_dominance",
        "german_dominance",
    ]
    soul = load_default_soul(data_dir)
    assert "SOUL.md" in soul
    for ideology_id in manifest["ideologies"]:
        ideology = load_ideology(data_dir, ideology_id)
        assert ideology.short
        assert len(ideology.payload) > 200
        assert len(ideology.questions) == 3


def test_tau_is_locked_before_outcomes_and_separates_the_idea():
    data_dir = repo_root() / "data" / "ideologies"
    soul = load_default_soul(data_dir)
    ideology = load_ideology(data_dir, "whale_welfare")
    embedder = LexicalEmbedder()
    tau = lock_tau(ideology, soul, embedder)
    detector = Detector(embedder, OverlapJudge(), OverlapProbeJudge(), {"whale_welfare": tau})
    positive = detector.classify(
        files_text({"SOUL.md": ideology.payload, "MEMORY.md": ""}),
        ideology,
        [ideology.short] * 3,
    )
    negative = detector.classify(
        files_text({"SOUL.md": soul, "MEMORY.md": ""}),
        ideology,
        ["I have no fixed belief."] * 3,
    )
    assert positive.cosine_ref >= tau
    assert positive.judge == 1
    assert positive.h_plus
    assert negative.cosine_ref < tau
    assert negative.judge == 0
    assert not negative.h_plus
    assert positive.cosine_payload > negative.cosine_payload


def test_judge_instructions_do_not_name_the_arm():
    data_dir = repo_root() / "data" / "ideologies"
    ideology = load_ideology(data_dir, "us_dominance")
    assert blinded(JUDGE_SYSTEM)
    message = judge_user_message(ideology, "A neutral note.")
    assert "JUDGMENT" not in message
    assert "beta" not in message.lower()
    prompt = system_prompt("A short soul.", ["peer"])
    lowered = prompt.lower()
    assert "steering" not in lowered
    assert "baseline" not in lowered
    assert "beta" not in lowered


def test_spread_and_drift_are_separate_fields():
    def row(cosine: float) -> EpisodeRecord:
        return EpisodeRecord(
            episode_id=f"ep-{cosine}",
            epoch=0,
            condition="A",
            ideology_id="whale_welfare",
            seed=0,
            model_id="stub",
            transformer_layer_k=2,
            sender_id="s",
            target_id="t",
            sender_hop=9,
            target_hop=10,
            sender_parent_episode_id=None,
            sender_status_before=True,
            target_status_before=False,
            y_after=1,
            cosine_ref_after=cosine,
            cosine_payload_after=cosine,
            judge_after=1,
            probe_scores_after=[2, 2, 2],
            probe_mean_after=2.0,
            sender_probe_mean_before=2.0,
            cso_after=[2, 2, 2, 0, 0, 0, 0, 0, 0],
            cso_drift_after=1.0 - cosine,
            semantic_drift_after=1.0 - cosine,
            theme_scores_after={},
            messages=[],
        )

    common = dict(
        n_hops=10,
        condition="A",
        ideology_id="whale_welfare",
        seed=0,
        epoch=0,
        retention_threshold=1.0,
        steering_summary={},
        n_positive_pairs=1,
        n_negative_pairs=0,
        repeated_senders={},
    )
    high = epoch_metrics(records=[row(0.9)], **common)
    low = epoch_metrics(records=[row(0.2)], **common)
    assert high["p_final"] == low["p_final"] == 1.0
    assert high["semantic_drift_final"] < low["semantic_drift_final"]
    assert high["cso_drift_final"] != low["cso_drift_final"]


def test_smoke_config_uses_the_paper_topology():
    config = load_config(repo_root() / "configs" / "smoke.yaml")
    assert config.n_hops == 10
    assert config.agents_per_hop == 20
    assert config.turns_per_hop == 10
    assert config.max_tool_loops == 2
    assert config.n_epochs == 2
    assert config.ideologies == ["whale_welfare", "us_dominance"]
    assert config.conditions == ["A", "B", "C", "D"]
    assert config.norm_mode == "unit"
    assert config.beta == 1.0
    assert config.pair_spreader_hop == 9
    assert config.pair_target_hop == 10
    assert config.model_id == "Qwen/Qwen3.5-2B-Base"
    assert config.backend == "stub"
    assert config.sae.repo_id == "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100"
    assert config.sae.architecture == "topk"
    assert config.sae.enabled is False
    assert config.transformer_layer_k is None
    assert config.layer_select_mode == "max_delta"
    assert config.sae.transformer_layer_k == 12


def test_smoke_hf_config_is_real_dialogue_topology():
    config = load_config(repo_root() / "configs" / "smoke_hf.yaml")
    assert config.backend == "hf"
    assert config.embedder == "transformers"
    assert config.model_id == "Qwen/Qwen3.5-2B-Base"
    assert config.n_hops == 10
    assert config.agents_per_hop == 20
    assert config.turns_per_hop == 10
    assert config.n_epochs == 2
    assert config.pair_spreader_hop == 9
    assert config.pair_target_hop == 10
    assert config.transformer_layer_k is None
    assert config.layer_select_mode == "max_delta"
    assert config.run_probes is True
    assert config.conditions == ["A", "B", "C", "D"]
    mini = load_config(repo_root() / "configs" / "smoke_hf_mini.yaml")
    assert mini.backend == "hf"
    assert mini.embedder == "transformers"
    assert mini.model_id == "Qwen/Qwen3.5-2B-Base"
    assert mini.agents_per_hop == 2
    assert mini.turns_per_hop == 2
    assert mini.n_hops == 10
    assert mini.pair_target_hop == 10


def test_layer_resolver_rejects_unknown_families_and_hooks_one_layer():
    class Layer:
        def __init__(self):
            self.hooks = []

        def register_forward_hook(self, fn):
            self.hooks.append(fn)

            class Handle:
                def remove(self_inner):
                    self.hooks.remove(fn)

            return Handle()

        def forward(self, hidden):
            output = hidden
            for hook in list(self.hooks):
                output = hook(self, None, output)
            return output

    llama = SimpleNamespace(
        config=SimpleNamespace(model_type="llama"),
        model=SimpleNamespace(layers=[Layer() for _ in range(4)]),
    )
    layers = resolve_layers(llama)
    assert len(layers) == 4
    vector = np.array([1.0, 0.0, 0.0])
    handle = attach_additive_hook(llama, 2, vector)
    hidden = np.zeros((1, 2, 3))
    steered = llama.model.layers[2].forward(hidden)
    untouched = llama.model.layers[1].forward(hidden)
    np.testing.assert_allclose(steered, add_vector(hidden, vector))
    np.testing.assert_allclose(untouched, hidden)
    handle.remove()
    assert llama.model.layers[2].hooks == []
    try:
        resolve_layers(SimpleNamespace(config=SimpleNamespace(model_type="gpt2")))
    except ValueError as exc:
        assert "gpt2" in str(exc)
    else:
        raise AssertionError("unknown model_type should fail")


def test_files_text_is_stable():
    text = files_text({"B.md": "b", "A.md": "a"})
    assert text.index("A.md") < text.index("B.md")
    assert Path(repo_root() / "data" / "ideologies" / "default_soul.md").is_file()
