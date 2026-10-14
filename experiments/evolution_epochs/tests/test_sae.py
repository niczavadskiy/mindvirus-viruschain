"""Offline SAE module tests (synthetic TopK SAE, no GPU / HF download)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from steerchain.config import ExperimentConfig, SAEConfig
from steerchain.sae.analyze import run_sae_analysis
from steerchain.sae.checkpoint import TopKSAE
from steerchain.sae.compat import check_compatibility
from steerchain.sae.mode_a import decoder_cosine_scores, mode_a_for_direction, topk_features
from steerchain.sae.mode_b import encode_pair_record, encode_state
from steerchain.sae.pool import agglomerative_clusters, consensus_topk, intersection_topk


def test_topk_encode_scatters_exact_preacts_not_relu():
    """TopK must keep signed pre-activations; ReLU-only simplification is forbidden."""
    d_model, d_sae, top_k = 8, 16, 3
    W_enc = np.zeros((d_sae, d_model), dtype=np.float64)
    # Feature 0 strongly positive, 1 strongly negative, 2 mild positive.
    W_enc[0, 0] = 10.0
    W_enc[1, 0] = -10.0
    W_enc[2, 0] = 1.0
    W_enc[3, 0] = 0.5
    W_dec = W_enc.T.copy()
    sae = TopKSAE(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=np.zeros(d_sae),
        b_dec=np.zeros(d_model),
        top_k=top_k,
        repo_id="synthetic://test",
        layer_k=0,
        checksum="x",
        source_path="synthetic",
    )
    h = np.zeros(d_model)
    h[0] = 1.0
    acts = sae.encode(h)
    assert acts.shape == (d_sae,)
    # TopK by value picks 10, 1, 0.5 — not the large negative pre-act.
    assert acts[0] == pytest.approx(10.0)
    assert acts[2] == pytest.approx(1.0)
    assert acts[3] == pytest.approx(0.5)
    assert acts[1] == 0.0
    assert np.count_nonzero(acts) == top_k
    # Non-selected features stay exactly zero (not ReLU of all pre-acts).
    assert np.sum(np.abs(acts[4:])) == 0.0


def test_topk_encode_preserves_negative_when_among_topk():
    d_model, d_sae, top_k = 4, 5, 2
    W_enc = np.eye(d_sae, d_model) * 0  # rebuild carefully
    W_enc = np.zeros((d_sae, d_model))
    W_enc[0, 0] = 1.0
    W_enc[1, 0] = 0.5
    W_enc[2, 0] = -0.1
    sae = TopKSAE(
        W_enc=W_enc,
        W_dec=W_enc.T.copy(),
        b_enc=np.array([-2.0, -2.0, -2.0, -2.0, -2.0]),
        b_dec=np.zeros(d_model),
        top_k=top_k,
        repo_id="synthetic://neg",
        layer_k=0,
        checksum="x",
        source_path="synthetic",
    )
    # pre = [−1, −1.5, −2.1, −2, −2]; TopK largest are −1 and −1.5 (features 0,1)
    acts = sae.encode(np.array([1.0, 0.0, 0.0, 0.0]))
    assert acts[0] == pytest.approx(-1.0)
    assert acts[1] == pytest.approx(-1.5)
    assert acts[2] == 0.0


def test_compat_qwen_ok_and_stub_incompatible():
    sae = SAEConfig(transformer_layer_k=12, d_model=2048)
    ok = check_compatibility(
        host_model_id="Qwen/Qwen3.5-2B-Base",
        sae=sae,
        residual_dim=2048,
        transformer_layer_k=12,
    )
    assert ok.compatible
    assert ok.status == "compatible"

    bad = check_compatibility(
        host_model_id="stub-model",
        sae=sae,
        residual_dim=32,
        transformer_layer_k=12,
    )
    assert not bad.compatible
    assert bad.status == "incompatible"
    assert any("host" in r.lower() or "stub" in r.lower() or "not Qwen" in r for r in bad.reasons)


def test_compat_synthetic_allowed():
    sae = SAEConfig(
        repo_id="synthetic://topk-d32-w64-k8",
        d_model=32,
        d_sae=64,
        top_k=8,
        transformer_layer_k=0,
    )
    report = check_compatibility(
        host_model_id="stub",
        sae=sae,
        residual_dim=32,
        transformer_layer_k=0,
        allow_synthetic=True,
    )
    assert report.compatible


def test_mode_a_ranks_aligned_decoder_column():
    sae = TopKSAE.synthetic(d_model=16, d_sae=32, top_k=4, layer_k=1, seed=7)
    # Align direction with feature 5 decoder column.
    direction = sae.W_dec[:, 5].copy()
    scores = decoder_cosine_scores(direction, sae)
    top = topk_features(scores, k=5, positive_only=True)
    assert top[0]["feature_id"] == 5
    assert top[0]["cosine"] == pytest.approx(1.0, abs=1e-9)
    result = mode_a_for_direction(
        direction=direction,
        sae=sae,
        ideology_id="whale_welfare",
        epoch=0,
        vector_kind="delta_plus_aggregate",
        decoder_topk=5,
    )
    assert result["topk"][0]["feature_id"] == 5
    assert "causal" not in result["disclaimer"].lower() or "not" in result["disclaimer"].lower()


def test_mode_b_encode_pair_record():
    sae = TopKSAE.synthetic(d_model=8, d_sae=16, top_k=3, layer_k=0, seed=1)
    n_layers = 3
    sender = np.random.default_rng(0).normal(size=(n_layers, 8))
    target = np.random.default_rng(1).normal(size=(n_layers, 8))
    episode = {
        "episode_id": "ep-1",
        "ideology_id": "whale_welfare",
        "epoch": 0,
        "y_after": 1,
    }
    row = encode_pair_record(
        episode=episode,
        sender_hidden=sender,
        target_hidden=target,
        sae=sae,
        layer=0,
    )
    assert row["sender"]["n_nonzero"] == 3
    assert row["target"]["n_nonzero"] == 3
    assert len(row["sender"]["active_feature_ids"]) == 3
    assert row["post_success_target_state"] is None
    single = encode_state(sender, sae, layer=0)
    assert single["n_nonzero"] == sae.top_k


def test_pool_intersection_consensus_clusters():
    rng = np.random.default_rng(0)
    # Shared feature 0 is top for both; idea-specific elsewhere.
    scores_a = rng.normal(size=20) * 0.01
    scores_b = rng.normal(size=20) * 0.01
    scores_a[0] = 0.9
    scores_b[0] = 0.8
    scores_a[1] = 0.7
    scores_b[2] = 0.7
    rows = {"a": scores_a, "b": scores_b}
    shared = intersection_topk(rows, k=2)
    assert 0 in shared
    consensus = consensus_topk(rows, k=2, threshold=0.5)
    assert any(item["feature_id"] == 0 and item["C_f"] == 1.0 for item in consensus)
    matrix = np.stack([scores_a, scores_b], axis=0)
    clusters = agglomerative_clusters(["a", "b"], matrix, n_clusters=1)
    assert clusters["n_clusters"] == 1
    assert set(clusters["labels"]) == {"a", "b"}


def _write_mini_run(root: Path, *, d_model: int = 16, n_layers: int = 4, layer_k: int = 1) -> None:
    rng = np.random.default_rng(42)
    for ideology in ("whale_welfare", "us_dominance"):
        seed_dir = root / ideology / "A" / "seed_0"
        states = seed_dir / "states"
        steering = seed_dir / "steering"
        states.mkdir(parents=True)
        steering.mkdir(parents=True)
        episodes = []
        for i, y in enumerate([1, 1, 0]):
            eid = f"{ideology}-A-s0-e0-h10-i{i}"
            sender = rng.normal(size=(n_layers, d_model))
            target = rng.normal(size=(n_layers, d_model)) * 0.5
            if y == 1:
                # Make a consistent layer-k direction for Mode A.
                sender[layer_k] += 2.0
            np.save(states / f"{eid}_sender.npy", sender)
            np.save(states / f"{eid}_target.npy", target)
            episodes.append(
                {
                    "episode_id": eid,
                    "epoch": 0,
                    "condition": "A",
                    "ideology_id": ideology,
                    "seed": 0,
                    "sender_hop": 9,
                    "target_hop": 10,
                    "sender_status_before": True,
                    "target_status_before": False,
                    "y_after": y,
                    "hidden_sender_path": str(states / f"{eid}_sender.npy"),
                    "hidden_target_path": str(states / f"{eid}_target.npy"),
                }
            )
        (seed_dir / "episodes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in episodes) + "\n",
            encoding="utf-8",
        )
        delta = rng.normal(size=(n_layers, d_model))
        delta[layer_k] += 1.0
        np.save(steering / "epoch_0_delta.npy", delta)
        np.save(steering / "epoch_0_cumulative.npy", delta * 0.5)
        (steering / "summary.jsonl").write_text(
            json.dumps({"epoch": 0, "n_pairs": 2}) + "\n",
            encoding="utf-8",
        )
        (seed_dir / "config.json").write_text("{}", encoding="utf-8")


def test_analyze_ok_with_synthetic(tmp_path: Path):
    d_model, layer_k = 16, 1
    _write_mini_run(tmp_path, d_model=d_model, layer_k=layer_k)
    sae = TopKSAE.synthetic(d_model=d_model, d_sae=32, top_k=5, layer_k=layer_k, seed=3)
    config = ExperimentConfig(
        model_id="stub",
        hidden_size=d_model,
        sae=SAEConfig(
            enabled=True,
            transformer_layer_k=layer_k,
            d_model=d_model,
            d_sae=32,
            top_k=5,
            decoder_topk=5,
            consensus_threshold=0.5,
        ),
    )
    summary = run_sae_analysis(run_dir=tmp_path, config=config, sae=sae)
    assert summary["status"] == "ok"
    out = tmp_path / "sae"
    assert (out / "compatibility.json").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "report.md").is_file()
    assert (out / "scores_mode_a.jsonl").is_file()
    assert (out / "activations_mode_b.jsonl").is_file()
    assert (out / "consensus.json").is_file()
    assert (out / "clusters.json").is_file()
    assert (out / "success_vs_fail.json").is_file()
    assert (out / "stability.json").is_file()
    assert (out / "hypotheses_for_causal.json").is_file()
    assert (tmp_path / "causal_sae" / "status.json").is_file()
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "correlational" in report.lower()
    assert summary["causal_separated"] is True


def test_analyze_skips_incompatible_without_crash(tmp_path: Path):
    _write_mini_run(tmp_path, d_model=32, layer_k=0)
    config = ExperimentConfig(
        model_id="stub-backend",
        hidden_size=32,
        sae=SAEConfig(
            enabled=True,
            repo_id="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100",
            d_model=2048,
            transformer_layer_k=12,
        ),
    )
    summary = run_sae_analysis(run_dir=tmp_path, config=config)
    assert summary["status"] == "skipped_incompatible"
    out = tmp_path / "sae"
    assert (out / "compatibility.json").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "report.md").is_file()
    # Must not write Mode A/B artifacts when incompatible.
    assert not (out / "scores_mode_a.jsonl").exists()
    assert not (out / "activations_mode_b.jsonl").exists()
