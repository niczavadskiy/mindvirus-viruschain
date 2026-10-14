"""End-to-end stub chain: hop 9 to hop 10, steering scope, and the report."""

import json
from pathlib import Path

import numpy as np

from steerchain.backend import build_backend
from steerchain.config import ExperimentConfig, repo_root
from steerchain.detector import lexical_detector
from steerchain.experiment import run_experiment
from steerchain.ideologies import load_default_soul, load_ideology


def _tiny(**overrides) -> ExperimentConfig:
    payload = dict(
        backend="stub",
        embedder="lexical",
        n_hops=10,
        agents_per_hop=2,
        turns_per_hop=1,
        max_tool_loops=2,
        n_epochs=2,
        seeds=[0],
        n_seed_agents=1,
        ideologies=["whale_welfare"],
        conditions=["A", "B", "C", "D"],
        norm_mode="unit",
        beta=1.0,
        fixed_norm=1.0,
        tau=None,
        pair_spreader_hop=9,
        pair_target_hop=10,
        probe_retention_threshold=1.0,
        transformer_layer_k=None,
        layer_select_mode="max_delta",
        hidden_size=8,
        run_probes=True,
        data_dir=str(repo_root() / "data" / "ideologies"),
    )
    payload.update(overrides)
    return ExperimentConfig(**payload)


def test_chain_extracts_hop9_to_hop10_and_steers_only_hop0(tmp_path: Path):
    rows = run_experiment(_tiny(), run_dir=tmp_path, stamp_run=False)
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    for condition, items in by_condition.items():
        epoch0 = next(item for item in items if item["epoch"] == 0)
        epoch1 = next(item for item in items if item["epoch"] == 1)
        assert epoch0["p_final"] == 1.0
        assert epoch0["r_into_final"] == 1.0
        assert epoch0["n_positive_pairs"] == 2
        assert epoch0["repeated_senders"] == {}
        assert epoch0["steering"]["applied"] is False
        assert epoch0["steering"]["N_status"] == "computed"
        assert epoch0["steering"]["selected_layer"] == 1
        assert epoch0["steering"]["layer_select_mode"] == "max_delta"
        assert epoch0["retention"] == 1.0
        assert epoch1["p_final"] == 1.0
        assert epoch0["steering"]["steer_hops"] == []
        if condition == "A":
            assert epoch1["steering"]["applied"] is False
            assert epoch1["steering"]["steer_hops"] == []
        else:
            assert epoch1["steering"]["applied"] is True
            assert abs(epoch1["steering"]["applied_norm"] - 1.0) < 1e-6
            assert epoch1["steering"]["steer_hops"]
            assert set(epoch1["steering"]["steer_hops"]) == {0}
            assert epoch1["steering"]["selected_layer"] == 1
    delta_path = tmp_path / "whale_welfare" / "A" / "seed_0" / "steering" / "epoch_0_delta.npy"
    saved = np.load(delta_path)
    state_dir = tmp_path / "whale_welfare" / "A" / "seed_0" / "states"
    senders = sorted(state_dir.glob("*h10*sender.npy"))
    targets = sorted(state_dir.glob("*h10*target.npy"))
    manual_stack = np.mean([np.load(path) for path in senders], axis=0) - np.mean(
        [np.load(path) for path in targets], axis=0
    )
    selected = int(np.argmax(np.linalg.norm(manual_stack, axis=1)))
    assert selected == 1
    np.testing.assert_allclose(saved, manual_stack[selected])
    assert saved.ndim == 1
    episode_lines = (tmp_path / "whale_welfare" / "B" / "seed_0" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    hop10 = [json.loads(line) for line in episode_lines if '"target_hop": 10' in line]
    assert hop10
    for episode in hop10:
        assert episode["target_status_before"] is False
        assert episode["y_after"] == 1
        assert episode["capture_sender"] == "t_send"
        assert episode["capture_target"] == "t_pre_end"
        assert episode["sender_hop"] == 9
        assert episode["hidden_sender_path"]
        assert episode["hidden_target_path"]
    assert (tmp_path / "comparison.csv").is_file()
    assert (tmp_path / "figures" / "p_final_by_epoch.png").is_file()
    assert (tmp_path / "figures" / "semantic_drift_by_epoch.png").is_file()
    assert (tmp_path / "figures" / "cso_drift_by_epoch.png").is_file()
    assert (tmp_path / "figures" / "transition_by_hop.png").is_file()


def test_no_successful_pairs_leaves_delta_absent(tmp_path: Path):
    config = _tiny(conditions=["A", "D"], n_epochs=2)
    data_dir = repo_root() / "data" / "ideologies"
    default_soul = load_default_soul(data_dir)
    ideology = load_ideology(data_dir, "whale_welfare")
    detector = lexical_detector([ideology], default_soul)
    from steerchain.chain import run_condition
    from steerchain.log import RunLogger

    backend = build_backend(config, default_soul)
    backend.transmit = False
    logger = RunLogger(tmp_path)
    rows = run_condition(
        config=config,
        ideology=ideology,
        condition="A",
        seed=0,
        default_soul=default_soul,
        backend=backend,
        detector=detector,
        logger=logger,
    )
    assert rows[0]["n_positive_pairs"] == 0
    assert rows[0]["p_final"] == 0.0
    assert rows[0]["steering"]["N_status"] == "absent_no_successful_pairs"
    assert rows[0]["steering"]["substituted_random"] is False
    assert not list((tmp_path / "steering").glob("*delta.npy"))
    summary = (tmp_path / "steering" / "summary.jsonl").read_text(encoding="utf-8")
    assert "absent_no_successful_pairs" in summary
    assert "substituted_random\": false" in summary or '"substituted_random": false' in summary


def test_cumulative_norm_grows_and_rerun_matches(tmp_path: Path):
    config = _tiny(conditions=["C"], n_epochs=3, agents_per_hop=1, turns_per_hop=1)
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_experiment(config, run_dir=first, stamp_run=False)
    run_experiment(config, run_dir=second, stamp_run=False)
    delta0 = np.load(first / "whale_welfare" / "C" / "seed_0" / "steering" / "epoch_0_delta.npy")
    delta1 = np.load(first / "whale_welfare" / "C" / "seed_0" / "steering" / "epoch_1_delta.npy")
    cumulative = np.load(first / "whale_welfare" / "C" / "seed_0" / "steering" / "epoch_1_cumulative.npy")
    np.testing.assert_allclose(cumulative, delta0 + delta1)
    again = np.load(second / "whale_welfare" / "C" / "seed_0" / "steering" / "epoch_1_cumulative.npy")
    np.testing.assert_allclose(cumulative, again)
    fixed_backend = build_backend(config, load_default_soul(repo_root() / "data" / "ideologies"))
    assert fixed_backend.transformer_layer_k == fixed_backend.n_layers // 2
    summary = json.loads(
        (first / "whale_welfare" / "C" / "seed_0" / "steering" / "summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert summary["selected_layer"] == 1
    norms = summary["layer_delta_norms"]
    assert norms.index(max(norms)) == 1
