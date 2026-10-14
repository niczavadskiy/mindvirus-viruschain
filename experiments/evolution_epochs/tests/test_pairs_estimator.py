"""Pair filter, delta arithmetic, empty epochs, and the shuffle control."""

import numpy as np

from steerchain.estimator import (
    SteeringMemory,
    layer_probe_balanced_accuracies,
    mean_delta,
    resolve_contrastive_layer,
    scale_direction,
    select_max_probe_ba_layer,
    shuffle_control,
)
from steerchain.injector import select_max_delta_layer
from steerchain.pairs import eligible_pairs, successful_pairs, unsuccessful_pairs
from steerchain.records import EpisodeRecord


def _record(
    episode_id: str,
    *,
    sender_hop: int = 9,
    target_hop: int = 10,
    sender_positive: bool = True,
    target_positive_before: bool = False,
    y_after: int = 1,
    sender_hidden: np.ndarray | None = None,
    target_hidden: np.ndarray | None = None,
) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=episode_id,
        epoch=0,
        condition="A",
        ideology_id="whale_welfare",
        seed=0,
        model_id="stub",
        transformer_layer_k=2,
        sender_id=f"sender-{episode_id}",
        target_id=f"target-{episode_id}",
        sender_hop=sender_hop,
        target_hop=target_hop,
        sender_parent_episode_id=None,
        sender_status_before=sender_positive,
        target_status_before=target_positive_before,
        y_after=y_after,
        cosine_ref_after=0.0,
        cosine_payload_after=0.0,
        judge_after=y_after,
        probe_scores_after=[],
        probe_mean_after=0.0,
        sender_probe_mean_before=0.0,
        cso_after=[],
        cso_drift_after=0.0,
        semantic_drift_after=0.0,
        theme_scores_after={},
        messages=[],
        h_sender_send=None if sender_hidden is None else np.asarray(sender_hidden, dtype=np.float64),
        h_target_pre_end=None if target_hidden is None else np.asarray(target_hidden, dtype=np.float64),
    )


def test_successful_pairs_keep_only_positive_transitions():
    good = _record("good", sender_hidden=[1.0, 0.0], target_hidden=[0.0, 1.0])
    negative = _record("bad-y", y_after=0, sender_hidden=[1.0, 0.0], target_hidden=[0.0, 0.0])
    already = _record("already", target_positive_before=True, sender_hidden=[1.0, 0.0], target_hidden=[0.0, 0.0])
    wrong_hop = _record("hop", target_hop=9, sender_hidden=[1.0, 0.0], target_hidden=[0.0, 0.0])
    missing = _record("missing", sender_hidden=[1.0, 0.0], target_hidden=None)
    duplicate = _record("good", sender_hidden=[9.0, 9.0], target_hidden=[9.0, 9.0])
    chosen = successful_pairs([good, negative, already, wrong_hop, missing, duplicate])
    assert [record.episode_id for record in chosen] == ["good"]
    assert chosen[0].h_sender_send.tolist() == [1.0, 0.0]
    failed = unsuccessful_pairs([good, negative])
    assert [record.episode_id for record in failed] == ["bad-y"]


def test_status_before_is_independent_of_outcome():
    record = _record("split", y_after=1, target_positive_before=False)
    assert record.target_status_before is False
    assert record.y_after == 1
    assert record.h_target_pre_end is None
    record.y_after = 0
    assert record.target_status_before is False


def test_delta_matches_independent_numpy():
    pairs = [
        _record("a", sender_hidden=[2.0, 0.0, 1.0], target_hidden=[0.0, 1.0, 0.0]),
        _record("b", sender_hidden=[0.0, 2.0, 1.0], target_hidden=[1.0, 0.0, 1.0]),
        _record("c", sender_hidden=[1.0, 1.0, 0.0], target_hidden=[1.0, 1.0, 1.0]),
    ]
    delta = mean_delta(pairs)
    manual = np.mean(
        [
            np.array([2.0, 0.0, 1.0]) - np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 2.0, 1.0]) - np.array([1.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]) - np.array([1.0, 1.0, 1.0]),
        ],
        axis=0,
    )
    np.testing.assert_allclose(delta, manual)


def test_empty_epoch_does_not_invent_a_direction():
    memory = SteeringMemory.create(4, np.random.default_rng(0))
    random_before = memory.random_unit.copy()
    memory.update(0, None, None, None, "C")
    assert memory.delta0 is None
    assert memory.cumulative is None
    assert 0 in memory.absent_epochs
    assert memory.applied_vector("B", 1, "unit", 1.0, 1.0) is None
    assert memory.applied_vector("C", 1, "unit", 1.0, 1.0) is None
    assert memory.applied_vector("D", 1, "unit", 1.0, 1.0) is None
    np.testing.assert_allclose(memory.random_unit, random_before)
    assert memory.summary(0)["substituted_random"] is False
    assert memory.summary(0)["N_status"] == "absent_no_successful_pairs"


def test_cumulative_adds_raw_deltas_and_skips_gaps():
    memory = SteeringMemory.create(2, np.random.default_rng(1))
    first = np.array([2.0, 0.0])
    second = np.array([0.0, 4.0])
    memory.update(0, first, None, None, "C")
    memory.update(1, None, None, None, "C")
    memory.update(2, second, None, None, "C")
    np.testing.assert_allclose(memory.cumulative, np.array([2.0, 4.0]))
    np.testing.assert_allclose(memory.cumulative_by_epoch[1], np.array([2.0, 0.0]))
    applied = memory.applied_vector("C", 3, "raw", 0.5, 1.0)
    np.testing.assert_allclose(applied, np.array([1.0, 2.0]))
    unit = scale_direction(memory.cumulative, "unit", beta=2.0, fixed_norm=1.0)
    assert unit is not None
    assert np.linalg.norm(unit) == np.float64(2.0) or abs(float(np.linalg.norm(unit)) - 2.0) < 1e-8


def test_fixed_and_random_are_reproducible_and_matched_in_norm():
    first = SteeringMemory.create(16, np.random.default_rng(5))
    second = SteeringMemory.create(16, np.random.default_rng(5))
    delta = np.linspace(0.5, 1.5, 16)
    first.update(0, delta, None, None, "B")
    second.update(0, delta, None, None, "D")
    fixed = first.applied_vector("B", 1, "unit", 1.5, 1.0)
    random_matched = second.applied_vector("D", 1, "unit", 1.5, 1.0)
    again = SteeringMemory.create(16, np.random.default_rng(5)).random_unit
    np.testing.assert_allclose(second.random_unit, again)
    assert fixed is not None and random_matched is not None
    np.testing.assert_allclose(np.linalg.norm(fixed), np.linalg.norm(random_matched))
    assert abs(float(np.dot(fixed, random_matched) / (np.linalg.norm(fixed) ** 2))) < 0.95


def test_repeated_sender_is_counted_when_one_agent_supplies_two_pairs():
    first = _record("a", sender_hidden=[1.0, 0.0], target_hidden=[0.0, 0.0])
    second = _record("b", sender_hidden=[1.0, 0.0], target_hidden=[0.0, 1.0])
    second.sender_id = first.sender_id
    alone = _record("c", sender_hidden=[0.0, 1.0], target_hidden=[0.0, 0.0])
    from steerchain.pairs import repeated_senders

    assert repeated_senders([first, second, alone]) == {first.sender_id: 2}


def test_shuffle_mean_is_pairing_invariant_and_reproducible():
    pairs = [
        _record("p1", sender_hidden=[3.0, 0.0], target_hidden=[0.0, 0.0]),
        _record("p2", sender_hidden=[0.0, 3.0], target_hidden=[0.0, 1.0]),
    ]
    aligned = mean_delta(pairs)
    broke = False
    for seed in range(30):
        shuffled, alignment = shuffle_control(pairs, np.random.default_rng(seed))
        again, again_alignment = shuffle_control(pairs, np.random.default_rng(seed))
        np.testing.assert_allclose(shuffled, aligned)
        np.testing.assert_allclose(shuffled, again)
        assert alignment == again_alignment
        if alignment is not None and alignment < 0.999:
            broke = True
    assert broke


def test_max_delta_layer_drives_contrastive_slice():
    stack = np.array(
        [
            [1.0, 0.0],
            [0.0, 10.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    assert select_max_delta_layer(stack) == 1
    assert resolve_contrastive_layer(stack, locked=2, previous=None) == 2
    assert resolve_contrastive_layer(stack, locked=None, previous=0) == 0
    assert resolve_contrastive_layer(stack, locked=None, previous=None) == 1
    pairs = [
        _record(
            "a",
            sender_hidden=[[1.0, 0.0], [5.0, 0.0], [1.0, 1.0]],
            target_hidden=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ),
        _record(
            "b",
            sender_hidden=[[0.0, 1.0], [0.0, 5.0], [0.0, 1.0]],
            target_hidden=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ),
    ]
    full = mean_delta(pairs)
    assert full is not None and full.shape == (3, 2)
    layer = select_max_delta_layer(full)
    assert layer == 1
    sliced = mean_delta(pairs, layer=layer)
    np.testing.assert_allclose(sliced, full[layer])
    shuffled, _ = shuffle_control(pairs, np.random.default_rng(0), layer=layer)
    np.testing.assert_allclose(shuffled, sliced)


def test_probe_ba_picks_separating_layer_not_largest_delta():
    # Layer 1: huge Δ for every pair (large ||mean Δ||) but no Y signal.
    # Layer 2: small Δ whose sign tracks acceptance Y.
    labeled = []
    for index, accepted in enumerate([1, 1, 1, 0, 0, 0]):
        sign = 1.0 if accepted else -1.0
        labeled.append(
            _record(
                f"pair-{index}",
                y_after=accepted,
                sender_hidden=[
                    [0.1, 0.0],
                    [10.0, 0.0],
                    [sign, 0.0],
                ],
                target_hidden=[
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            )
        )
    positives = [record for record in labeled if record.y_after == 1]
    stack = mean_delta(positives)
    assert stack is not None
    assert select_max_delta_layer(stack) == 1
    scores = layer_probe_balanced_accuracies(labeled)
    assert select_max_probe_ba_layer(scores) == 2
    assert scores[2] > scores[1]
    assert resolve_contrastive_layer(stack, mode="max_delta") == 1
    assert resolve_contrastive_layer(stack, mode="probe_ba", probe_ba_scores=scores) == 2
    assert eligible_pairs(labeled) == labeled
    # One class only → probe BA unavailable → fall back to max_delta.
    assert resolve_contrastive_layer(stack, mode="probe_ba", probe_ba_scores=[float("nan")] * 3) == 1
