"""Stability diagnostics for Mode A rankings across epochs and seeds."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from steerchain.sae.mode_a import topk_features


def rank_map(scores: np.ndarray, k: int) -> dict[int, int]:
    return {row["feature_id"]: row["rank"] for row in topk_features(scores, k=k, positive_only=True)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def stability_report(
    *,
    by_epoch: dict[tuple[str, int], np.ndarray],
    by_seed: dict[tuple[str, int], np.ndarray],
    decoder_topk: int,
    consensus_threshold: float,
) -> dict:
    """Compare Top-k sets across epochs and seeds per ideology."""
    epoch_groups: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (ideology_id, epoch), scores in sorted(by_epoch.items()):
        epoch_groups[ideology_id].append((epoch, scores))

    seed_groups: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (ideology_id, seed), scores in sorted(by_seed.items()):
        seed_groups[ideology_id].append((seed, scores))

    per_ideology = {}
    for ideology_id, items in epoch_groups.items():
        sets = [set(rank_map(scores, decoder_topk)) for _, scores in items]
        pairwise = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                pairwise.append(jaccard(sets[i], sets[j]))
        # Feature frequency across epochs.
        counts: dict[int, int] = {}
        for s in sets:
            for feature_id in s:
                counts[feature_id] = counts.get(feature_id, 0) + 1
        n_epochs = max(len(sets), 1)
        frequent = [
            {"feature_id": fid, "epoch_freq": count / n_epochs}
            for fid, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if count / n_epochs >= consensus_threshold
        ]
        per_ideology[ideology_id] = {
            "n_epochs": len(items),
            "mean_topk_jaccard_across_epochs": float(np.mean(pairwise)) if pairwise else None,
            "stable_features": frequent,
        }

    seed_summary = {}
    for ideology_id, items in seed_groups.items():
        sets = [set(rank_map(scores, decoder_topk)) for _, scores in items]
        pairwise = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                pairwise.append(jaccard(sets[i], sets[j]))
        seed_summary[ideology_id] = {
            "n_seeds": len(items),
            "mean_topk_jaccard_across_seeds": float(np.mean(pairwise)) if pairwise else None,
        }

    return {
        "decoder_topk": decoder_topk,
        "consensus_threshold": consensus_threshold,
        "by_ideology_epochs": per_ideology,
        "by_ideology_seeds": seed_summary,
        "note": (
            "Stability is correlational. Dependent episodes that share senders "
            "inflate agreement; bootstrap over independent branches is preferred when available."
        ),
    }


def bootstrap_topk_frequency(
    deltas: list[np.ndarray],
    score_fn,
    *,
    k: int,
    n_bootstrap: int = 50,
    seed: int = 0,
) -> list[dict]:
    """Episode-level bootstrap of Top-k membership. Treats pairs as exchangeable."""
    if not deltas:
        return []
    rng = np.random.default_rng(seed)
    n = len(deltas)
    counts: dict[int, int] = {}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = [deltas[i] for i in idx]
        mean = np.mean(np.stack(sample, axis=0), axis=0)
        scores = score_fn(mean)
        for row in topk_features(scores, k=k, positive_only=True):
            counts[row["feature_id"]] = counts.get(row["feature_id"], 0) + 1
    ranked = [
        {"feature_id": fid, "bootstrap_freq": count / n_bootstrap}
        for fid, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return ranked
