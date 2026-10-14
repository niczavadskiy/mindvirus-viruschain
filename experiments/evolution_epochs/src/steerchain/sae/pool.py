"""Shared SAE-feature discovery across a pool of ideologies."""

from __future__ import annotations

import numpy as np

from steerchain.embeddings import cosine
from steerchain.sae.mode_a import topk_features


def intersection_topk(score_rows: dict[str, np.ndarray], k: int) -> list[int]:
    sets = []
    for scores in score_rows.values():
        top = topk_features(scores, k=k, positive_only=True)
        sets.append({row["feature_id"] for row in top})
    if not sets:
        return []
    shared = set.intersection(*sets)
    return sorted(shared)


def consensus_topk(
    score_rows: dict[str, np.ndarray],
    *,
    k: int,
    threshold: float,
) -> list[dict]:
    n = len(score_rows)
    if n == 0:
        return []
    counts: dict[int, int] = {}
    for scores in score_rows.values():
        for row in topk_features(scores, k=k, positive_only=True):
            counts[row["feature_id"]] = counts.get(row["feature_id"], 0) + 1
    ranked = []
    for feature_id, count in counts.items():
        cf = count / n
        if cf >= threshold:
            ranked.append(
                {
                    "feature_id": int(feature_id),
                    "C_f": float(cf),
                    "n_ideas": int(count),
                    "n_pool": int(n),
                }
            )
    ranked.sort(key=lambda item: (-item["C_f"], item["feature_id"]))
    return ranked


def specific_features(
    score_rows: dict[str, np.ndarray],
    *,
    k: int,
    shared: set[int],
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for ideology_id, scores in score_rows.items():
        top = {row["feature_id"] for row in topk_features(scores, k=k, positive_only=True)}
        out[ideology_id] = sorted(top - shared)
    return out


def agglomerative_clusters(
    ideology_ids: list[str],
    matrix: np.ndarray,
    *,
    n_clusters: int | None = None,
) -> dict:
    """Average-linkage clustering on cosine distance of SAE-score vectors.

    Signs are preserved in the score vectors, so opposite directions stay dissimilar.
    """
    n = len(ideology_ids)
    if n == 0:
        return {
            "method": "agglomerative_average_cosine",
            "n_clusters": 0,
            "labels": {},
            "distance_matrix": [],
            "parameters": {},
        }
    target = 1 if n_clusters is None else max(1, min(int(n_clusters), n))
    # Start with each idea alone.
    clusters = [{i} for i in range(n)]
    # Precompute distances.
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine(matrix[i], matrix[j])
            # Cosine can be negative; distance in [0, 2].
            d = 1.0 - sim
            dist[i, j] = dist[j, i] = d

    def average_link(a: set[int], b: set[int]) -> float:
        total = 0.0
        count = 0
        for i in a:
            for j in b:
                total += dist[i, j]
                count += 1
        return total / max(count, 1)

    while len(clusters) > target:
        best = None
        best_d = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = average_link(clusters[i], clusters[j])
                if d < best_d:
                    best_d = d
                    best = (i, j)
        assert best is not None
        i, j = best
        merged = clusters[i] | clusters[j]
        clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
        clusters.append(merged)

    labels = {}
    for label, members in enumerate(clusters):
        for idx in members:
            labels[ideology_ids[idx]] = label

    return {
        "method": "agglomerative_average_cosine",
        "n_clusters": len(clusters),
        "labels": labels,
        "distance_matrix": dist.tolist(),
        "parameters": {"linkage": "average", "metric": "cosine_distance", "n_clusters": target},
        "note": "Opposite SAE-score directions remain dissimilar because cosine retains sign.",
    }
