"""Mode A: decoder-direction cosine search against steering vectors."""

from __future__ import annotations

import numpy as np

from steerchain.embeddings import cosine
from steerchain.sae.checkpoint import TopKSAE


def decoder_cosine_scores(direction: np.ndarray, sae: TopKSAE) -> np.ndarray:
    """Return cosine(Δ, d_f) for every feature. Shape (d_sae,)."""
    delta = np.asarray(direction, dtype=np.float64).reshape(-1)
    if delta.shape[0] != sae.d_model:
        raise ValueError(f"Direction dim {delta.shape[0]} != SAE d_model {sae.d_model}")
    dec = sae.decoder_directions()  # (d_model, d_sae)
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm == 0.0:
        return np.zeros(sae.d_sae, dtype=np.float64)
    col_norms = np.linalg.norm(dec, axis=0)
    dots = delta @ dec
    scores = np.zeros(sae.d_sae, dtype=np.float64)
    nonzero = col_norms > 0
    scores[nonzero] = dots[nonzero] / (delta_norm * col_norms[nonzero])
    return scores


def topk_features(
    scores: np.ndarray,
    *,
    k: int,
    positive_only: bool = True,
) -> list[dict]:
    """Rank features by cosine. Default: largest positive scores."""
    scores = np.asarray(scores, dtype=np.float64)
    if positive_only:
        candidates = np.where(scores > 0)[0]
        if candidates.size == 0:
            return []
        order = candidates[np.argsort(scores[candidates])[::-1]]
    else:
        order = np.argsort(scores)[::-1]
    chosen = order[: max(0, int(k))]
    rows = []
    for rank, feature_id in enumerate(chosen, start=1):
        score = float(scores[int(feature_id)])
        rows.append(
            {
                "feature_id": int(feature_id),
                "cosine": score,
                "rank": rank,
                "sign": 0 if score == 0 else (1 if score > 0 else -1),
            }
        )
    return rows


def mode_a_for_direction(
    *,
    direction: np.ndarray,
    sae: TopKSAE,
    ideology_id: str,
    epoch: int | None,
    vector_kind: str,
    decoder_topk: int,
) -> dict:
    scores = decoder_cosine_scores(direction, sae)
    top = topk_features(scores, k=decoder_topk, positive_only=True)
    return {
        "ideology_id": ideology_id,
        "epoch": epoch,
        "vector_kind": vector_kind,
        "sae_repo_id": sae.repo_id,
        "transformer_layer_k": sae.layer_k,
        "activation_site": sae.activation_site,
        "direction_norm": float(np.linalg.norm(direction)),
        "n_features": sae.d_sae,
        "topk": top,
        "scores": scores,
        "disclaimer": (
            "Geometric proximity of decoder directions only. "
            "Does not imply semantic match or causal effect on propagation."
        ),
    }


def score_matrix(directions: dict[str, np.ndarray], sae: TopKSAE) -> tuple[list[str], np.ndarray]:
    """Rows=ideologies, columns=features."""
    ids = sorted(directions)
    if not ids:
        return [], np.zeros((0, sae.d_sae), dtype=np.float64)
    rows = [decoder_cosine_scores(directions[ideology_id], sae) for ideology_id in ids]
    return ids, np.stack(rows, axis=0)
