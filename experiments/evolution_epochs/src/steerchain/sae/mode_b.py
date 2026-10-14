"""Mode B: encode saved hidden states with the actual TopK SAE forward."""

from __future__ import annotations

import numpy as np

from steerchain.estimator import at_layer
from steerchain.sae.checkpoint import TopKSAE


def encode_state(
    hidden: np.ndarray,
    sae: TopKSAE,
    *,
    layer: int,
    reconstruct: bool = True,
) -> dict:
    h = at_layer(hidden, layer)
    acts = sae.encode(h)
    active = np.nonzero(acts)[0]
    row = {
        "transformer_layer_k": layer,
        "capture_site": None,
        "n_nonzero": int(active.size),
        "active_feature_ids": [int(i) for i in active.tolist()],
        "active_values": [float(acts[i]) for i in active.tolist()],
        "hidden_norm": float(np.linalg.norm(h)),
        "activation_l1": float(np.sum(np.abs(acts))),
    }
    if reconstruct:
        recon = sae.decode(acts)
        row["recon_norm"] = float(np.linalg.norm(recon))
        row["recon_mse"] = float(np.mean((recon - h) ** 2))
    return row


def encode_pair_record(
    *,
    episode: dict,
    sender_hidden: np.ndarray,
    target_hidden: np.ndarray,
    sae: TopKSAE,
    layer: int,
) -> dict:
    sender = encode_state(sender_hidden, sae, layer=layer)
    sender["capture_site"] = "t_send"
    sender["role"] = "sender_Hplus_hop9"
    target = encode_state(target_hidden, sae, layer=layer)
    target["capture_site"] = "t_pre_end"
    target["role"] = "target_Hminus_before"
    return {
        "episode_id": episode.get("episode_id"),
        "ideology_id": episode.get("ideology_id"),
        "epoch": episode.get("epoch"),
        "condition": episode.get("condition"),
        "seed": episode.get("seed"),
        "sender_id": episode.get("sender_id"),
        "target_id": episode.get("target_id"),
        "sender_status_before": episode.get("sender_status_before"),
        "target_status_before": episode.get("target_status_before"),
        "y_after": episode.get("y_after"),
        "post_success_target_state": None,
        "post_success_note": (
            "H+_10 after success is not captured by the default protocol; "
            "field left null."
        ),
        "sender": sender,
        "target": target,
        "disclaimer": (
            "SAE activations are descriptive. They do not establish causal influence."
        ),
    }
