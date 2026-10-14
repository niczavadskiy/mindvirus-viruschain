"""Episode records. Hop index and transformer layer k stay distinct."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeRecord:
    episode_id: str
    epoch: int
    condition: str
    ideology_id: str
    seed: int
    model_id: str
    transformer_layer_k: int
    sender_id: str
    target_id: str
    sender_hop: int
    target_hop: int
    sender_parent_episode_id: str | None
    sender_status_before: bool
    target_status_before: bool
    y_after: int
    cosine_ref_after: float
    cosine_payload_after: float
    judge_after: int
    probe_scores_after: list[float]
    probe_mean_after: float
    sender_probe_mean_before: float
    cso_after: list[float]
    cso_drift_after: float
    semantic_drift_after: float
    theme_scores_after: dict[str, int]
    messages: list[dict]
    h_sender_send: np.ndarray | None = None
    h_target_pre_end: np.ndarray | None = None
    hidden_sender_path: str | None = None
    hidden_target_path: str | None = None
    started_at: str = ""
    ended_at: str = ""
    capture_sender: str = "t_send"
    capture_target: str = "t_pre_end"
    repeated_sender: bool = False
    extra: dict = field(default_factory=dict)

    def to_public_dict(self) -> dict:
        """JSON-safe view. Hidden states are paths, not inline arrays."""
        return {
            "episode_id": self.episode_id,
            "epoch": self.epoch,
            "condition": self.condition,
            "ideology_id": self.ideology_id,
            "seed": self.seed,
            "model_id": self.model_id,
            "transformer_layer_k": self.transformer_layer_k,
            "sender_id": self.sender_id,
            "target_id": self.target_id,
            "sender_hop": self.sender_hop,
            "target_hop": self.target_hop,
            "sender_parent_episode_id": self.sender_parent_episode_id,
            "sender_status_before": self.sender_status_before,
            "target_status_before": self.target_status_before,
            "y_after": self.y_after,
            "cosine_ref_after": self.cosine_ref_after,
            "cosine_payload_after": self.cosine_payload_after,
            "judge_after": self.judge_after,
            "probe_scores_after": self.probe_scores_after,
            "probe_mean_after": self.probe_mean_after,
            "sender_probe_mean_before": self.sender_probe_mean_before,
            "cso_after": self.cso_after,
            "cso_drift_after": self.cso_drift_after,
            "semantic_drift_after": self.semantic_drift_after,
            "theme_scores_after": self.theme_scores_after,
            "messages": self.messages,
            "hidden_sender_path": self.hidden_sender_path,
            "hidden_target_path": self.hidden_target_path,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "capture_sender": self.capture_sender,
            "capture_target": self.capture_target,
            "repeated_sender": self.repeated_sender,
        }
