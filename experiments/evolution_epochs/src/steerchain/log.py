"""Persist episode lineage and the per-layer hidden-state snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from steerchain.records import EpisodeRecord


class RunLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.states = self.run_dir / "states"
        self.steering = self.run_dir / "steering"
        self.states.mkdir(parents=True, exist_ok=True)
        self.steering.mkdir(parents=True, exist_ok=True)

    def write_config(self, payload: dict) -> None:
        (self.run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_episode(self, record: EpisodeRecord) -> None:
        if record.h_sender_send is not None:
            path = self.states / f"{record.episode_id}_sender.npy"
            np.save(path, record.h_sender_send)
            record.hidden_sender_path = str(path)
        if record.h_target_pre_end is not None:
            path = self.states / f"{record.episode_id}_target.npy"
            np.save(path, record.h_target_pre_end)
            record.hidden_target_path = str(path)
        with (self.run_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_public_dict(), ensure_ascii=False) + "\n")

    def write_steering_epoch(self, epoch: int, memory, applied: np.ndarray | None, extra: dict | None = None) -> dict:
        summary = memory.summary(epoch)
        summary["applied_norm"] = None if applied is None else float(np.linalg.norm(applied))
        summary["applied"] = applied is not None
        if extra:
            summary.update(extra)
        delta = memory.delta_by_epoch.get(epoch)
        negative = memory.delta_negative_by_epoch.get(epoch)
        cumulative = memory.cumulative_by_epoch.get(epoch)
        shuffled = memory.shuffled_by_epoch.get(epoch)
        self._save(f"epoch_{epoch}_delta.npy", delta)
        self._save(f"epoch_{epoch}_delta_negative.npy", negative)
        self._save(f"epoch_{epoch}_cumulative.npy", cumulative)
        self._save(f"epoch_{epoch}_shuffled.npy", shuffled)
        self._save(f"epoch_{epoch}_applied.npy", applied)
        path = self.steering / "summary.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        return summary

    def _save(self, name: str, vector: np.ndarray | None) -> None:
        if vector is None:
            return
        np.save(self.steering / name, np.asarray(vector, dtype=np.float64))
