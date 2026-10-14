"""Load run artifacts for offline SAE analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_runs(run_dir: Path) -> list[dict]:
    """Find ideology/condition/seed subruns under a root run directory."""
    root = Path(run_dir)
    found = []
    for episodes in root.glob("*/*/seed_*/episodes.jsonl"):
        seed_dir = episodes.parent
        condition_dir = seed_dir.parent
        ideology_dir = condition_dir.parent
        found.append(
            {
                "ideology_id": ideology_dir.name,
                "condition": condition_dir.name,
                "seed": int(seed_dir.name.split("_", 1)[1]),
                "path": seed_dir,
                "episodes": _read_jsonl(episodes),
                "config": _read_json(seed_dir / "config.json"),
                "steering_summary": _read_jsonl(seed_dir / "steering" / "summary.jsonl"),
            }
        )
    return found


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_vector(path: Path | None) -> np.ndarray | None:
    if path is None or not Path(path).is_file():
        return None
    return np.load(path)


def load_episode_hiddens(episode: dict, seed_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    sender = episode.get("hidden_sender_path")
    target = episode.get("hidden_target_path")
    if sender is None:
        sender = seed_dir / "states" / f"{episode['episode_id']}_sender.npy"
    if target is None:
        target = seed_dir / "states" / f"{episode['episode_id']}_target.npy"
    return load_vector(Path(sender) if sender else None), load_vector(Path(target) if target else None)


def eligible_pair(episode: dict, spreader_hop: int = 9, target_hop: int = 10) -> bool:
    return (
        episode.get("target_hop") == target_hop
        and episode.get("sender_hop") == spreader_hop
        and bool(episode.get("sender_status_before"))
        and not bool(episode.get("target_status_before"))
    )


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_score_table(path: Path, rows: list[dict]) -> None:
    """JSONL table (Parquet optional if pyarrow present)."""
    write_jsonl(path.with_suffix(".jsonl"), rows)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return
    if not rows:
        return
    # Flatten topk for parquet companion.
    flat = []
    for row in rows:
        base = {key: value for key, value in row.items() if key not in {"topk", "scores"}}
        for item in row.get("topk", []):
            flat.append({**base, **item})
    if flat:
        table = pa.Table.from_pylist(flat)
        pq.write_table(table, path.with_suffix(".parquet"))
