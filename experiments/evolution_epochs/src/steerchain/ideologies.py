"""Published ideological payloads, short descriptions, and probe questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from steerchain.config import ExperimentConfig


@dataclass(frozen=True)
class ProbeQuestion:
    question: str
    scoring: str


@dataclass(frozen=True)
class Ideology:
    ideology_id: str
    short: str
    payload: str
    questions: tuple[ProbeQuestion, ...]


def load_ideology(data_dir: Path, ideology_id: str) -> Ideology:
    path = data_dir / f"{ideology_id}.json"
    if not path.is_file():
        known = sorted(p.stem for p in data_dir.glob("*.json") if p.stem != "manifest")
        raise FileNotFoundError(f"Unknown ideology {ideology_id!r}. Known: {known}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = tuple(
        ProbeQuestion(question=item["question"], scoring=item.get("scoring", ""))
        for item in raw["ideology_questions"]
    )
    return Ideology(
        ideology_id=ideology_id,
        short=raw["ideology"].strip(),
        payload=raw["payload"].strip(),
        questions=questions,
    )


def load_default_soul(data_dir: Path) -> str:
    return (data_dir / "default_soul.md").read_text(encoding="utf-8").strip()


def load_manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))


def ideologies_for(config: ExperimentConfig) -> list[Ideology]:
    data_dir = config.resolved_data_dir()
    return [load_ideology(data_dir, ideology_id) for ideology_id in config.ideologies]
