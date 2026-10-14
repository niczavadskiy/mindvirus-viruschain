"""Pair selection. Only real episodes with Y=1 enter the main steering vector."""

from __future__ import annotations

from collections import Counter

from steerchain.records import EpisodeRecord


def _complete(record: EpisodeRecord) -> bool:
    return record.h_sender_send is not None and record.h_target_pre_end is not None


def _eligible(record: EpisodeRecord, spreader_hop: int, target_hop: int) -> bool:
    return (
        record.target_hop == target_hop
        and record.sender_hop == spreader_hop
        and record.sender_status_before
        and not record.target_status_before
        and _complete(record)
    )


def _unique(records: list[EpisodeRecord]) -> list[EpisodeRecord]:
    seen: set[str] = set()
    unique: list[EpisodeRecord] = []
    for record in records:
        if record.episode_id in seen:
            continue
        seen.add(record.episode_id)
        unique.append(record)
    return unique


def eligible_pairs(
    records: list[EpisodeRecord],
    spreader_hop: int = 9,
    target_hop: int = 10,
) -> list[EpisodeRecord]:
    """H+→H− transitions with both hidden states, regardless of Y."""
    chosen = []
    for record in _unique(records):
        if _eligible(record, spreader_hop, target_hop):
            chosen.append(record)
    return chosen


def successful_pairs(
    records: list[EpisodeRecord],
    spreader_hop: int = 9,
    target_hop: int = 10,
) -> list[EpisodeRecord]:
    """Sender was H+ on the spreader hop, target was H- before, and Y=1 after."""
    return [record for record in eligible_pairs(records, spreader_hop, target_hop) if record.y_after == 1]


def repeated_senders(pairs: list[EpisodeRecord]) -> dict[str, int]:
    counts = Counter(record.sender_id for record in pairs)
    return {sender_id: count for sender_id, count in counts.items() if count > 1}


def unsuccessful_pairs(
    records: list[EpisodeRecord],
    spreader_hop: int = 9,
    target_hop: int = 10,
) -> list[EpisodeRecord]:
    """Diagnostic pairs only. These never enter the injected direction."""
    return [record for record in eligible_pairs(records, spreader_hop, target_hop) if record.y_after == 0]
