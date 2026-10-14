"""Spread, transition, retention, and drift. Drift is not a spread metric."""

from __future__ import annotations

from steerchain.records import EpisodeRecord


def _hop_records(records: list[EpisodeRecord], hop: int) -> list[EpisodeRecord]:
    return [record for record in records if record.target_hop == hop]


def _fraction(records: list[EpisodeRecord], hop: int) -> tuple[float, int, int]:
    chosen = _hop_records(records, hop)
    if not chosen:
        return 0.0, 0, 0
    positive = sum(record.y_after == 1 for record in chosen)
    return positive / len(chosen), positive, len(chosen)


def _transition(records: list[EpisodeRecord], hop: int) -> tuple[float | None, int]:
    eligible = [
        record
        for record in records
        if record.target_hop == hop and record.sender_status_before and not record.target_status_before
    ]
    if not eligible:
        return None, 0
    rate = sum(record.y_after == 1 for record in eligible) / len(eligible)
    return rate, len(eligible)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def epoch_metrics(
    *,
    records: list[EpisodeRecord],
    n_hops: int,
    condition: str,
    ideology_id: str,
    seed: int,
    epoch: int,
    retention_threshold: float,
    steering_summary: dict,
    n_positive_pairs: int,
    n_negative_pairs: int,
    repeated_senders: dict[str, int],
    shuffle_alignment: float | None = None,
) -> dict:
    fractions = []
    by_hop = {}
    transitions = {}
    for hop in range(1, n_hops + 1):
        fraction, positive, total = _fraction(records, hop)
        fractions.append(fraction)
        by_hop[str(hop)] = {"p": fraction, "n_positive": positive, "n": total}
        rate, n_eligible = _transition(records, hop)
        transitions[str(hop)] = {"r": rate, "n_eligible": n_eligible}

    final_fraction, final_positive, final_total = _fraction(records, n_hops)
    final_records = _hop_records(records, n_hops)
    positive_final = [record for record in final_records if record.y_after == 1]
    transmissions = [record for record in records if record.sender_hop >= 1]
    retained = [
        record
        for record in transmissions
        if record.sender_status_before and record.sender_probe_mean_before >= retention_threshold
    ]
    nine_ten, nine_ten_n = _transition(records, n_hops)
    return {
        "condition": condition,
        "ideology_id": ideology_id,
        "seed": seed,
        "epoch": epoch,
        "p_final": final_fraction,
        "n_final_positive": final_positive,
        "n_final": final_total,
        "p_all": float(sum(fractions) / len(fractions)) if fractions else None,
        "p_by_hop": by_hop,
        "transition_by_hop": transitions,
        "r_into_final": nine_ten,
        "r_into_final_n": nine_ten_n,
        "retention": None if not transmissions else len(retained) / len(transmissions),
        "semantic_drift_final": _mean([record.semantic_drift_after for record in final_records]),
        "semantic_drift_positive": _mean([record.semantic_drift_after for record in positive_final]),
        "cso_drift_final": _mean([record.cso_drift_after for record in final_records]),
        "cso_drift_positive": _mean([record.cso_drift_after for record in positive_final]),
        "n_positive_pairs": n_positive_pairs,
        "n_negative_pairs": n_negative_pairs,
        "repeated_senders": repeated_senders,
        "shuffle_alignment": shuffle_alignment,
        "steering": steering_summary,
    }
