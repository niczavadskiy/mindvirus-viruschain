"""Tables and figures for conditions A–D."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_FLAT = (
    "ideology_id",
    "condition",
    "seed",
    "epoch",
    "p_final",
    "p_all",
    "r_into_final",
    "retention",
    "semantic_drift_final",
    "cso_drift_final",
    "n_positive_pairs",
    "n_negative_pairs",
    "shuffle_alignment",
)


def _series(rows: list[dict]) -> dict[tuple[str, str, int], list[dict]]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["ideology_id"], row["condition"], row["seed"])
        grouped[key].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: item["epoch"])
    return grouped


def _xy(items: list[dict], field: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for item in items:
        value = item.get(field)
        if field.startswith("steering."):
            value = (item.get("steering") or {}).get(field.split(".", 1)[1])
        if value is None:
            continue
        xs.append(int(item["epoch"]))
        ys.append(float(value))
    return xs, ys


def write_report(run_dir: Path, rows: list[dict]) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(run_dir / "comparison.csv", rows)
    if not rows:
        return
    grouped = _series(rows)
    ideologies = sorted({row["ideology_id"] for row in rows})
    _plot_by_epoch(figures / "p_final_by_epoch.png", grouped, ideologies, "p_final", "Share of carriers on the final hop")
    _plot_by_epoch(figures / "p_all_by_epoch.png", grouped, ideologies, "p_all", "Mean share of carriers across hops")
    _plot_by_epoch(
        figures / "semantic_drift_by_epoch.png",
        grouped,
        ideologies,
        "semantic_drift_final",
        "Semantic drift on the final hop",
    )
    _plot_by_epoch(
        figures / "cso_drift_by_epoch.png",
        grouped,
        ideologies,
        "cso_drift_final",
        "CSO drift on the final hop",
    )
    _plot_by_epoch(
        figures / "steering_norm_by_epoch.png",
        grouped,
        ideologies,
        "steering.delta_norm",
        "Norm of the epoch steering delta",
    )
    _plot_by_epoch(
        figures / "steering_cosine_by_epoch.png",
        grouped,
        ideologies,
        "steering.cosine_delta_previous",
        "Cosine of successive steering deltas",
    )
    _plot_transitions(figures / "transition_by_hop.png", rows, ideologies)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(_FLAT) + ["delta_norm", "applied_norm", "cosine_delta_previous", "cumulative_norm"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            steering = row.get("steering") or {}
            flat = {key: row.get(key) for key in _FLAT}
            flat["delta_norm"] = steering.get("delta_norm")
            flat["applied_norm"] = steering.get("applied_norm")
            flat["cosine_delta_previous"] = steering.get("cosine_delta_previous")
            flat["cumulative_norm"] = steering.get("cumulative_norm")
            writer.writerow(flat)


def _plot_by_epoch(path: Path, grouped, ideologies: list[str], field: str, title: str) -> None:
    fig, axes = plt.subplots(1, len(ideologies), figsize=(5 * len(ideologies), 4), squeeze=False)
    for axis, ideology in zip(axes[0], ideologies, strict=True):
        for (ideology_id, condition, seed), items in sorted(grouped.items()):
            if ideology_id != ideology:
                continue
            xs, ys = _xy(items, field)
            if not xs:
                continue
            axis.plot(xs, ys, marker="o", label=f"{condition} seed {seed}")
        axis.set_title(ideology)
        axis.set_xlabel("epoch")
        axis.set_ylabel(field)
        _legend_if_labeled(axis)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _legend_if_labeled(axis) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, fontsize=8)


def _plot_transitions(path: Path, rows: list[dict], ideologies: list[str]) -> None:
    fig, axes = plt.subplots(1, len(ideologies), figsize=(5 * len(ideologies), 4), squeeze=False)
    last_epoch = max(row["epoch"] for row in rows)
    for axis, ideology in zip(axes[0], ideologies, strict=True):
        relevant = [
            row for row in rows if row["ideology_id"] == ideology and row["epoch"] == last_epoch
        ]
        for row in relevant:
            hops = []
            rates = []
            for hop, payload in sorted(row["transition_by_hop"].items(), key=lambda item: int(item[0])):
                if payload["r"] is None:
                    continue
                hops.append(int(hop))
                rates.append(payload["r"])
            if hops:
                axis.plot(hops, rates, marker="o", label=f"{row['condition']} seed {row['seed']}")
        axis.set_title(ideology)
        axis.set_xlabel("target hop")
        axis.set_ylabel("transition success")
        _legend_if_labeled(axis)
    fig.suptitle(f"Transition success at epoch {last_epoch}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
