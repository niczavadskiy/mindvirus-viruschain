"""Markdown report for SAE analysis (correlational only)."""

from __future__ import annotations

from pathlib import Path


def write_sae_report(
    out_dir: Path,
    summary: dict,
    *,
    consensus: list[dict] | None = None,
    shared: list[int] | None = None,
    clusters: dict | None = None,
) -> None:
    lines = [
        "# SAE analysis report",
        "",
        "This report is **correlational**. Geometric proximity of decoder directions "
        "or SAE activations does **not** establish that a feature causes idea propagation.",
        "",
        f"- status: `{summary.get('status')}`",
        f"- causal outputs separated: `{summary.get('causal_separated', True)}`",
        f"- run_causal: `{summary.get('run_causal', False)}`",
        "",
        "## Compatibility",
        "",
        "```json",
        _json(summary.get("compatibility", {})),
        "```",
        "",
    ]
    if summary.get("status") == "skipped_incompatible":
        lines.extend(
            [
                "## Skipped",
                "",
                summary.get("note", "Incompatible SAE was not applied."),
                "",
            ]
        )
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return

    meta = summary.get("sae_metadata") or {}
    lines.extend(
        [
            "## SAE checkpoint",
            "",
            f"- repo: `{meta.get('repo_id')}`",
            f"- revision: `{meta.get('revision')}`",
            f"- architecture: `{meta.get('architecture')}` TopK={meta.get('top_k')}",
            f"- layer k: `{meta.get('transformer_layer_k')}` site=`{meta.get('activation_site')}`",
            f"- d_model={meta.get('d_model')} d_sae={meta.get('d_sae')}",
            f"- checksum: `{meta.get('checksum')}`",
            f"- source: `{meta.get('source_path')}`",
            "",
            "## Pool summary",
            "",
            f"- ideologies with Δ+: {summary.get('n_ideologies_with_delta_plus')}",
            f"- Mode A rows: {summary.get('n_mode_a_rows')}",
            f"- Mode B rows: {summary.get('n_mode_b_rows')}",
            f"- intersection Top-k size: {summary.get('shared_intersection_size')}",
            f"- consensus size: {summary.get('consensus_size')}",
            "",
        ]
    )
    if shared is not None:
        lines.extend(["### Intersection Top-k", "", ", ".join(str(x) for x in shared) or "(empty)", ""])
    if consensus:
        lines.extend(["### Frequency consensus", ""])
        for row in consensus[:20]:
            lines.append(f"- feature {row['feature_id']}: C_f={row['C_f']:.3f}")
        lines.append("")
    if clusters:
        lines.extend(
            [
                "### Clusters",
                "",
                f"- method: `{clusters.get('method')}`",
                f"- n_clusters: {clusters.get('n_clusters')}",
                f"- labels: `{clusters.get('labels')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Hypotheses for causal follow-up",
            "",
            "These are candidates only. See `hypotheses_for_causal.json` and `../causal_sae/`.",
            "",
        ]
    )
    for item in summary.get("hypotheses") or []:
        lines.append(f"- feature {item.get('feature_id')} ({item.get('source')}): {item.get('hypothesis')}")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _json(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2)
