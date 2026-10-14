"""Orchestrate SAE Modes A/B, pool search, stability, and reporting."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from steerchain.config import ExperimentConfig, SAEConfig
from steerchain.embeddings import cosine
from steerchain.estimator import at_layer
from steerchain.sae.checkpoint import TopKSAE, load_topk_sae
from steerchain.sae.compat import check_compatibility
from steerchain.sae.io import (
    discover_runs,
    eligible_pair,
    load_episode_hiddens,
    load_vector,
    write_json,
    write_jsonl,
    write_score_table,
)
from steerchain.sae.mode_a import decoder_cosine_scores, mode_a_for_direction, score_matrix
from steerchain.sae.mode_b import encode_pair_record
from steerchain.sae.pool import (
    agglomerative_clusters,
    consensus_topk,
    intersection_topk,
    specific_features,
)
from steerchain.sae.report import write_sae_report
from steerchain.sae.stability import bootstrap_topk_frequency, stability_report


def run_sae_analysis(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    sae: TopKSAE | None = None,
    residual_dim: int | None = None,
) -> dict:
    """Analyze an existing experiment run directory. Never mutates steering vectors."""
    root = Path(run_dir)
    out = root / "sae"
    out.mkdir(parents=True, exist_ok=True)
    sae_cfg = config.sae

    runs = discover_runs(root)
    if residual_dim is None:
        residual_dim = _infer_residual_dim(runs, sae_cfg.transformer_layer_k)

    layer_k = (
        sae_cfg.transformer_layer_k
        if sae_cfg.transformer_layer_k is not None
        else config.transformer_layer_k
    )
    allow_synthetic = bool(sae is not None and str(getattr(sae, "repo_id", "")).startswith("synthetic://"))
    if sae is not None and sae.repo_id.startswith("synthetic://"):
        # Align config dims for synthetic offline tests.
        sae_cfg = SAEConfig(
            **{
                **sae_cfg.__dict__,
                "repo_id": sae.repo_id,
                "d_model": sae.d_model,
                "d_sae": sae.d_sae,
                "top_k": sae.top_k,
                "transformer_layer_k": sae.layer_k,
            }
        )
        layer_k = sae.layer_k
        allow_synthetic = True

    compat = check_compatibility(
        host_model_id=config.model_id,
        sae=sae_cfg,
        residual_dim=residual_dim,
        transformer_layer_k=layer_k,
        allow_synthetic=allow_synthetic,
    )
    write_json(out / "compatibility.json", compat.to_dict())
    if not compat.compatible:
        report = {
            "status": "skipped_incompatible",
            "compatibility": compat.to_dict(),
            "causal_separated": True,
            "note": "Incompatible SAE was not applied. No Mode A/B artifacts written.",
        }
        write_json(out / "summary.json", report)
        write_sae_report(out, report)
        if sae_cfg.run_causal:
            _write_causal_placeholder(root / "causal_sae")
        return report

    if sae is None:
        local = getattr(sae_cfg, "local_dir", None)
        sae = load_topk_sae(
            repo_id=sae_cfg.repo_id,
            layer_k=int(layer_k),
            top_k=sae_cfg.top_k,
            local_dir=local,
            revision=getattr(sae_cfg, "revision", None),
        )
    write_json(out / "sae_metadata.json", sae.metadata())

    # Collect pair deltas and directions.
    pos_deltas: dict[str, list[np.ndarray]] = defaultdict(list)
    neg_deltas: dict[str, list[np.ndarray]] = defaultdict(list)
    mode_b_rows: list[dict] = []
    mode_a_rows: list[dict] = []
    epoch_scores: dict[tuple[str, int], np.ndarray] = {}
    seed_scores: dict[tuple[str, int], np.ndarray] = {}
    epoch_delta_vectors: dict[tuple[str, int], np.ndarray] = {}

    for sub in runs:
        for episode in sub["episodes"]:
            if not eligible_pair(
                episode,
                config.pair_spreader_hop,
                config.pair_target_hop,
            ):
                continue
            sender_h, target_h = load_episode_hiddens(episode, sub["path"])
            if sender_h is None or target_h is None:
                continue
            try:
                delta = at_layer(sender_h, int(layer_k)) - at_layer(target_h, int(layer_k))
            except ValueError:
                continue
            if episode.get("y_after") == 1:
                pos_deltas[sub["ideology_id"]].append(delta)
                if sae_cfg.run_mode_b:
                    mode_b_rows.append(
                        encode_pair_record(
                            episode=episode,
                            sender_hidden=sender_h,
                            target_hidden=target_h,
                            sae=sae,
                            layer=int(layer_k),
                        )
                    )
            elif episode.get("y_after") == 0:
                neg_deltas[sub["ideology_id"]].append(delta)

        # Per-epoch saved Δ if present.
        for summary in sub["steering_summary"]:
            epoch = int(summary["epoch"])
            delta_path = sub["path"] / "steering" / f"epoch_{epoch}_delta.npy"
            vector = load_vector(delta_path)
            if vector is None:
                continue
            try:
                direction = at_layer(vector, int(layer_k)) if np.asarray(vector).ndim == 2 else np.asarray(vector)
            except ValueError:
                continue
            if direction.shape[-1] != sae.d_model:
                continue
            result = mode_a_for_direction(
                direction=direction,
                sae=sae,
                ideology_id=sub["ideology_id"],
                epoch=epoch,
                vector_kind="delta_epoch",
                decoder_topk=sae_cfg.decoder_topk,
            )
            epoch_scores[(sub["ideology_id"], epoch)] = result["scores"]
            epoch_delta_vectors[(sub["ideology_id"], epoch)] = direction
            mode_a_rows.append(_public_mode_a(result))

            cum_path = sub["path"] / "steering" / f"epoch_{epoch}_cumulative.npy"
            cum = load_vector(cum_path)
            if cum is not None:
                try:
                    cum_dir = at_layer(cum, int(layer_k)) if np.asarray(cum).ndim == 2 else np.asarray(cum)
                except ValueError:
                    cum_dir = None
                if cum_dir is not None and cum_dir.shape[-1] == sae.d_model:
                    cum_result = mode_a_for_direction(
                        direction=cum_dir,
                        sae=sae,
                        ideology_id=sub["ideology_id"],
                        epoch=epoch,
                        vector_kind="cumulative_D",
                        decoder_topk=sae_cfg.decoder_topk,
                    )
                    mode_a_rows.append(_public_mode_a(cum_result))

    # Aggregate Δ+ / Δ- per ideology.
    directions_plus: dict[str, np.ndarray] = {}
    directions_minus: dict[str, np.ndarray] = {}
    contrast_rows: list[dict] = []
    for ideology_id, deltas in pos_deltas.items():
        if not deltas:
            continue
        delta_plus = np.mean(np.stack(deltas, axis=0), axis=0)
        directions_plus[ideology_id] = delta_plus
        result = mode_a_for_direction(
            direction=delta_plus,
            sae=sae,
            ideology_id=ideology_id,
            epoch=None,
            vector_kind="delta_plus_aggregate",
            decoder_topk=sae_cfg.decoder_topk,
        )
        mode_a_rows.append(_public_mode_a(result))
        seed_scores[(ideology_id, 0)] = result["scores"]
        if ideology_id in neg_deltas and neg_deltas[ideology_id]:
            delta_minus = np.mean(np.stack(neg_deltas[ideology_id], axis=0), axis=0)
            directions_minus[ideology_id] = delta_minus
            scores_plus = result["scores"]
            scores_minus = decoder_cosine_scores(delta_minus, sae)
            q = scores_plus - scores_minus
            order = np.argsort(q)[::-1][: sae_cfg.decoder_topk]
            contrast_rows.append(
                {
                    "ideology_id": ideology_id,
                    "n_positive_pairs": len(deltas),
                    "n_negative_pairs": len(neg_deltas[ideology_id]),
                    "cosine_delta_plus_vs_minus": cosine(delta_plus, delta_minus),
                    "topk_by_q": [
                        {
                            "feature_id": int(fid),
                            "q": float(q[fid]),
                            "cos_plus": float(scores_plus[fid]),
                            "cos_minus": float(scores_minus[fid]),
                        }
                        for fid in order
                    ],
                    "note": "Diagnostic only. Δ- never enters the injector.",
                }
            )

    score_ids, matrix = score_matrix(directions_plus, sae)
    score_rows = {ideology_id: matrix[i] for i, ideology_id in enumerate(score_ids)}
    shared = intersection_topk(score_rows, sae_cfg.decoder_topk)
    consensus = consensus_topk(
        score_rows,
        k=sae_cfg.decoder_topk,
        threshold=sae_cfg.consensus_threshold,
    )
    specific = specific_features(score_rows, k=sae_cfg.decoder_topk, shared=set(shared))
    n_clusters = max(1, min(len(score_ids), 3)) if score_ids else 0
    clusters = agglomerative_clusters(
        score_ids,
        matrix,
        n_clusters=n_clusters if n_clusters else None,
    )

    # Seed scores from aggregates (one per ideology) already set; also fill from epoch0 if present.
    for (ideology_id, epoch), scores in epoch_scores.items():
        if epoch == 0:
            seed_scores[(ideology_id, 0)] = scores

    stability = stability_report(
        by_epoch=epoch_scores,
        by_seed=seed_scores,
        decoder_topk=sae_cfg.decoder_topk,
        consensus_threshold=sae_cfg.consensus_threshold,
    )

    bootstrap = {}
    for ideology_id, deltas in pos_deltas.items():
        bootstrap[ideology_id] = bootstrap_topk_frequency(
            deltas,
            lambda mean, _sae=sae: decoder_cosine_scores(mean, _sae),
            k=sae_cfg.decoder_topk,
            n_bootstrap=min(50, max(10, len(deltas) * 2)),
            seed=0,
        )[: sae_cfg.decoder_topk]

    hypotheses = _hypotheses(consensus, contrast_rows)

    if sae_cfg.run_mode_a:
        write_score_table(out / "scores_mode_a", mode_a_rows)
        write_json(
            out / "score_matrix.json",
            {
                "ideology_ids": score_ids,
                "shape": list(matrix.shape),
                "matrix": matrix.tolist(),
                "note": "S[i,f]=cos(Δ_i^+, d_f). Geometric only.",
            },
        )
        np.save(out / "score_matrix.npy", matrix)
    if sae_cfg.run_mode_b:
        write_jsonl(out / "activations_mode_b.jsonl", mode_b_rows)

    write_json(
        out / "consensus.json",
        {
            "method_intersection_topk": shared,
            "method_frequency_consensus": consensus,
            "specific_by_ideology": specific,
            "decoder_topk": sae_cfg.decoder_topk,
            "consensus_threshold": sae_cfg.consensus_threshold,
        },
    )
    write_json(out / "clusters.json", clusters)
    write_json(out / "success_vs_fail.json", {"by_ideology": contrast_rows})
    write_json(out / "stability.json", {**stability, "bootstrap_topk": bootstrap})
    write_json(out / "hypotheses_for_causal.json", hypotheses)

    if sae_cfg.run_causal:
        _write_causal_placeholder(root / "causal_sae")
    else:
        write_json(
            root / "causal_sae" / "status.json",
            {
                "run_causal": False,
                "note": "Causal SAE intervention disabled. Correlation report must not be read as causation.",
            },
        )

    summary = {
        "status": "ok",
        "compatibility": compat.to_dict(),
        "sae_metadata": sae.metadata(),
        "n_subruns": len(runs),
        "n_ideologies_with_delta_plus": len(directions_plus),
        "n_mode_a_rows": len(mode_a_rows),
        "n_mode_b_rows": len(mode_b_rows),
        "shared_intersection_size": len(shared),
        "consensus_size": len(consensus),
        "causal_separated": True,
        "run_causal": bool(sae_cfg.run_causal),
        "hypotheses": hypotheses,
    }
    write_json(out / "summary.json", summary)
    write_sae_report(out, summary, consensus=consensus, shared=shared, clusters=clusters)
    return summary


def _public_mode_a(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "scores"}


def _infer_residual_dim(runs: list[dict], layer_k: int | None) -> int | None:
    for sub in runs:
        for episode in sub["episodes"]:
            sender, _ = load_episode_hiddens(episode, sub["path"])
            if sender is None:
                continue
            array = np.asarray(sender)
            if array.ndim == 1:
                return int(array.shape[0])
            if array.ndim == 2:
                if layer_k is not None and 0 <= layer_k < array.shape[0]:
                    return int(array.shape[1])
                return int(array.shape[1])
    return None


def _hypotheses(consensus: list[dict], contrast_rows: list[dict]) -> list[dict]:
    items = []
    for row in consensus[:10]:
        items.append(
            {
                "feature_id": row["feature_id"],
                "source": "frequency_consensus",
                "C_f": row["C_f"],
                "hypothesis": (
                    f"Feature {row['feature_id']} co-occurs in Top-k decoder alignments "
                    "across ideas; candidate for amplify/ablate tests only."
                ),
                "causal_claim": False,
            }
        )
    for block in contrast_rows:
        for row in block.get("topk_by_q", [])[:5]:
            items.append(
                {
                    "feature_id": row["feature_id"],
                    "source": "success_minus_fail",
                    "ideology_id": block["ideology_id"],
                    "q": row["q"],
                    "hypothesis": (
                        f"Feature {row['feature_id']} is more aligned with Δ+ than Δ- "
                        f"for {block['ideology_id']}; correlational only."
                    ),
                    "causal_claim": False,
                }
            )
    return items


def _write_causal_placeholder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(
        path / "README.json",
        {
            "status": "not_run",
            "required": [
                "baseline",
                "amplify/ablate selected features",
                "matched control features",
                "multiple intervention strengths",
                "P_final / P_all / semantic / CSO drift",
                "matched seeds",
            ],
            "note": (
                "Causal SAE intervention is a separate experiment. "
                "Do not infer causation from sae/ correlation outputs."
            ),
        },
    )
