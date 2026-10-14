"""Hop chain and the outer epoch loop."""

from __future__ import annotations

import numpy as np

from steerchain.episode import Agent, run_episode
from steerchain.estimator import (
    SteeringMemory,
    layer_delta_norms,
    layer_probe_balanced_accuracies,
    mean_delta,
    resolve_contrastive_layer,
    shuffle_control,
)
from steerchain.injector import layer_index
from steerchain.metrics import epoch_metrics
from steerchain.pairs import eligible_pairs, repeated_senders, successful_pairs, unsuccessful_pairs
from steerchain.workspace import Workspace


def _spread_plan(spreaders: list[Agent], n_targets: int, rng: np.random.Generator) -> list[Agent]:
    if not spreaders or n_targets <= 0:
        return []
    order = rng.permutation(len(spreaders))
    counts = [n_targets // len(spreaders)] * len(spreaders)
    for slot in range(n_targets % len(spreaders)):
        counts[int(order[slot])] += 1
    plan: list[Agent] = []
    for spreader, count in zip(spreaders, counts, strict=True):
        plan.extend([spreader] * count)
    return plan


def _seed_agents(epoch: int, ideology, n_seed_agents: int) -> list[Agent]:
    agents = []
    for index in range(n_seed_agents):
        agents.append(
            Agent(
                agent_id=f"e{epoch}-h0-a{index}",
                hop=0,
                workspace=Workspace({"SOUL.md": ideology.payload, "MEMORY.md": ""}),
            )
        )
    return agents


def _naive_agent(epoch: int, hop: int, index: int, default_soul: str) -> Agent:
    return Agent(
        agent_id=f"e{epoch}-h{hop}-a{index}",
        hop=hop,
        workspace=Workspace({"SOUL.md": default_soul, "MEMORY.md": ""}),
    )


def run_condition(
    *,
    config,
    ideology,
    condition: str,
    seed: int,
    default_soul: str,
    backend,
    detector,
    logger,
) -> list[dict]:
    vector_rng = np.random.default_rng(seed + 9176)
    pair_rng = np.random.default_rng(seed)
    shuffle_rng = np.random.default_rng(seed + 1009)
    memory = SteeringMemory.create(backend.hidden_size, vector_rng)
    if config.transformer_layer_k is not None:
        locked = layer_index(backend.n_layers, config.transformer_layer_k)
        memory.selected_layer = locked
        backend.set_steering_layer(locked)
    rows: list[dict] = []
    for epoch in range(config.n_epochs):
        applied = memory.applied_vector(
            condition,
            epoch,
            config.norm_mode,
            config.beta,
            config.fixed_norm,
        )
        logged = len(getattr(backend, "steer_log", []))
        records = _run_epoch(
            config=config,
            ideology=ideology,
            condition=condition,
            seed=seed,
            epoch=epoch,
            default_soul=default_soul,
            backend=backend,
            detector=detector,
            pair_rng=pair_rng,
            applied=applied,
            logger=logger,
        )
        positives = successful_pairs(records, config.pair_spreader_hop, config.pair_target_hop)
        negatives = unsuccessful_pairs(records, config.pair_spreader_hop, config.pair_target_hop)
        labeled = eligible_pairs(records, config.pair_spreader_hop, config.pair_target_hop)
        delta_stack = mean_delta(positives)
        negative_stack = mean_delta(negatives)
        selected_layer = memory.selected_layer
        norms: list[float] | None = None
        probe_ba: list[float] | None = None
        delta = None
        delta_negative = None
        shuffled = None
        shuffle_alignment = None
        mode = getattr(config, "layer_select_mode", "max_delta")
        if delta_stack is not None:
            norms = layer_delta_norms(delta_stack)
            if mode == "probe_ba":
                probe_ba = layer_probe_balanced_accuracies(labeled)
            locked = None
            if config.transformer_layer_k is not None:
                locked = layer_index(backend.n_layers, config.transformer_layer_k)
            selected_layer = resolve_contrastive_layer(
                delta_stack,
                locked=locked,
                previous=memory.selected_layer,
                mode=mode,
                probe_ba_scores=probe_ba,
            )
            backend.set_steering_layer(selected_layer)
            if delta_stack.ndim == 1:
                delta = delta_stack
            else:
                delta = delta_stack[selected_layer]
            if negative_stack is not None:
                delta_negative = (
                    negative_stack if negative_stack.ndim == 1 else negative_stack[selected_layer]
                )
            shuffled, shuffle_alignment = shuffle_control(positives, shuffle_rng, layer=selected_layer)
        memory.update(
            epoch,
            delta,
            delta_negative,
            shuffled,
            condition,
            selected_layer=selected_layer,
            layer_norms=norms,
            layer_probe_ba=probe_ba,
            layer_select_mode=mode,
        )
        steer_hops = list(getattr(backend, "steer_log", []))[logged:]
        if any(hop != 0 for hop in steer_hops) or getattr(backend, "illegal_steering", []):
            raise RuntimeError(f"Steering left hop 0: hops={steer_hops} illegal={backend.illegal_steering}")
        summary = logger.write_steering_epoch(
            epoch,
            memory,
            applied,
            extra={"steer_hops": steer_hops},
        )
        counts = repeated_senders(positives)
        row = epoch_metrics(
            records=records,
            n_hops=config.n_hops,
            condition=condition,
            ideology_id=ideology.ideology_id,
            seed=seed,
            epoch=epoch,
            retention_threshold=config.probe_retention_threshold,
            steering_summary=summary,
            n_positive_pairs=len(positives),
            n_negative_pairs=len(negatives),
            repeated_senders=counts,
            shuffle_alignment=shuffle_alignment,
        )
        rows.append(row)
    return rows


def _run_epoch(
    *,
    config,
    ideology,
    condition: str,
    seed: int,
    epoch: int,
    default_soul: str,
    backend,
    detector,
    pair_rng,
    applied,
    logger,
) -> list:
    spreaders = _seed_agents(epoch, ideology, config.n_seed_agents)
    records = []
    for hop in range(1, config.n_hops + 1):
        plan = _spread_plan(spreaders, config.agents_per_hop, pair_rng)
        if not plan:
            break
        infected: list[Agent] = []
        for index, sender in enumerate(plan):
            target = _naive_agent(epoch, hop, index, default_soul)
            episode_id = f"{ideology.ideology_id}-{condition}-s{seed}-e{epoch}-h{hop}-i{index}"
            sender.wipe_context()
            record = run_episode(
                episode_id=episode_id,
                epoch=epoch,
                condition=condition,
                ideology=ideology,
                seed=seed,
                model_id=backend.model_id,
                sender=sender,
                target=target,
                backend=backend,
                detector=detector,
                turns=config.turns_per_hop,
                max_tool_loops=config.max_tool_loops,
                run_probes=config.run_probes,
                applied_vector=applied,
                require_tool_action=bool(getattr(config, "require_tool_action", True)),
                tool_retries=int(getattr(config, "tool_retries", 1)),
            )
            logger.write_episode(record)
            records.append(record)
            if record.y_after == 1:
                infected.append(target.spawn_carrier(record.episode_id))
        spreaders = infected
    return records
