"""Delta, cumulative direction, normalization, layer probes, and shuffle control."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from steerchain.embeddings import cosine
from steerchain.injector import select_max_delta_layer
from steerchain.records import EpisodeRecord

LAYER_SELECT_MODES = ("max_delta", "probe_ba")


def at_layer(hidden: np.ndarray, layer: int | None) -> np.ndarray:
    """Return a 1D residual. Multi-layer stacks require an explicit layer index."""
    array = np.asarray(hidden, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError(f"Hidden state must be 1D or 2D, got shape {array.shape}.")
    if layer is None:
        raise ValueError("Multi-layer hidden state requires a selected layer.")
    if not 0 <= layer < array.shape[0]:
        raise ValueError(f"Layer {layer} is outside 0..{array.shape[0] - 1}.")
    return array[layer]


def pair_deltas(pairs: list[EpisodeRecord], layer: int | None = None) -> list[np.ndarray]:
    deltas = []
    for record in pairs:
        if record.h_sender_send is None or record.h_target_pre_end is None:
            raise ValueError(f"Pair {record.episode_id} is missing a hidden state.")
        if record.h_sender_send.shape != record.h_target_pre_end.shape:
            raise ValueError(
                f"Pair {record.episode_id} mixes hidden sizes "
                f"{record.h_sender_send.shape} and {record.h_target_pre_end.shape}."
            )
        sender = np.asarray(record.h_sender_send, dtype=np.float64)
        target = np.asarray(record.h_target_pre_end, dtype=np.float64)
        if layer is None:
            deltas.append(sender - target)
        else:
            deltas.append(at_layer(sender, layer) - at_layer(target, layer))
    return deltas


def mean_delta(pairs: list[EpisodeRecord], layer: int | None = None) -> np.ndarray | None:
    """None when N_e = 0. Without layer, returns the full (n_layers, H) stack when available."""
    if not pairs:
        return None
    return np.mean(np.stack(pair_deltas(pairs, layer=layer), axis=0), axis=0)


def layer_delta_norms(delta_stack: np.ndarray) -> list[float]:
    stack = np.asarray(delta_stack, dtype=np.float64)
    if stack.ndim == 1:
        return [float(np.linalg.norm(stack))]
    return [float(value) for value in np.linalg.norm(stack, axis=1)]


def balanced_accuracy(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """(TPR + TNR) / 2. NaN when either class is missing."""
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_hat, dtype=int)
    positives = truth == 1
    negatives = truth == 0
    if not np.any(positives) or not np.any(negatives):
        return float("nan")
    tpr = float(np.mean(pred[positives] == 1))
    tnr = float(np.mean(pred[negatives] == 0))
    return 0.5 * (tpr + tnr)


def _with_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float64)], axis=1)


def fit_linear_probe(features: np.ndarray, labels: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Ridge linear regression of Y ∈ {0,1} on residual features (+ bias)."""
    design = _with_bias(np.asarray(features, dtype=np.float64))
    target = np.asarray(labels, dtype=np.float64)
    dim = design.shape[1]
    gram = design.T @ design + float(ridge) * np.eye(dim, dtype=np.float64)
    return np.linalg.solve(gram, design.T @ target)


def predict_linear_probe(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _with_bias(np.asarray(features, dtype=np.float64)) @ np.asarray(weights, dtype=np.float64)


def leave_one_out_probe_scores(features: np.ndarray, labels: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.zeros(len(labels), dtype=np.float64)
    for index in range(len(labels)):
        mask = np.ones(len(labels), dtype=bool)
        mask[index] = False
        weights = fit_linear_probe(features[mask], labels[mask], ridge=ridge)
        scores[index] = float(predict_linear_probe(features[index : index + 1], weights)[0])
    return scores


def probe_balanced_accuracy_for_layer(
    pairs: list[EpisodeRecord],
    layer: int | None,
    *,
    ridge: float = 1e-3,
) -> float:
    """Linear probe on Δ=h_sender−h_target predicting Y (idea accepted on new hop)."""
    if len(pairs) < 2:
        return float("nan")
    deltas = []
    labels = []
    for record in pairs:
        if record.h_sender_send is None or record.h_target_pre_end is None:
            raise ValueError(f"Pair {record.episode_id} is missing a hidden state.")
        delta = at_layer(record.h_sender_send, layer) - at_layer(record.h_target_pre_end, layer)
        deltas.append(delta)
        labels.append(int(record.y_after))
    features = np.stack(deltas, axis=0)
    truth = np.asarray(labels, dtype=int)
    if len(np.unique(truth)) < 2:
        return float("nan")
    if len(truth) >= 3:
        scores = leave_one_out_probe_scores(features, truth.astype(np.float64), ridge=ridge)
    else:
        weights = fit_linear_probe(features, truth.astype(np.float64), ridge=ridge)
        scores = predict_linear_probe(features, weights)
    return balanced_accuracy(truth, (scores >= 0.5).astype(int))


def layer_probe_balanced_accuracies(
    pairs: list[EpisodeRecord],
    *,
    ridge: float = 1e-3,
) -> list[float]:
    """Per-layer balanced accuracy of the acceptance probe. Empty when no pairs."""
    if not pairs:
        return []
    sample = np.asarray(pairs[0].h_sender_send)
    if sample.ndim == 1:
        return [probe_balanced_accuracy_for_layer(pairs, None, ridge=ridge)]
    return [
        probe_balanced_accuracy_for_layer(pairs, layer, ridge=ridge)
        for layer in range(int(sample.shape[0]))
    ]


def select_max_probe_ba_layer(scores: list[float]) -> int:
    array = np.asarray(scores, dtype=np.float64)
    if array.size < 1 or not np.any(np.isfinite(array)):
        raise ValueError("No finite probe balanced accuracies to select from.")
    filled = np.where(np.isfinite(array), array, -np.inf)
    return int(np.argmax(filled))


def resolve_contrastive_layer(
    delta_stack: np.ndarray,
    *,
    locked: int | None = None,
    previous: int | None = None,
    mode: str = "max_delta",
    probe_ba_scores: list[float] | None = None,
) -> int:
    """Prefer a config lock, else keep the prior choice, else apply layer_select_mode."""
    if locked is not None:
        return int(locked)
    if previous is not None:
        return int(previous)
    if mode not in LAYER_SELECT_MODES:
        raise ValueError(f"Unknown layer_select_mode {mode!r}. Expected one of {LAYER_SELECT_MODES}.")
    if mode == "max_delta":
        return select_max_delta_layer(delta_stack)
    if probe_ba_scores is None or not any(np.isfinite(np.asarray(probe_ba_scores, dtype=np.float64))):
        # Need both Y=0 and Y=1 pairs; otherwise fall back to max ||Δ||.
        return select_max_delta_layer(delta_stack)
    return select_max_probe_ba_layer(probe_ba_scores)


def shuffle_control(
    pairs: list[EpisodeRecord],
    rng: np.random.Generator,
    layer: int | None = None,
) -> tuple[np.ndarray | None, float | None]:
    """Permute which sender state is subtracted from which target state.

    The mean of differences is invariant to that permutation. The alignment
    score is the mean cosine between each real pair-delta and its shuffled
    counterpart, and it drops when the individual pairings actually differed.
    """
    if not pairs:
        return None, None
    if any(record.h_sender_send is None or record.h_target_pre_end is None for record in pairs):
        raise ValueError("Shuffle control requires both hidden states.")
    senders = [at_layer(record.h_sender_send, layer).copy() for record in pairs]
    targets = [at_layer(record.h_target_pre_end, layer).copy() for record in pairs]
    original = [sender - target for sender, target in zip(senders, targets, strict=True)]
    order = rng.permutation(len(senders))
    shuffled = [senders[int(index)] - target for index, target in zip(order, targets, strict=True)]
    mean = np.mean(np.stack(shuffled, axis=0), axis=0)
    if len(pairs) < 2:
        return mean, None
    alignment = float(np.mean([cosine(left, right) for left, right in zip(original, shuffled, strict=True)]))
    return mean, alignment


def scale_direction(
    source: np.ndarray | None,
    mode: str,
    beta: float,
    fixed_norm: float,
) -> np.ndarray | None:
    if source is None:
        return None
    vector = np.asarray(source, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if mode == "raw":
        return beta * vector
    if norm == 0.0:
        return None
    if mode == "unit":
        return beta * (vector / norm)
    if mode == "fixed_norm":
        return beta * (vector / norm) * fixed_norm
    raise ValueError(f"Unknown norm_mode {mode!r}. Expected raw, unit, or fixed_norm.")


def unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    vector = rng.normal(size=dim)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        vector = np.zeros(dim, dtype=np.float64)
        vector[0] = 1.0
        return vector
    return vector / norm


@dataclass
class SteeringMemory:
    """Raw deltas stay stored. Normalization is applied only when a direction is requested."""

    hidden_size: int
    random_unit: np.ndarray
    selected_layer: int | None = None
    layer_select_mode: str = "max_delta"
    layer_delta_norms_by_epoch: dict[int, list[float]] = field(default_factory=dict)
    layer_probe_ba_by_epoch: dict[int, list[float | None]] = field(default_factory=dict)
    delta_by_epoch: dict[int, np.ndarray | None] = field(default_factory=dict)
    delta_negative_by_epoch: dict[int, np.ndarray | None] = field(default_factory=dict)
    shuffled_by_epoch: dict[int, np.ndarray | None] = field(default_factory=dict)
    cumulative: np.ndarray | None = None
    cumulative_by_epoch: dict[int, np.ndarray | None] = field(default_factory=dict)
    absent_epochs: list[int] = field(default_factory=list)

    @classmethod
    def create(cls, hidden_size: int, rng: np.random.Generator) -> SteeringMemory:
        return cls(hidden_size=hidden_size, random_unit=unit_vector(rng, hidden_size))

    @property
    def delta0(self) -> np.ndarray | None:
        return self.delta_by_epoch.get(0)

    @property
    def shuffled0(self) -> np.ndarray | None:
        return self.shuffled_by_epoch.get(0)

    def update(
        self,
        epoch: int,
        delta: np.ndarray | None,
        delta_negative: np.ndarray | None,
        shuffled: np.ndarray | None,
        condition: str,
        *,
        selected_layer: int | None = None,
        layer_norms: list[float] | None = None,
        layer_probe_ba: list[float] | None = None,
        layer_select_mode: str | None = None,
    ) -> None:
        if selected_layer is not None:
            self.selected_layer = int(selected_layer)
        if layer_select_mode is not None:
            self.layer_select_mode = str(layer_select_mode)
        if layer_norms is not None:
            self.layer_delta_norms_by_epoch[epoch] = [float(value) for value in layer_norms]
        if layer_probe_ba is not None:
            cleaned: list[float | None] = []
            for value in layer_probe_ba:
                if value is None or not np.isfinite(value):
                    cleaned.append(None)
                else:
                    cleaned.append(float(value))
            self.layer_probe_ba_by_epoch[epoch] = cleaned
        self.delta_by_epoch[epoch] = None if delta is None else np.asarray(delta, dtype=np.float64).copy()
        self.delta_negative_by_epoch[epoch] = None if delta_negative is None else np.asarray(delta_negative, dtype=np.float64).copy()
        self.shuffled_by_epoch[epoch] = None if shuffled is None else np.asarray(shuffled, dtype=np.float64).copy()
        if delta is None:
            self.absent_epochs.append(epoch)
            self.cumulative_by_epoch[epoch] = None if self.cumulative is None else self.cumulative.copy()
            return
        if condition == "C":
            self.cumulative = delta.copy() if self.cumulative is None else self.cumulative + delta
        self.cumulative_by_epoch[epoch] = None if self.cumulative is None else self.cumulative.copy()

    def applied_vector(self, condition: str, epoch: int, mode: str, beta: float, fixed_norm: float) -> np.ndarray | None:
        if epoch == 0 or condition == "A":
            return None
        if condition == "B":
            return scale_direction(self.delta0, mode, beta, fixed_norm)
        if condition == "C":
            return scale_direction(self.cumulative, mode, beta, fixed_norm)
        if condition == "D":
            matched = scale_direction(self.delta0, mode, beta, fixed_norm)
            if matched is None:
                return None
            return self.random_unit * float(np.linalg.norm(matched))
        if condition == "E":
            return scale_direction(self.shuffled0, mode, beta, fixed_norm)
        raise ValueError(f"Unknown condition {condition!r}. Expected A, B, C, D, or E.")

    def summary(self, epoch: int) -> dict:
        delta = self.delta_by_epoch.get(epoch)
        previous = self.delta_by_epoch.get(epoch - 1)
        cumulative = self.cumulative_by_epoch.get(epoch)
        previous_cumulative = self.cumulative_by_epoch.get(epoch - 1)
        return {
            "epoch": epoch,
            "N_status": "absent_no_successful_pairs" if delta is None else "computed",
            "substituted_random": False,
            "selected_layer": self.selected_layer,
            "layer_select_mode": self.layer_select_mode,
            "layer_delta_norms": self.layer_delta_norms_by_epoch.get(epoch),
            "layer_probe_ba": self.layer_probe_ba_by_epoch.get(epoch),
            "delta_norm": None if delta is None else float(np.linalg.norm(delta)),
            "delta_negative_norm": None
            if self.delta_negative_by_epoch.get(epoch) is None
            else float(np.linalg.norm(self.delta_negative_by_epoch[epoch])),
            "cumulative_norm": None if cumulative is None else float(np.linalg.norm(cumulative)),
            "cosine_delta_previous": None
            if delta is None or previous is None
            else cosine(delta, previous),
            "cosine_cumulative_previous": None
            if cumulative is None or previous_cumulative is None
            else cosine(cumulative, previous_cumulative),
            "cosine_delta_vs_negative": None
            if delta is None or self.delta_negative_by_epoch.get(epoch) is None
            else cosine(delta, self.delta_negative_by_epoch[epoch]),
        }
