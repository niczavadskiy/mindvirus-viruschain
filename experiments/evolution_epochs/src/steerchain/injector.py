"""Additive residual intervention at one transformer layer."""

from __future__ import annotations

from typing import Any

import numpy as np

_KNOWN_PREFIXES = ("llama", "qwen2", "qwen3", "gemma2")


def resolve_layers(model: Any) -> Any:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if not isinstance(model_type, str) or not model_type.startswith(_KNOWN_PREFIXES):
        raise ValueError(
            f"Unsupported model_type {model_type!r}. "
            "Expected a llama, qwen2, qwen3, or gemma2 checkpoint."
        )
    parent = getattr(model, "model", None)
    layers = getattr(parent, "layers", None)
    if layers is None:
        raise ValueError(f"Could not find model.layers on model_type {model_type!r}.")
    return layers


def layer_index(n_layers: int, requested: int | None) -> int:
    """Resolve a layer index. None keeps the mid-layer placeholder until auto-select runs."""
    if n_layers < 1:
        raise ValueError("Model has no transformer layers.")
    if requested is None:
        return n_layers // 2
    if not 0 <= requested < n_layers:
        raise ValueError(f"transformer layer k={requested} is outside 0..{n_layers - 1}.")
    return requested


def select_max_delta_layer(delta_stack: np.ndarray) -> int:
    """Pick the residual layer whose mean pair-delta has the largest L2 norm."""
    stack = np.asarray(delta_stack, dtype=np.float64)
    if stack.ndim != 2:
        raise ValueError(f"Expected delta stack of shape (n_layers, hidden), got {stack.shape}.")
    if stack.shape[0] < 1:
        raise ValueError("Delta stack has no layers.")
    norms = np.linalg.norm(stack, axis=1)
    return int(np.argmax(norms))


def add_vector(hidden: Any, vector: np.ndarray) -> Any:
    vector = np.asarray(vector, dtype=np.float64)
    if hasattr(hidden, "detach"):
        import torch

        vec = torch.as_tensor(vector, device=hidden.device, dtype=hidden.dtype)
        view = [1] * (hidden.ndim - 1) + [-1]
        return hidden + vec.view(*view)
    array = np.asarray(hidden, dtype=np.float64)
    return array + vector


def additive_hook(vector: np.ndarray):
    direction = np.asarray(vector, dtype=np.float64).copy()

    def hook(module, inputs, output):
        del module, inputs
        if isinstance(output, tuple):
            hidden = add_vector(output[0], direction)
            return (hidden,) + output[1:]
        return add_vector(output, direction)

    return hook


def attach_additive_hook(model: Any, layer_k: int, vector: np.ndarray):
    layers = resolve_layers(model)
    if not 0 <= layer_k < len(layers):
        raise ValueError(f"transformer layer k={layer_k} is outside 0..{len(layers) - 1}.")
    return layers[layer_k].register_forward_hook(additive_hook(vector))
