"""Compatibility checks between host model activations and Qwen-Scope SAE."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from steerchain.config import SAEConfig

_COMPATIBLE_HOSTS = (
    "qwen/qwen3.5-2b-base",
    "qwen3.5-2b-base",
)
_COMPATIBLE_SAE_PREFIXES = (
    "qwen/sae-res-qwen3.5-2b-base-",
)


@dataclass
class CompatibilityReport:
    compatible: bool
    status: str
    host_model_id: str
    sae_repo_id: str
    architecture: str
    activation_site: str
    transformer_layer_k: int | None
    residual_dim: int | None
    expected_d_model: int
    expected_d_sae: int
    expected_top_k: int
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_id(value: str) -> str:
    return value.strip().lower()


def check_compatibility(
    *,
    host_model_id: str,
    sae: SAEConfig,
    residual_dim: int | None,
    transformer_layer_k: int | None,
    allow_synthetic: bool = False,
) -> CompatibilityReport:
    reasons: list[str] = []
    host = _norm_id(host_model_id)
    repo = _norm_id(sae.repo_id)

    if sae.architecture != "topk":
        reasons.append(f"architecture {sae.architecture!r} is not topk")
    if sae.activation_site != "resid_post_layer":
        reasons.append(f"activation_site {sae.activation_site!r} is not resid_post_layer")

    host_ok = any(host == item or host.endswith("/" + item) for item in _COMPATIBLE_HOSTS)
    if allow_synthetic and sae.repo_id.startswith("synthetic://"):
        host_ok = True
    elif not host_ok:
        reasons.append(
            f"host {host_model_id!r} is not Qwen/Qwen3.5-2B-Base "
            "(Qwen-Scope residual SAEs are bound to that backbone)"
        )

    sae_ok = any(repo.startswith(prefix) for prefix in _COMPATIBLE_SAE_PREFIXES)
    if allow_synthetic and sae.repo_id.startswith("synthetic://"):
        sae_ok = True
    elif not sae_ok:
        reasons.append(
            f"SAE repo {sae.repo_id!r} is not a Qwen-Scope SAE-Res-Qwen3.5-2B-Base checkpoint"
        )

    layer = sae.transformer_layer_k if sae.transformer_layer_k is not None else transformer_layer_k
    if layer is None:
        reasons.append("transformer_layer_k is unset for SAE")
    elif not allow_synthetic and not (0 <= int(layer) <= 23):
        reasons.append(f"transformer_layer_k={layer} is outside Qwen-Scope range 0..23")

    if residual_dim is not None and int(residual_dim) != int(sae.d_model):
        reasons.append(
            f"residual dim {residual_dim} does not match SAE d_model {sae.d_model}"
        )

    compatible = not reasons
    return CompatibilityReport(
        compatible=compatible,
        status="compatible" if compatible else "incompatible",
        host_model_id=host_model_id,
        sae_repo_id=sae.repo_id,
        architecture=sae.architecture,
        activation_site=sae.activation_site,
        transformer_layer_k=None if layer is None else int(layer),
        residual_dim=residual_dim,
        expected_d_model=int(sae.d_model),
        expected_d_sae=int(sae.d_sae),
        expected_top_k=int(sae.top_k),
        reasons=reasons,
    )
