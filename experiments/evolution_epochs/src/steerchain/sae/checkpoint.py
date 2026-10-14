"""TopK SAE loader matching Qwen-Scope checkpoint layout."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TopKSAE:
    """Qwen-Scope TopK SAE. Encoding is TopK(pre-acts), never ReLU."""

    W_enc: np.ndarray  # (d_sae, d_model)
    W_dec: np.ndarray  # (d_model, d_sae)
    b_enc: np.ndarray  # (d_sae,)
    b_dec: np.ndarray  # (d_model,)
    top_k: int
    repo_id: str
    layer_k: int
    checksum: str
    source_path: str
    revision: str | None = None
    architecture: str = "topk"
    activation_site: str = "resid_post_layer"
    normalization: str = "none"

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[1])

    def metadata(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "architecture": self.architecture,
            "top_k": self.top_k,
            "d_sae": self.d_sae,
            "d_model": self.d_model,
            "transformer_layer_k": self.layer_k,
            "activation_site": self.activation_site,
            "normalization": self.normalization,
            "checksum": self.checksum,
            "source_path": self.source_path,
            "feature_ids": "0..d_sae-1 contiguous",
        }

    def encode(self, hidden: np.ndarray) -> np.ndarray:
        """Exact Qwen-Scope TopK forward: pre = h @ W_enc.T + b_enc; keep TopK."""
        h = np.asarray(hidden, dtype=np.float64)
        single = h.ndim == 1
        if single:
            h = h[None, :]
        if h.shape[-1] != self.d_model:
            raise ValueError(f"Hidden dim {h.shape[-1]} != SAE d_model {self.d_model}")
        flat = h.reshape(-1, self.d_model)
        pre = flat @ self.W_enc.T + self.b_enc
        k = min(self.top_k, self.d_sae)
        idx = np.argpartition(pre, -k, axis=-1)[:, -k:]
        acts = np.zeros_like(pre)
        rows = np.arange(pre.shape[0])[:, None]
        # Qwen-Scope: scatter the TopK pre-activation values (sign preserved).
        acts[rows, idx] = pre[rows, idx]
        out = acts.reshape(*h.shape[:-1], self.d_sae)
        return out[0] if single else out

    def decode(self, acts: np.ndarray) -> np.ndarray:
        z = np.asarray(acts, dtype=np.float64)
        single = z.ndim == 1
        if single:
            z = z[None, :]
        recon = z @ self.W_dec.T + self.b_dec
        return recon[0] if single else recon

    def decoder_directions(self) -> np.ndarray:
        """Columns of W_dec: shape (d_model, d_sae)."""
        return np.asarray(self.W_dec, dtype=np.float64)

    @classmethod
    def from_dict(
        cls,
        payload: dict,
        *,
        top_k: int,
        repo_id: str,
        layer_k: int,
        source_path: str,
        checksum: str,
        revision: str | None = None,
    ) -> TopKSAE:
        required = ("W_enc", "W_dec", "b_enc", "b_dec")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"SAE checkpoint missing keys: {missing}")
        W_enc = np.asarray(payload["W_enc"], dtype=np.float64)
        W_dec = np.asarray(payload["W_dec"], dtype=np.float64)
        b_enc = np.asarray(payload["b_enc"], dtype=np.float64).reshape(-1)
        b_dec = np.asarray(payload["b_dec"], dtype=np.float64).reshape(-1)
        if W_enc.ndim != 2 or W_dec.ndim != 2:
            raise ValueError("W_enc and W_dec must be 2D")
        if W_enc.shape[0] != W_dec.shape[1] or W_enc.shape[1] != W_dec.shape[0]:
            raise ValueError(f"Incompatible SAE shapes W_enc={W_enc.shape} W_dec={W_dec.shape}")
        if b_enc.shape != (W_enc.shape[0],) or b_dec.shape != (W_dec.shape[0],):
            raise ValueError("SAE bias shapes do not match weight matrices")
        return cls(
            W_enc=W_enc,
            W_dec=W_dec,
            b_enc=b_enc,
            b_dec=b_dec,
            top_k=int(top_k),
            repo_id=repo_id,
            layer_k=int(layer_k),
            checksum=checksum,
            source_path=source_path,
            revision=revision,
        )

    @classmethod
    def synthetic(
        cls,
        *,
        d_model: int,
        d_sae: int,
        top_k: int,
        layer_k: int,
        seed: int = 0,
    ) -> TopKSAE:
        """Deterministic offline SAE for unit tests. Not a Qwen-Scope weight file."""
        rng = np.random.default_rng(seed)
        W_enc = rng.normal(size=(d_sae, d_model))
        W_enc /= np.linalg.norm(W_enc, axis=1, keepdims=True) + 1e-12
        W_dec = W_enc.T.copy()
        b_enc = rng.normal(size=(d_sae,)) * 0.01
        b_dec = np.zeros(d_model, dtype=np.float64)
        checksum = hashlib.sha256(
            W_enc.tobytes() + W_dec.tobytes() + b_enc.tobytes() + b_dec.tobytes()
        ).hexdigest()
        return cls(
            W_enc=W_enc,
            W_dec=W_dec,
            b_enc=b_enc,
            b_dec=b_dec,
            top_k=top_k,
            repo_id=f"synthetic://topk-d{d_model}-w{d_sae}-k{top_k}",
            layer_k=layer_k,
            checksum=checksum,
            source_path="synthetic",
            revision=None,
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_topk_sae(
    *,
    repo_id: str,
    layer_k: int,
    top_k: int,
    local_dir: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> TopKSAE:
    """Load layer{k}.sae.pt from a local directory or Hugging Face hub."""
    filename = f"layer{layer_k}.sae.pt"
    if local_dir is not None:
        path = Path(local_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Local SAE file not found: {path}")
        return _load_pt(path, repo_id=repo_id, layer_k=layer_k, top_k=top_k, revision=revision)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download Qwen-Scope SAE weights. "
            "Install with: pip install huggingface_hub"
        ) from exc

    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
    )
    return _load_pt(path, repo_id=repo_id, layer_k=layer_k, top_k=top_k, revision=revision)


def _load_pt(
    path: Path,
    *,
    repo_id: str,
    layer_k: int,
    top_k: int,
    revision: str | None,
) -> TopKSAE:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to load Qwen-Scope .pt SAE checkpoints") from exc

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict checkpoint in {path}")
    arrays = {key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value) for key, value in payload.items()}
    return TopKSAE.from_dict(
        arrays,
        top_k=top_k,
        repo_id=repo_id,
        layer_k=layer_k,
        source_path=str(path),
        checksum=file_sha256(path),
        revision=revision,
    )
