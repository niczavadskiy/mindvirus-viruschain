"""One-shot remote bootstrap: patch embeddings + mini HF config, print env fixes.

Run in notebook:
  %run /workspace/mind-viruses/experiments/evolution_epochs/scripts/bootstrap_remote_smoke.py
or:
  !python /workspace/mind-viruses/experiments/evolution_epochs/scripts/bootstrap_remote_smoke.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src" / "steerchain").is_dir():
    ROOT = Path("/workspace/mind-viruses/experiments/evolution_epochs")

EMBEDDINGS = '''"""Text embeddings."""

from __future__ import annotations

import re
import zlib

import numpy as np

_STOP = frozenset(
    """
    a an the of and or to for in on with that this it is are was were be been being
    as by from at not no nor so if when what who how why which about into over than
    then their its your you we they them our my i he she his her do does did have has
    had will would can could should just than too very also only
    """.split()
)


def content_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [word for word in words if word not in _STOP and len(word) > 2]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


class LexicalEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float64)
        for token in content_tokens(text):
            index = zlib.crc32(token.encode("utf-8")) % self.dim
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


class TransformersEmbedder:
    """BGE via HF mean pooling. Does not import sentence_transformers."""

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        torch = self.torch
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            vector = (summed / counts)[0].float().cpu().numpy()
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


def build_embedder(kind: str, model_id: str):
    if kind == "lexical":
        return LexicalEmbedder()
    if kind in ("transformers", "sentence_transformer"):
        # Always prefer TransformersEmbedder on this stack; ST pulls broken torchvision paths.
        return TransformersEmbedder(model_id)
    raise ValueError(f"Unknown embedder {kind!r}")
'''

MINI_CFG = """model_id: Qwen/Qwen3.5-2B-Base
judge_model_id: null
embedding_model_id: BAAI/bge-small-en-v1.5
backend: hf
embedder: transformers

n_hops: 10
agents_per_hop: 2
turns_per_hop: 2
max_tool_loops: 3
n_epochs: 2
seeds: [0]
n_seed_agents: 1

ideologies:
  - whale_welfare
conditions: [A, B]

norm_mode: unit
beta: 1.0
fixed_norm: 1.0
tau: null
pair_spreader_hop: 9
pair_target_hop: 10
probe_retention_threshold: 1.0
transformer_layer_k: null
layer_select_mode: max_delta
hidden_size: 2048
max_new_tokens: 128
run_probes: true
run_dir: runs/smoke_hf_mini

sae:
  enabled: false
  repo_id: Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100
  architecture: topk
  top_k: 100
  d_sae: 32768
  d_model: 2048
  activation_site: resid_post_layer
  transformer_layer_k: 12
  decoder_topk: 50
  consensus_threshold: 0.5
  cluster_method: agglomerative
  run_mode_a: true
  run_mode_b: true
  run_causal: false
"""


def main() -> None:
    emb = ROOT / "src" / "steerchain" / "embeddings.py"
    emb.write_text(EMBEDDINGS, encoding="utf-8")
    cfg_dir = ROOT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    mini = cfg_dir / "smoke_hf_mini.yaml"
    mini.write_text(MINI_CFG, encoding="utf-8")
    print(f"patched {emb}")
    print(f"wrote {mini}")
    print(
        """
NEXT:

!pip uninstall -y httpx2 sentence-transformers torchaudio torchvision
!pip install -q "httpx>=0.27,<0.28" -U huggingface_hub -r requirements-hf.txt -e .
!python -m steerchain --config {mini}
""".format(
            mini=mini
        )
    )


if __name__ == "__main__":
    main()
