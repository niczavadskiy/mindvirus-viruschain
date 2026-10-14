"""Text embeddings. The lexical embedder is deterministic and does not load weights."""

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
    """Hashed bag of content tokens. Same text always yields the same vector."""

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


class SentenceTransformerEmbedder:
    """Separate from the steered model. Used only when a real embedding run is requested."""

    def __init__(self, model_id: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id)

    def embed(self, text: str) -> np.ndarray:
        vector = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float64)


class TransformersEmbedder:
    """BGE / HF encoder via mean pooling. Avoids the sentence-transformers dependency."""

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
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
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


def build_embedder(
    kind: str,
    model_id: str,
) -> LexicalEmbedder | SentenceTransformerEmbedder | TransformersEmbedder:
    if kind == "lexical":
        return LexicalEmbedder()
    if kind in ("transformers", "sentence_transformer"):
        # Prefer HF mean-pool; sentence_transformers pulls torchvision and breaks mismatched envs.
        return TransformersEmbedder(model_id)
    raise ValueError(
        f"Unknown embedder {kind!r}. Expected lexical, sentence_transformer, or transformers."
    )
