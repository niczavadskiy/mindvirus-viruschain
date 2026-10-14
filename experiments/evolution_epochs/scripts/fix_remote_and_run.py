"""Paste-friendly remote fix: write missing files, pin Hub/httpx, run Qwen3.5-2B mini.

Notebook (one cell):

```python
%cd /workspace/mind-viruses/experiments/evolution_epochs
TOKEN = "hf_..."  # required
exec(open("scripts/fix_remote_and_run.py", encoding="utf-8").read())
```

If this file is missing on the instance, paste the whole file into a cell and set TOKEN above the exec, or run:

```python
TOKEN = "hf_..."
# then paste the body of main() below
```
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path("/workspace/mind-viruses/experiments/evolution_epochs")
if not (ROOT / "src" / "steerchain").is_dir():
    ROOT = Path(__file__).resolve().parents[1]

REQUIREMENTS = dedent(
    """\
    numpy>=1.24,<2
    torch>=2.5.0
    transformers>=4.54.0
    accelerate>=0.33.0
    huggingface_hub>=0.26.0
    httpx>=0.27.0,<0.28
    pyyaml>=6.0
    matplotlib>=3.7
    """
)

MINI_YAML = dedent(
    """\
    model_id: Qwen/Qwen3.5-2B-Base
    judge_model_id: null
    embedding_model_id: BAAI/bge-small-en-v1.5
    backend: hf
    embedder: transformers

    n_hops: 10
    agents_per_hop: 2
    turns_per_hop: 2
    max_tool_loops: 3
    require_tool_action: true
    tool_retries: 1
    native_tools: true
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
    hf_token: null

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
)

HF_AUTH = dedent(
    '''\
    """Resolve and apply Hugging Face Hub credentials."""
    from __future__ import annotations
    import os
    from pathlib import Path

    def _load_dotenv(path: Path) -> None:
        if not path.is_file():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value

    def resolve_hf_token(config=None):
        root = Path(__file__).resolve().parents[2]
        for path in (root / ".env", root / "configs" / ".env"):
            _load_dotenv(path)
        if config is not None:
            token = (getattr(config, "hf_token", None) or "").strip()
            if token:
                return token
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            value = (os.environ.get(key) or "").strip()
            if value:
                return value
        return None

    def apply_hf_token(config=None):
        token = resolve_hf_token(config)
        if not token:
            return None
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass
        return token
    '''
)


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def write_files(token: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "requirements-hf.txt").write_text(REQUIREMENTS, encoding="utf-8")
    cfg = ROOT / "configs"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "smoke_hf_mini.yaml").write_text(MINI_YAML, encoding="utf-8")
    (cfg / "smoke_hf_mini_qwen25.yaml").write_text(MINI_YAML, encoding="utf-8")
    (ROOT / ".env").write_text(f"HF_TOKEN={token.strip()}\n", encoding="utf-8")
    auth = ROOT / "src" / "steerchain" / "hf_auth.py"
    auth.parent.mkdir(parents=True, exist_ok=True)
    if not auth.is_file():
        auth.write_text(HF_AUTH, encoding="utf-8")
    print("wrote requirements-hf.txt, smoke_hf_mini*.yaml, .env, hf_auth.py (if missing)")


def fix_hub_stack() -> None:
    # httpx2 / brotli decoder bug: process() takes no keyword arguments
    subprocess.call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "httpx2"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _pip("install", "-q", "--upgrade", "huggingface_hub")
    _pip("install", "-q", "httpx>=0.27.0,<0.28")
    _pip("install", "-q", "-r", str(ROOT / "requirements-hf.txt"))
    _pip("install", "-q", "-e", str(ROOT))
    subprocess.call(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torchaudio",
            "torchvision",
            "sentence-transformers",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main(token: str | None = None) -> int:
    token = (token or os.environ.get("TOKEN") or os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            "Set TOKEN in the notebook cell first:\n"
            '  TOKEN = "hf_..."\n'
            "then re-run this script."
        )
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    os.chdir(ROOT)
    write_files(token)
    fix_hub_stack()
    try:
        import torch

        print("cuda", torch.cuda.is_available(), "torch", torch.__version__)
    except Exception as exc:  # noqa: BLE001
        print("torch check:", exc)

    cfg = ROOT / "configs" / "smoke_hf_mini.yaml"
    print("USING", cfg, "→ Qwen/Qwen3.5-2B-Base")
    return subprocess.call(
        [sys.executable, "-m", "steerchain", "--config", str(cfg)],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    # Notebook: TOKEN = "hf_..."; exec(open(...).read()) puts TOKEN in globals.
    tok = globals().get("TOKEN") or os.environ.get("HF_TOKEN")
    raise SystemExit(main(tok if isinstance(tok, str) else None))
