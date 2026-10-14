# HF smoke environment

| Piece | Value |
|---|---|
| Python | 3.10.x |
| Host model | `Qwen/Qwen3.5-2B-Base` |
| Embedder | `transformers` → BGE `BAAI/bge-small-en-v1.5` mean-pool |
| Config (reduced) | `configs/smoke_hf_mini.yaml` |
| Config (tool-instruct) | `configs/smoke_hf_mini_instruct.yaml` (preferred for Base + tools) |
| Config (full) | `configs/smoke_hf.yaml` |
| Fallback host | `configs/smoke_hf_mini_qwen25.yaml` (`Qwen2.5-1.5B-Instruct`) |
| Deps | `requirements-hf.txt` (`torch>=2.5`, transformers with `qwen3_5`) |

## Auth (HF_TOKEN)

`.env` is **gitignored** — it is **not** cloned to the instance. Create it there once.

```python
%cd /workspace/mind-viruses/experiments/evolution_epochs
from pathlib import Path
import os
TOKEN = "hf_..."  # paste once; stays on this machine only
Path(".env").write_text(f"HF_TOKEN={TOKEN}\n")
os.environ["HF_TOKEN"] = TOKEN
print("ok", Path(".env").resolve())
```

## Instance cell (writes missing files + Qwen3.5 mini)

```python
%cd /workspace/mind-viruses/experiments/evolution_epochs
TOKEN = "hf_..."
from pathlib import Path
import os
Path(".env").write_text(f"HF_TOKEN={TOKEN}\n")
os.environ["HF_TOKEN"] = TOKEN
Path("requirements-hf.txt").write_text(
    "numpy>=1.24,<2\ntorch>=2.5.0\ntransformers>=4.54.0\naccelerate>=0.33.0\n"
    "huggingface_hub>=0.26.0\nhttpx>=0.27.0,<0.28\npyyaml>=6.0\nmatplotlib>=3.7\n"
)
Path("configs").mkdir(exist_ok=True)
Path("configs/smoke_hf_mini.yaml").write_text("""model_id: Qwen/Qwen3.5-2B-Base
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
ideologies: [whale_welfare]
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
""")
!pip uninstall -y httpx2 sentence-transformers torchaudio torchvision
!pip install -q "httpx>=0.27,<0.28" -U huggingface_hub -r requirements-hf.txt -e .
!python -m steerchain --config configs/smoke_hf_mini.yaml
```

## Hub download crash (`httpx2` / `process() takes no keyword arguments`)

Do **not** pin `huggingface_hub==0.30` — current transformers needs `ResolvedRevision` (hub 1.x).
Keep hub 1.x and bypass httpx2 via requests:

```python
!pip install -q "huggingface_hub>=1.5,<2" requests
# Restart Kernel
%cd /workspace/mind-viruses/experiments/evolution_epochs
# inline patch if src is stale:
import requests
from huggingface_hub import file_download
def http_get(url, temp_file, *, proxies=None, resume_size=0, headers=None, expected_size=None, displayed_filename=None, tqdm_class=None, timeout=None, **kw):
    headers = dict(headers or {}); headers["Accept-Encoding"]="identity"
    if resume_size: headers["Range"]=f"bytes={resume_size}-"
    with requests.get(url, stream=True, headers=headers, proxies=proxies, timeout=timeout or 120) as r:
        r.raise_for_status()
        for c in r.iter_content(8*1024*1024):
            if c: temp_file.write(c)
file_download.http_get = http_get
!python -m steerchain --config configs/smoke_hf_mini.yaml
```

## Explicitly excluded

- `torchaudio` / `torchvision` / `sentence-transformers`

## Fallback without `qwen3_5`

```bash
python -m steerchain --config configs/smoke_hf_mini_qwen25.yaml
```
