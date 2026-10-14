# evolution_epochs

Hidden-state steering of idea spread on a virus-chain topology (hops × agents × epochs), with optional Qwen-Scope SAE analysis.

See `docs/TZ.md` for the full protocol. HF env lock: `docs/ENV_HF.md`.

## Smoke configs

| Config | Backend | What it checks |
|---|---|---|
| `configs/smoke.yaml` | `stub` | Protocol only (no LLM dialogues) |
| `configs/smoke_hf_mini.yaml` | `hf` | Reduced batch + HS (`Qwen3.5-2B-Base`) |
| `configs/smoke_hf_mini_instruct.yaml` | `hf` | Same Base host, hardened TOOL/ARGS prompts + more retries |
| `configs/smoke_hf_mini_qwen25.yaml` | `hf` | Fallback host if no `qwen3_5` support |
| `configs/smoke_hf.yaml` | `hf` | Full paper topology + `Qwen3.5-2B-Base` |

Artifacts: `runs/<name>/<YYYYMMDD_HHMMSS>/` (gitignored). Each CLI/notebook run gets a fresh timestamped subfolder. Pass `--no-stamp` to write directly into `run_dir`. HS: `*/states/*.npy`.

## Stub smoke (no GPU)

```bash
cd experiments/evolution_epochs
pip install -e .
python -m pytest tests/ -q
python -m steerchain --config configs/smoke.yaml
# → runs/smoke/20261014_123719/
```

## Real smoke — locked env / GPU instance

```bash
cd experiments/evolution_epochs
cp .env.example .env   # set HF_TOKEN=...
bash scripts/run_instance_hf_mini.sh
# → runs/smoke_hf_mini/YYYYMMDD_HHMMSS/
```

Notebook on the instance:

```python
%cd /workspace/mind-viruses/experiments/evolution_epochs
%run scripts/run_instance_hf_mini.py
```

Freeze exact versions from a working kernel:

```python
!python scripts/freeze_hf_env.py
```

## Full paper topology

```bash
python -m steerchain --config configs/smoke_hf.yaml
```

Fallback if the instance has no `qwen3_5` support:

```bash
python -m steerchain --config configs/smoke_hf_mini_qwen25.yaml
```
