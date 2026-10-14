# evolution_epochs

Hidden-state steering of idea spread on a virus-chain topology (hops × agents × epochs), with optional Qwen-Scope SAE analysis.

See `docs/TZ.md` for the full protocol.

## Smoke (no GPU)

From this directory:

```bash
pip install -e .
python -m pytest tests/ -q
python -m steerchain --config configs/smoke.yaml
```

Or from the repo root:

```bash
PYTHONPATH=experiments/evolution_epochs/src python -m steerchain --config experiments/evolution_epochs/configs/smoke.yaml
```

Artifacts land under `runs/` (gitignored).
