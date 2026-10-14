"""One-shot notebook bootstrap for a GPU instance (no local .env sync).

Paste in a notebook cell and run (creates .env + missing runner if needed):

```python
%cd /workspace/mind-viruses/experiments/evolution_epochs
HF_TOKEN = "hf_..."  # set once per machine; stays on disk in .env (gitignored)
exec(open("scripts/bootstrap_instance_notebook.py", encoding="utf-8").read())
# if this file is missing on the instance, paste the whole script into a cell instead
```

Or, without any new files on disk yet, set the token in the cell and call write_env + run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace/mind-viruses/experiments/evolution_epochs")
if not (ROOT / "src" / "steerchain").is_dir():
    ROOT = Path(__file__).resolve().parents[1]


def write_env(token: str) -> Path:
    path = ROOT / ".env"
    path.write_text(f"HF_TOKEN={token.strip()}\n", encoding="utf-8")
    os.environ["HF_TOKEN"] = token.strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token.strip()
    print(f"wrote {path}")
    return path


def ensure_hf_auth_module() -> None:
    """If the instance tree is stale, drop a minimal hf_auth.py."""
    target = ROOT / "src" / "steerchain" / "hf_auth.py"
    if target.is_file():
        return
    target.write_text(
        '''"""Resolve and apply Hugging Face Hub credentials."""
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
''',
        encoding="utf-8",
    )
    print(f"wrote {target}")


def install_deps() -> None:
    req = ROOT / "requirements-hf.txt"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)])
    subprocess.call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio", "torchvision", "sentence-transformers"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_smoke(config: str = "configs/smoke_hf_mini.yaml") -> int:
    ensure_hf_auth_module()
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from steerchain.hf_auth import apply_hf_token
        from steerchain.config import load_config

        apply_hf_token(load_config(ROOT / config))
    except Exception as exc:  # noqa: BLE001
        print(f"auth preload note: {exc}")
    cmd = [sys.executable, "-m", "steerchain", "--config", str(ROOT / config)]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    # When executed via %run / python, require HF_TOKEN already in env or .env.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and (ROOT / ".env").is_file():
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break
    if not token:
        raise SystemExit(
            "Set HF_TOKEN in the notebook cell first, e.g.\n"
            '  import os; os.environ["HF_TOKEN"] = "hf_..."\n'
            "then re-run this script. .env is gitignored and is NOT cloned."
        )
    write_env(token)
    install_deps()
    raise SystemExit(run_smoke())
