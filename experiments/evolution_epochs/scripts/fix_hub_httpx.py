"""Fix Hub downloads without breaking transformers.

Problem:
  - huggingface_hub v2 → httpx2 → zlib decoder crash on this image
  - huggingface_hub 0.30 → too old for current transformers (ResolvedRevision)

Solution:
  - Install hub in the range transformers wants (1.x)
  - Monkeypatch file_download.http_get to use requests (bypass httpx2)

```python
%cd /workspace/mind-viruses/experiments/evolution_epochs
%run scripts/fix_hub_httpx.py
# Restart Kernel, then run the steerchain cell that calls prepare_hub_downloads()
```
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    py = sys.executable
    # Do NOT pin hub to 0.30 — transformers 5.x needs ResolvedRevision (hub 1.x).
    subprocess.check_call([py, "-m", "pip", "install", "-q", "requests"])
    subprocess.check_call(
        [
            py,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "huggingface_hub>=1.5,<2",
            "transformers>=4.54",
            "accelerate",
        ]
    )
    # Ensure patch module exists even on a stale checkout.
    patch = ROOT / "src" / "steerchain" / "hub_patch.py"
    if not patch.is_file():
        print("WARNING: hub_patch.py missing — paste prepare_hub_downloads into the next cell")
    print("hub/transformers upgraded.")
    print("Next: Restart Kernel, then BEFORE steerchain:")
    print("  from steerchain.hub_patch import prepare_hub_downloads; prepare_hub_downloads()")
    print("Or use the self-contained notebook cell in docs/ENV_HF.md")


if __name__ == "__main__":
    main()
