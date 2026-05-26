"""Deprecated — use: python scripts/run.py --config configs/dock.yaml quantize"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
config = REPO / "configs" / "dock.yaml"

print("This script is deprecated. Running the new pipeline instead...")
print(f"  python scripts/run.py --config {config} quantize\n")

raise SystemExit(
    subprocess.call(
        [sys.executable, str(REPO / "scripts" / "run.py"), "--config", str(config), "quantize"],
        cwd=str(REPO),
    )
)
