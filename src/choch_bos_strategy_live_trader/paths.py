"""Shared filesystem paths for the ChoCH/BOS strategy package."""

from __future__ import annotations

from pathlib import Path

# Package location: /workspace/.../src/choch_bos_strategy_live_trader
PACKAGE_ROOT = Path(__file__).resolve().parent
# Repository root lives one level above the src directory.
REPO_ROOT = PACKAGE_ROOT.parents[1]
DATA_DIR = REPO_ROOT / "data"

# Ensure the data directory always exists for JSON artifacts created at runtime.
DATA_DIR.mkdir(parents=True, exist_ok=True)
