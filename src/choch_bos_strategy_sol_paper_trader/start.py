"""Single-script entry point for the ChoCH/BOS paper-trading workflow.

Running this file will:
1) Backtest and optimize parameters.
2) Save the best parameters to ``data/best_params.json``.
3) Launch the paper-trading loop with those parameters.

Usage (from repo root):
    python -m choch_bos_strategy_sol_paper_trader.start
    # or, when invoked directly:
    python src/choch_bos_strategy_sol_paper_trader/start.py
"""

from __future__ import annotations

import os
import sys

try:
    if __package__:
        from .main_engine import run  # type: ignore
    else:
        raise ImportError
except ImportError:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from choch_bos_strategy_sol_paper_trader.main_engine import run  # type: ignore


if __name__ == "__main__":
    run()
