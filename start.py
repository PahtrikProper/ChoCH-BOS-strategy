"""Single-script entry point for the Stoch RSI live trading workflow.

Running this file will:
1) Backtest and optimize parameters.
2) Save the best parameters to ``best_params.json``.
3) Launch the live trading loop with those parameters.

Usage (from repo root):
    python -m BTCSTOCHRSI.start
    # or, when invoked directly:
    python BTCSTOCHRSI/start.py
"""

import os
import sys

try:
    # Module execution: prefer relative import
    if __package__:
        from .main_engine import run  # type: ignore
    else:
        raise ImportError
except ImportError:
    # Direct script execution: add repo root to sys.path and import absolutely
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from BTCSTOCHRSI.main_engine import run  # type: ignore


if __name__ == "__main__":
    run()
