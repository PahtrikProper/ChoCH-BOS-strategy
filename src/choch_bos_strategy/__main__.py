"""CLI entry point for the ChoCH/BOS live trader.

Runs the backtest/optimization pass, writes ``data/best_params.json``, and then
starts the live trading loop with the optimal parameters.

Usage:
    python -m choch_bos_strategy
"""

from .main_engine import run


def main() -> None:
    """Execute the orchestrator."""

    run()


if __name__ == "__main__":
    main()
