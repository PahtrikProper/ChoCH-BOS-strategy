"""CLI entry point for the ChoCH/BOS paper trader.

Runs the backtest/optimization pass, writes ``data/best_params.json``, and then
starts the paper-trading loop with the optimal parameters and simulated fills.

Usage:
    python -m choch_bos_strategy_paper_trader
"""

from .main_engine import run


def main() -> None:
    """Execute the orchestrator."""

    run()


if __name__ == "__main__":
    main()
