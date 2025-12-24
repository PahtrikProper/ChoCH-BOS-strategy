"""CLI entry point for the live trader.

Runs the backtest/optimization pass, writes ``best_params.json``, and then
starts the live trading loop with the optimal parameters.

Usage:
    python -m BTCSTOCHRSI
"""

from .main_engine import run


def main() -> None:
    """Execute the orchestrator."""

    run()


if __name__ == "__main__":
    main()
