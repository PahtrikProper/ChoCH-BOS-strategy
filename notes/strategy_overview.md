# ChoCH/BOS Strategy Notes

## High-level workflow
- **Data sourcing:** Each package’s `DataClient` pulls Bybit klines using the configured symbol, category, and aggregation interval. Defaults: `BTCUSDT` or `SOLUSDT` on spot with a 30‑day window and 1‑minute bars (override via each package’s `TraderConfig`).
- **Backtesting and optimization:** `MainEngine` invokes `BacktestEngine.grid_search_with_progress` across swing lookback, BOS lookback, and Fibonacci pullback bands (`fib_low`, `fib_high`). The top long configuration is saved to `data/best_params.json`.
- **Queueing future runs:** `OptimizationQueue` appends each optimization summary to `data/optimization_queue.json`, targeting a 2‑day cadence for reruns.
- **Live trading loop:** The live packages (`choch_bos_strategy_btc_live`, `choch_bos_strategy_sol_live`) reuse optimized parameters, stream fresh klines, and manage the trade lifecycle via `LiveTradingEngine` with `BuyOrderEngine`/`SellOrderEngine`.
- **Margin/leverage:** Live variants set Bybit **isolated mode (tradeMode=1) with 10x leverage** using `/v5/position/set-leverage`. Adjust `trade_mode`/`leverage` inside each package’s `live.py` if your account requires different settings.
- **Mainnet only:** Live variants enforce `https://api.bybit.com` and abort if DRY_RUN/testnet is supplied.
- **Post-backtest menu:** After optimization, the CLI offers a menu (re-run backtests, start live trading, or exit). Live trading requires typing `YES` after a risk disclaimer stating the strategy/code are unproven and that crypto trading is gambling and can lead to loss.

## Strategy logic (long bias)
1. Build 15m swing highs/lows from aggregated data.
2. Compute Fibonacci pullback band between `fib_low` and `fib_high` of the swing range.
3. Require 1m ChoCH + BOS confirmation and demand alignment inside the Fibonacci band.
4. Enter when price taps the upper bound of the fib zone; exit on fib breakdown or demand loss.
5. Apply simulated fees, spread, slippage, and liquidation checks using `order_utils` (includes Bybit-style fee model and liquidation math).

## Components and entry points
- **Optimizer + orchestration:** `main_engine.py` within each package (e.g., `src/ChoCH-BOS-strategy-BTC-LIVE/choch_bos_strategy_btc_live/main_engine.py`).
- **Backtester:** `backtest_engine.py` within each package.
- **Live trading loop:** `live.py` within the live packages.
- **Order helpers:** `order_utils.py` within each package.
- **Paths and artifacts:** `paths.py` in every package keeps outputs in `data/`.

## Key artifacts
- `data/best_params.json`: latest optimized parameter set and summary metrics.
- `data/optimization_queue.json`: queue of scheduled optimizer reruns.

## Quickstart commands
- BTC live: `PYTHONPATH=src/ChoCH-BOS-strategy-BTC-LIVE python -m choch_bos_strategy_btc_live`
- SOL live: `PYTHONPATH=src/ChoCH-BOS-strategy-SOL-LIVE python -m choch_bos_strategy_sol_live`
- BTC paper trader: `PYTHONPATH=src/ChoCH-BOS-strategy-BTC-PAPER-TRADER python -m choch_bos_strategy_btc_paper_trader`
- SOL paper trader: `PYTHONPATH=src/ChoCH-BOS-strategy-SOL-PAPER-TRADER python -m choch_bos_strategy_sol_paper_trader`
