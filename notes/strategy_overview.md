# ChoCH/BOS Strategy Notes

## High-level workflow
- **Data sourcing:** `DataClient` pulls Bybit klines using the configured symbol, category, and aggregation interval. It defaults to a 30‑day backtest window and 1‑minute bars but can be overridden via `TraderConfig`.
- **Backtesting and optimization:** `MainEngine` calls `BacktestEngine.grid_search_with_progress` to evaluate parameter grids for swing lookback, BOS lookback, and Fibonacci pullback bands. The best-performing long configuration is persisted to `data/best_params.json`.
- **Queueing future runs:** Each optimization enqueues metadata in `data/optimization_queue.json` via `OptimizationQueue`, enabling scheduled reruns every two days.
- **Live trading loop:** The live entry point (`python -m choch_bos_strategy` or `python -m choch_bos_strategy.start`) reuses the optimized parameters, streams fresh klines, and manages trade lifecycle through `LiveTradingEngine` with `BuyOrderEngine`/`SellOrderEngine`.

## Strategy logic (long bias)
1. Build 15m swing highs/lows from aggregated data.
2. Compute Fibonacci pullback band between `fib_low` and `fib_high` of the swing range.
3. Require 1m ChoCH + BOS confirmation and demand alignment inside the Fibonacci band.
4. Enter when price taps the upper bound of the fib zone; exit on fib breakdown or demand loss.
5. Apply simulated fees, spread, slippage, and liquidation checks using `order_utils`.

## Key artifacts
- `data/best_params.json`: latest optimized parameter set and summary metrics.
- `data/optimization_queue.json`: queue of scheduled optimizer reruns.

## Quickstart commands
- Optimize + launch live loop: `python -m choch_bos_strategy`
- Optimize only (import the orchestrator): `python -m choch_bos_strategy.start`
- Package entry as module: `python -m choch_bos_strategy`
