# ChoCH/BOS Strategy Notes

## High-level workflow
- **Data sourcing:** `DataClient` pulls Bybit klines using the configured symbol, category, and aggregation interval. Defaults: `BTCUSDT` spot, 30‑day window, 1‑minute bars. Override via `TraderConfig`.
- **Backtesting and optimization:** `MainEngine` invokes `BacktestEngine.grid_search_with_progress` across swing lookback, BOS lookback, and Fibonacci pullback bands (`fib_low`, `fib_high`). The top long configuration is saved to `data/best_params.json`.
- **Queueing future runs:** `OptimizationQueue` appends each optimization summary to `data/optimization_queue.json`, targeting a 2‑day cadence for reruns.
- **Live trading loop:** The entry point (`python -m choch_bos_strategy` or `python -m choch_bos_strategy.start`) reuses optimized parameters, streams fresh klines, and manages trade lifecycle via `LiveTradingEngine` with `BuyOrderEngine`/`SellOrderEngine`.
- **Margin/leverage:** `live.py` sets Bybit **isolated mode (tradeMode=1) with 10x leverage** using `/v5/position/set-leverage`. Adjust `trade_mode`/`leverage` in that file if your account requires different settings.
- **Mainnet only:** `live.py` enforces `https://api.bybit.com` and aborts if DRY_RUN/testnet is supplied.

## Strategy logic (long bias)
1. Build 15m swing highs/lows from aggregated data.
2. Compute Fibonacci pullback band between `fib_low` and `fib_high` of the swing range.
3. Require 1m ChoCH + BOS confirmation and demand alignment inside the Fibonacci band.
4. Enter when price taps the upper bound of the fib zone; exit on fib breakdown or demand loss.
5. Apply simulated fees, spread, slippage, and liquidation checks using `order_utils` (includes Bybit-style fee model and liquidation math).

## Components and entry points
- **Optimizer + live orchestration:** `src/choch_bos_strategy/main_engine.py`
- **Backtester:** `src/choch_bos_strategy/backtest_engine.py`
- **Live trading loop:** `src/choch_bos_strategy/live.py`
- **Order helpers:** `src/choch_bos_strategy/order_utils.py`
- **Paths and artifacts:** `src/choch_bos_strategy/paths.py` keeps outputs in `data/`.

## Key artifacts
- `data/best_params.json`: latest optimized parameter set and summary metrics.
- `data/optimization_queue.json`: queue of scheduled optimizer reruns.

## Quickstart commands
- Optimize + launch live loop: `python -m choch_bos_strategy`
- Explicit orchestrator entry: `python -m choch_bos_strategy.start`
- Package entry as module (imports `main_engine.run`): `python -m choch_bos_strategy`
