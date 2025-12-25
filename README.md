# ChoCH-BOS Strategy

A Change of Character (ChoCH) and Break of Structure (BOS) trading strategy, split into four self-contained packages (BTC vs SOL, live vs paper). Use this repository to research entry/exit rules, backtest parameter sets, and run either live or simulated trading loops.

## Repository layout
- `src/` – four parallel strategy packages:
  - `src/ChoCH-BOS-strategy-BTC-LIVE/choch_bos_strategy_btc_live/` – **BTC live trading** (optimizer + Bybit live loop).
  - `src/ChoCH-BOS-strategy-SOL-LIVE/choch_bos_strategy_sol_live/` – **SOL live trading**.
  - `src/ChoCH-BOS-strategy-BTC-PAPER-TRADER/choch_bos_strategy_btc_paper_trader/` – **BTC paper trading** (simulated fills, spreads, slippage, fees; no live orders).
  - `src/ChoCH-BOS-strategy-SOL-PAPER-TRADER/choch_bos_strategy_sol_paper_trader/` – **SOL paper trading** (simulated).
- `data/` – runtime artifacts written by the engines (`best_params.json`, `optimization_queue.json`).
- `notes/` – research notes and workflow references (see `notes/strategy_overview.md`).
- `tests/` – placeholder for automated checks (currently empty).

```
.
├── src/ChoCH-BOS-strategy-BTC-LIVE/choch_bos_strategy_btc_live/              # Optimizer + live trading (BTC)
├── src/ChoCH-BOS-strategy-SOL-LIVE/choch_bos_strategy_sol_live/              # Optimizer + live trading (SOL)
├── src/ChoCH-BOS-strategy-BTC-PAPER-TRADER/choch_bos_strategy_btc_paper_trader/  # Optimizer + simulator (BTC)
├── src/ChoCH-BOS-strategy-SOL-PAPER-TRADER/choch_bos_strategy_sol_paper_trader/  # Optimizer + simulator (SOL)
├── data/                          # JSON artifacts produced at runtime
├── notes/                         # Strategy notes
└── tests/                         # (empty placeholder)
```

## Entry points
Each package ships two entry points: the package module itself (`python -m ...`) and an explicit `.start` module. Set `PYTHONPATH` to the corresponding package directory under `src/` before running:

- BTC live:
  ```bash
  PYTHONPATH=src/ChoCH-BOS-strategy-BTC-LIVE python -m choch_bos_strategy_btc_live
  # or
  PYTHONPATH=src/ChoCH-BOS-strategy-BTC-LIVE python -m choch_bos_strategy_btc_live.start
  ```
- SOL live:
  ```bash
  PYTHONPATH=src/ChoCH-BOS-strategy-SOL-LIVE python -m choch_bos_strategy_sol_live
  PYTHONPATH=src/ChoCH-BOS-strategy-SOL-LIVE python -m choch_bos_strategy_sol_live.start
  ```
- BTC paper trader (simulated):
  ```bash
  PYTHONPATH=src/ChoCH-BOS-strategy-BTC-PAPER-TRADER python -m choch_bos_strategy_btc_paper_trader
  PYTHONPATH=src/ChoCH-BOS-strategy-BTC-PAPER-TRADER python -m choch_bos_strategy_btc_paper_trader.start
  ```
- SOL paper trader (simulated):
  ```bash
  PYTHONPATH=src/ChoCH-BOS-strategy-SOL-PAPER-TRADER python -m choch_bos_strategy_sol_paper_trader
  PYTHONPATH=src/ChoCH-BOS-strategy-SOL-PAPER-TRADER python -m choch_bos_strategy_sol_paper_trader.start
  ```

## Engine behavior (common across packages)
- `BacktestEngine` runs grid searches over swing/BOS lookbacks and Fibonacci pullback bands (`fib_low`, `fib_high`).
- `MainEngine` orchestrates optimization, persists `data/best_params.json`, and queues reruns in `data/optimization_queue.json`.
- `LiveTradingEngine` (live variants) streams Bybit klines, applies ChoCH/BOS signals, and routes entries/exits via `BuyOrderEngine` and `SellOrderEngine`.
- The paper-trading packages reuse the ChoCH/BOS logic but simulate fills (spread, slippage, Bybit-style fees, random rejections, latency) with a **400 USDT** default balance.
- `paths.py` centralizes repository/data paths so all artifacts land in `data/`.

## Safety and configuration
- **Bybit mainnet only for live variants:** `live.py` enforces `https://api.bybit.com` and aborts when pointed at testnet or when `DRY_RUN` is set.
- Margin mode and leverage default to **isolated 10x** via `/v5/position/set-leverage`; adjust `trade_mode`/`leverage` inside each `live.py` if your account requires different settings.
- Live trading prompts for an explicit `YES` acknowledgement after the risk disclaimer (strategy is unproven; crypto trading can lead to losses).

## Getting started
1. Create a Python virtual environment (e.g., `python -m venv .venv`) and activate it.
2. Install dependencies once they are defined (for example, via `pip install -r requirements.txt`).
3. Choose a package (BTC/SOL, live/paper) and run it using the commands above with the appropriate `PYTHONPATH`.
4. Outputs land in `data/best_params.json` and `data/optimization_queue.json`; review/update `TraderConfig` inside each package to target your symbol/category.

## Next steps
- Tune `swing_lookback`, `bos_lookback`, `fib_low`, and `fib_high` in each package’s `TraderConfig` to guide optimization.
- Extend `DataClient` if you need alternate feeds or caching.
- Add tests under `tests/` to cover entry/exit conditions, order simulation utilities, and CLI flows.
- Capture experiment outcomes and hypotheses in `notes/` (see `notes/strategy_overview.md` for the current strategy outline).

## Credits
- Strategy inspiration and workflow steps were shared by a community member on Reddit: [Forexstrategy thread (comment)](https://www.reddit.com/r/Forexstrategy/comments/1oh8ukp/comment/nvnwobr/?context=3). Thanks for outlining the ChoCH/BOS process and fib zone guidance.

For a conceptual overview of the ChoCH/BOS logic and workflow, see `notes/strategy_overview.md`.
