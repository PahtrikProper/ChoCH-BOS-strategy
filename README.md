# ChoCH-BOS Strategy

A minimal repository scaffold for developing a Change of Character (ChoCH) and Break of Structure (BOS) trading strategy. Use this project to research entry/exit rules, design experiments, and track backtesting results.

## Repository layout
- `src/` – strategy code, utilities, and orchestrators (see `src/choch_bos_strategy/` for the active ChoCH/BOS implementation).
- `tests/` – automated checks for strategy logic and helpers.
- `notes/` – research notes, hypotheses, and observations (e.g., `notes/strategy_overview.md`).
- `data/` – generated artifacts such as optimizer outputs (`best_params.json`, `optimization_queue.json`) plus local datasets.

```
.
├── src/choch_bos_strategy/        # Optimizer, live loop, Bybit client utilities
├── data/                          # Runtime artifacts written by the optimizer/live loop
├── notes/                         # Strategy notes and references
└── tests/                         # Space for automated checks
```

## Strategy package
The ChoCH/BOS trading workflow is organized under `src/choch_bos_strategy/` with an executable module entrypoint:

- Optimize then launch live loop: `python -m choch_bos_strategy` or `python -m choch_bos_strategy.start`.
- Core components:
  - `BacktestEngine` runs grid searches over swing/BOS lookbacks and Fibonacci pullback bands.
  - `MainEngine` orchestrates optimization, persists `data/best_params.json`, and queues reruns in `data/optimization_queue.json`.
  - `LiveTradingEngine` streams Bybit klines, applies ChoCH/BOS signals, and routes entries/exits via `BuyOrderEngine` and `SellOrderEngine`.
  - Post-backtest menu: choose to re-run backtests, start live trading, or exit. Live trading requires typing `YES` after a risk disclaimer (unproven strategy; losses are possible; crypto trading is gambling).
  - `paths.py` centralizes repository/data paths to keep artifacts in `data/`.
  - `live.py` configures **isolated margin (tradeMode=1) with 10x leverage** via the Bybit v5 `set-leverage` endpoint for linear contracts; adjust `trade_mode`/`leverage` in that module if needed (see Bybit docs: Position – Set Leverage).
  - **Mainnet only:** `live.py` enforces `https://api.bybit.com` (mainnet) and aborts if DRY_RUN/testnet endpoints are provided.

## Getting started
1. Create a Python virtual environment (e.g., `python -m venv .venv`) and activate it.
2. Install dependencies once they are defined (for example, via `pip install -r requirements.txt`).
3. Add your strategy implementation under `src/` and corresponding tests in `tests/`.
4. Run your test suite (e.g., `pytest`) to validate any changes.

## Next steps
- Tune parameters (`swing_lookback`, `bos_lookback`, `fib_low`, `fib_high`) in `TraderConfig` to guide optimization.
- Extend data loaders if you need alternate feeds or caching.
- Add tests around entry/exit conditions and order simulation utilities.
- Capture experiment outcomes and hypotheses in `notes/` (see `notes/strategy_overview.md` for the current strategy outline).

## Running the strategy
1. Ensure dependencies are installed (minimum: `pandas`, `numpy`, `requests`, `tqdm`).
2. From the repo root, run the optimizer + live loop:
   ```bash
   python -m choch_bos_strategy
   # or explicitly
   python -m choch_bos_strategy.start
   ```
3. Artifacts:
   - Best params: `data/best_params.json`
   - Optimization queue: `data/optimization_queue.json`
4. Live trading uses Bybit klines; provide the desired symbol/category via `TraderConfig` (defaults to `BTCUSDT` spot).
5. Margin mode and leverage: `src/choch_bos_strategy/live.py` requests **isolated 10x** (`trade_mode=1`, `leverage=10`) using `/v5/position/set-leverage`; ensure your account supports the requested mode before running.
6. Safety: the live loop is **mainnet-only** and will exit if pointed at testnet or if `DRY_RUN=true`.
7. Live-trade consent: when prompted, type `YES` to accept the risk disclaimer (unproven strategy; losses possible; crypto trading is gambling) or return to the menu to stay in backtest mode.

## Credits
- Strategy inspiration and workflow steps were shared by a community member on Reddit: [Forexstrategy thread (comment)](https://www.reddit.com/r/Forexstrategy/comments/1oh8ukp/comment/nvnwobr/?context=3). Thanks for outlining the ChoCH/BOS process and fib zone guidance.

For a conceptual overview of the ChoCH/BOS logic and workflow, see `notes/strategy_overview.md`.
