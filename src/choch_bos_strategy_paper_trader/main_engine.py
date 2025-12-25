from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .backtest_engine import BacktestEngine, StrategyParams, summarize_long_results
from .buy_order_engine import BuyOrderEngine
from .config import TraderConfig
from .data_client import DataClient
from .order_utils import PositionState, bybit_fee_fn, mark_to_market_equity
from .optimization_queue import OptimizationQueue
from .sell_order_engine import SellOrderEngine
from .paths import DATA_DIR


class LiveTradingEngine:
    def __init__(self, config: TraderConfig, long_params: pd.Series, short_params: pd.Series, long_results: Dict[str, float]):
        self.config = config
        self.long_params = long_params
        self.short_params = pd.Series(dtype=float)
        self.long_results = long_results
        self.data_client = DataClient(config)
        self.buy_engine = BuyOrderEngine(config)
        self.sell_engine = SellOrderEngine(config)
        self.position: Optional[PositionState] = None
        self.tradelog = []
        self.equity = config.starting_balance
        self._last_long_signal_ts: Optional[pd.Timestamp] = None

    def _prepare_live_dataframe(self) -> pd.DataFrame:
        df = self.data_client.fetch_bybit_bars(days=self.config.live_history_days, interval_minutes=self.config.agg_minutes)
        swing_lookback = int(self.long_params.get("swing_lookback", self.config.swing_lookback))
        bos_lookback = int(self.long_params.get("bos_lookback", self.config.bos_lookback))
        fib_low = float(self.long_params.get("fib_low", self.config.fib_low))
        fib_high = float(self.long_params.get("fib_high", self.config.fib_high))

        data = df.copy().sort_index()
        ht = (
            data.resample("15T")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
        )
        ht["swing_high"] = ht["High"].rolling(swing_lookback).max()
        ht["swing_low"] = ht["Low"].rolling(swing_lookback).min()
        ht_sw_high = ht["swing_high"].reindex(data.index, method="ffill")
        ht_sw_low = ht["swing_low"].reindex(data.index, method="ffill")

        range_height = ht_sw_high - ht_sw_low
        fib_upper = ht_sw_low + fib_high * range_height
        fib_lower = ht_sw_low + fib_low * range_height
        in_fib = (data["Close"] <= fib_upper) & (data["Close"] >= fib_lower)

        bos_up = data["Close"] > ht_sw_high.shift(1)
        choch_up = (data["High"] > data["High"].shift(bos_lookback)) & (data["Low"] > data["Low"].shift(bos_lookback))
        demand_trigger = data["Low"].rolling(bos_lookback).min()
        demand_ok = (demand_trigger >= ht_sw_low) & (demand_trigger >= fib_lower)
        structure_conf = bos_up & choch_up

        data["entry_signal"] = in_fib & structure_conf & demand_ok & (data["Low"] <= fib_upper)
        data["exit_signal"] = (data["Close"] < fib_lower) | (data["Close"] < demand_trigger)
        data["tradable"] = data[["entry_signal", "exit_signal"]].notna().all(axis=1)
        return data

    def _log_live_summary(self):
        trades_df = pd.DataFrame(self.tradelog)
        total_pnl = trades_df["pnl"].sum()
        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
        print("\n==== LIVE SUMMARY (LONG + SHORT) ====")
        print(f"Total trades: {total_trades}")
        print(f"Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Average win: {avg_win:.2f}")
        print(f"Average loss: {avg_loss:.2f}")
        print("=====================================\n")

    def _print_entry(self, nowstr: str, direction: str, position: PositionState):
        print(
            f"{nowstr} | ENTRY ({direction.upper()}) @ {position.entry_price:.2f} | "
            f"qty={position.qty:.3f} | LIQ={position.liq_price:.2f}"
        )

    def _handle_exit(self, row: pd.Series, nowstr: str):
        if not self.position:
            return

        exit_cond = False
        liq_hit = False
        if self.position.side == "long":
            if row["Low"] <= self.position.liq_price:  # type: ignore[operator]
                exit_cond = True
                liq_hit = True
            elif row.get("exit_signal"):
                exit_cond = True
        elif self.position.side == "short":
            if row["High"] >= self.position.liq_price:  # type: ignore[operator]
                exit_cond = True
                liq_hit = True
            elif row.get("exit_signal"):
                exit_cond = True

        if exit_cond and self.position:
            if liq_hit:
                exit_price = self.position.liq_price
                net_pnl = -(self.position.margin_used)
                # equity already represents remaining cash after margin was locked and fees were paid
                status = "LIQUIDATED"
            else:
                exit_price = row["Close"]
                if self.position.side == "long":
                    gross = (exit_price - self.position.entry_price) * self.position.qty  # type: ignore[operator]
                else:
                    gross = (self.position.entry_price - exit_price) * self.position.qty  # type: ignore[operator]
                exit_fee = bybit_fee_fn(self.position.qty * exit_price, self.config)  # type: ignore[arg-type]
                net_pnl = gross - exit_fee
                # Return margin to cash equity plus realized PnL (entry fee already accounted for on entry)
                self.equity += self.position.margin_used + net_pnl
                status = "LOSS" if net_pnl < 0 else "EXIT"

            self.tradelog.append(
                {
                    "entry_time": self.position.entry_bar_time,
                    "exit_time": row.name,
                    "side": self.position.side.upper(),
                    "entry_price": self.position.entry_price,
                    "exit_price": exit_price,
                    "qty": self.position.qty,
                    "pnl": net_pnl,
                    "status": status,
                    "equity": self.equity,
                    "margin_used": self.position.margin_used,
                }
            )
            print(f"{nowstr} | EXIT @ {exit_price:.2f} | {status} | NetPnL={net_pnl:.2f} | Equity={self.equity:.2f}")
            self.position = None

    def run(self):
        print(
            "\n--- Live 15m swing → fib 60-70% + 1m ChoCH/BOS trader "
            f"HT swing_lb={self.long_params.get('swing_lookback', self.config.swing_lookback)}, "
            f"1m bos_lb={self.long_params.get('bos_lookback', self.config.bos_lookback)}, "
            f"fib={self.long_params.get('fib_low', self.config.fib_low):.2f}-{self.long_params.get('fib_high', self.config.fib_high):.2f} | "
            f"{self.config.agg_minutes}m ---\n"
        )

        while True:
            try:
                data = self._prepare_live_dataframe()
                min_required = (
                    max(
                        int(self.long_params.get("swing_lookback", self.config.swing_lookback)) * 15,
                        int(self.long_params.get("bos_lookback", self.config.bos_lookback)) * 2,
                    )
                    + self.config.min_history_padding
                )
                if len(data) < min_required:
                    print("Waiting for enough bars...")
                    time.sleep(2)
                    continue

                row = data.iloc[-1]
                nowstr = time.strftime("%Y-%m-%d %H:%M", time.gmtime())

                if self.long_results["Total PnL"] <= 0:
                    print(f"{nowstr} | NO EDGE detected by optimizer – standing aside.")
                    time.sleep(60 * self.config.agg_minutes)
                    continue

                if not self.position:
                    if self.buy_engine.should_enter(row, None) and row.name != self._last_long_signal_ts:
                        position, status, entry_fee, _, margin_used = self.buy_engine.open_position(
                            row,
                            available_usdt=self.equity,
                        )
                        if status == "rejected" or not position:
                            print(f"{nowstr} | ENTRY (LONG) rejected – simulated failure (no trade)")
                        elif status == "insufficient_funds":
                            print(f"{nowstr} | ENTRY (LONG) skipped – insufficient USDT balance")
                        else:
                            self.equity -= (entry_fee + margin_used)
                            self.position = position
                            self._print_entry(nowstr, "long", position)
                            self._last_long_signal_ts = row.name
                    else:
                        print(f"{nowstr} | NO TRADE – waiting for a new signal.")

                self._handle_exit(row, nowstr)

                marked_equity = mark_to_market_equity(
                    self.equity,
                    1 if self.position and self.position.side == "long" else (-1 if self.position and self.position.side == "short" else 0),
                    self.position.entry_price if self.position else None,
                    self.position.qty if self.position else 0,
                    row["Close"],
                    self.position.margin_used if self.position else 0.0,
                )
                if self.position:
                    print(
                        f"{nowstr} | Equity (realized/unrealized): {self.equity:.2f} / {marked_equity:.2f} | Trades: {len(self.tradelog)}"
                    )
                else:
                    print(f"{nowstr} | Equity: {self.equity:.2f} | Trades: {len(self.tradelog)}")

                if len(self.tradelog) > 0:
                    self._log_live_summary()

                time.sleep(60 * self.config.agg_minutes)

            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as exc:  # noqa: BLE001
                print("Exception:", exc)
                time.sleep(2)


class MainEngine:
    def __init__(self, config: Optional[TraderConfig] = None, best_params_path: Path | str | None = None):
        self.config = config or TraderConfig()
        self.data_client = DataClient(self.config)
        self.backtest_engine = BacktestEngine(self.config)
        self.best_params_path = Path(best_params_path) if best_params_path else DATA_DIR / "best_params.json"
        self.best_params_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimization_queue = OptimizationQueue()

    def log_config(self):
        print("\n===== LIVE TRADER CONFIGURATION =====")
        print(self.config.as_log_string())
        print("======================================\n")

    def _normalize_row(self, row: pd.Series) -> Dict:
        normalized: Dict[str, float | int | str] = {}
        for key, value in row.to_dict().items():
            if isinstance(value, (np.floating, np.integer)):
                normalized[key] = value.item()
            else:
                normalized[key] = value
        return normalized

    def save_best_params(self, best_long: pd.Series, long_results: Dict[str, float]) -> None:
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "symbol": self.config.symbol,
            "category": self.config.category,
            "agg_minutes": self.config.agg_minutes,
            "leverage": self.config.leverage,
            "spread_bps": self.config.spread_bps,
            "slippage_bps": self.config.slippage_bps,
            "order_reject_prob": self.config.order_reject_prob,
            "max_fill_latency": self.config.max_fill_latency,
            "long_params": self._normalize_row(best_long),
            "short_params": {},
            "long_results": long_results,
        }
        self.best_params_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved optimal parameters to {self.best_params_path.resolve()}")

    def queue_best_params(self, best_long: pd.Series, long_results: Dict[str, float], elapsed_seconds: float) -> Dict:
        queued_at = datetime.utcnow()
        ready_at = queued_at + timedelta(days=2)
        payload = {
            "symbol": self.config.symbol,
            "category": self.config.category,
            "agg_minutes": self.config.agg_minutes,
            "backtest_days": self.config.backtest_days,
            "starting_balance": self.config.starting_balance,
            "long_params": self._normalize_row(best_long),
            "short_params": {},
            "long_results": long_results,
        }
        queued_item = self.optimization_queue.enqueue(
            queued_at=queued_at,
            ready_at=ready_at,
            elapsed_seconds=elapsed_seconds,
            payload=payload,
        )
        print(f"Queued next optimization run for ~{ready_at.isoformat()}Z (elapsed {elapsed_seconds:.2f}s; cadence=2d).")
        print(f"Queue file: {self.optimization_queue.queue_path.resolve()}")
        return queued_item

    def run_backtests(self) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        print(f"Fetching data and running optimizer on {self.config.agg_minutes}m bars...")
        start_time = time.monotonic()
        df = self.data_client.fetch_bybit_bars(interval_minutes=self.config.agg_minutes, days=self.config.backtest_days)
        required_bars = max(self.config.swing_lookback_range) * 15 + max(self.config.bos_lookback_range) * 2 + self.config.min_history_padding
        if len(df) < required_bars:
            raise ValueError(f"Not enough candles fetched for optimizer warmup: need {required_bars}, got {len(df)}")

        dfres_long, dfres_short = self.backtest_engine.grid_search_with_progress(df)
        best_long = dfres_long.sort_values("pnl_pct", ascending=False).head(1).drop(columns=["drawdown"])
        long_results = summarize_long_results(best_long, self.config.starting_balance)

        print(f"\n==================== BEST LONG PARAMETERS ({self.config.agg_minutes}m) ====================")
        print(best_long.to_string(index=False))
        print("\n============== BEST RESULTS (LONG ONLY) ==============")
        for k, v in long_results.items():
            print(f"{k}: {v}")
        print("==================================================================\n")

        self.save_best_params(best_long.iloc[0], long_results)
        elapsed_seconds = time.monotonic() - start_time
        self.queue_best_params(best_long.iloc[0], long_results, elapsed_seconds)

        # Re-run best params once to capture trade-by-trade details for reporting
        best_series = best_long.iloc[0]
        best_params = StrategyParams(
            int(best_series["swing_lookback"]),
            int(best_series["bos_lookback"]),
            float(best_series["fib_low"]),
            float(best_series["fib_high"]),
            "long",
        )
        _, trades_df = self.backtest_engine.run_backtest_with_trades(df, best_params)
        if not trades_df.empty:
            cols = ["entry_time", "exit_time", "entry_price", "exit_price", "pnl_value", "pnl_pct", "qty"]
            print("\n==== TRADES (LONG, BEST PARAMS) ====")
            print(trades_df[cols].to_string(index=False))
            print("====================================\n")
        else:
            print("\nNo trades recorded for best parameters.\n")

        return best_long, dfres_long, long_results

    def run(self):
        self.log_config()
        best_long, _, long_results = self.run_backtests()

        print(
            "\nStarting paper-trading loop with simulated fills, slippage, spread, and fees.\n"
            f"Initial paper balance: {self.config.starting_balance} USDT\n"
        )
        LiveTradingEngine(
            config=self.config,
            long_params=best_long.iloc[0],
            short_params=pd.Series(dtype=float),
            long_results=long_results,
        ).run()


def run():
    MainEngine().run()


if __name__ == "__main__":
    run()
