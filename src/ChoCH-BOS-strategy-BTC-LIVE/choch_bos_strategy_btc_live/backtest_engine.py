from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import TraderConfig
from .order_utils import bybit_fee_fn, calc_liq_price_long, calc_liq_price_short


@dataclass
class StrategyParams:
    swing_lookback: int
    bos_lookback: int
    fib_low: float
    fib_high: float
    direction: str = "long"


@dataclass
class BacktestMetrics:
    pnl_pct: float
    pnl_value: float
    final_balance: float
    avg_win: float
    avg_loss: float
    win_rate: float
    rr_ratio: float | None
    sharpe: float
    drawdown: float
    wins: int
    losses: int


def summarize_long_results(best_long: pd.DataFrame, starting_balance: float) -> Dict[str, float]:
    l = best_long.iloc[0]
    total_trades = int(l["wins"] + l["losses"])
    total_wins = int(l["wins"])
    total_losses = int(l["losses"])
    total_pnl = float(l["pnl_value"])
    combined_final_balance = float(l["final_balance"])
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    avg_win = float(l["avg_win"])
    avg_loss = float(l["avg_loss"])
    return {
        "Total Trades": total_trades,
        "Wins": total_wins,
        "Losses": total_losses,
        "Win Rate %": round(win_rate, 2),
        "Total PnL": round(total_pnl, 2),
        "Final Balance": round(combined_final_balance, 2),
        "Average Win": round(avg_win, 2),
        "Average Loss": round(avg_loss, 2),
    }


class BacktestEngine:
    def __init__(self, config: TraderConfig):
        self.config = config
        self._last_trades: List[Dict] = []
        self._placeholder_short_row = {
            "swing_lookback": self.config.swing_lookback,
            "bos_lookback": self.config.bos_lookback,
            "fib_low": self.config.fib_low,
            "fib_high": self.config.fib_high,
            "direction": "short",
            "pnl_pct": 0.0,
            "pnl_value": 0.0,
            "final_balance": self.config.starting_balance,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_rate": 0.0,
            "rr_ratio": None,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "wins": 0,
            "losses": 0,
        }

    def _run_backtest(
        self,
        df_1m: pd.DataFrame,
        params: StrategyParams,
        capture_trades: bool = False,
        df_5m: pd.DataFrame | None = None,
    ) -> BacktestMetrics:
        if df_5m is None:
            raise ValueError("5m dataframe is required; do not resample 1m data.")
        data = df_1m.copy().sort_index()
        ht = df_5m.copy().sort_index()

        ht["swing_high"] = ht["High"].rolling(params.swing_lookback).max()
        ht["swing_low"] = ht["Low"].rolling(params.swing_lookback).min()
        ht_sw_high = ht["swing_high"].reindex(data.index, method="ffill")
        ht_sw_low = ht["swing_low"].reindex(data.index, method="ffill")

        range_height = ht_sw_high - ht_sw_low
        fib_upper = ht_sw_low + params.fib_high * range_height
        fib_lower = ht_sw_low + params.fib_low * range_height
        in_fib = (data["Close"] <= fib_upper) & (data["Close"] >= fib_lower)

        # 1m structure: need both ChoCH and BOS; enter on the demand that fueled the ChoCH
        bos_up = data["Close"] > ht_sw_high.shift(1)
        choch_up = (data["High"] > data["High"].shift(params.bos_lookback)) & (
            data["Low"] > data["Low"].shift(params.bos_lookback)
        )
        demand_trigger = data["Low"].rolling(params.bos_lookback).min()
        demand_ok = (demand_trigger >= ht_sw_low) & (demand_trigger >= fib_lower)
        structure_conf = bos_up & choch_up

        data["entry_signal"] = in_fib & structure_conf & demand_ok & (data["Low"] <= fib_upper)
        data["exit_signal"] = (data["Close"] < fib_lower) | (data["Close"] < demand_trigger)
        data["tradable"] = data[["entry_signal", "exit_signal"]].notna().all(axis=1)

        balance = self.config.starting_balance
        equity_curve: List[float] = []
        position = 0
        entry_price = None
        entry_time = None
        liq_price = None
        qty = 0.0
        wins = 0
        losses = 0
        win_sizes: List[float] = []
        loss_sizes: List[float] = []
        in_liquidation = False
        trades: List[Dict] = [] if capture_trades else []

        warmup = max(params.swing_lookback * 5, params.bos_lookback * 2)
        for i in range(warmup, len(data)):
            if balance <= 0:
                equity_curve.append(0)
                continue

            row = data.iloc[i]
            open_, high, low, close = row["Open"], row["High"], row["Low"], row["Close"]

            entry_cond = bool(row["entry_signal"] and row["tradable"] and position == 0)

            if entry_cond and not in_liquidation:
                entry_price = close
                entry_time = row.name
                trade_value = balance * self.config.leverage
                qty = trade_value / entry_price
                entry_fee = bybit_fee_fn(trade_value, self.config)
                balance -= entry_fee
                liq_price = calc_liq_price_long(entry_price, self.config.leverage)
                position = 1

            if position == 1 and liq_price and low <= liq_price:
                if capture_trades and entry_price is not None and entry_time is not None:
                    net_pnl = -self.config.starting_balance
                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": row.name,
                            "side": "LONG",
                            "entry_price": entry_price,
                            "exit_price": liq_price,
                            "pnl_value": net_pnl,
                            "pnl_pct": (net_pnl / self.config.starting_balance) * 100,
                            "qty": qty,
                        }
                    )
                balance = 0
                losses += 1
                loss_sizes.append(-100)
                position = 0
                entry_price = None
                entry_time = None
                qty = 0
                in_liquidation = True
                equity_curve.append(0)
                continue
            if position == -1 and liq_price and high >= liq_price:
                balance = 0
                losses += 1
                loss_sizes.append(-100)
                position = 0
                entry_price = None
                qty = 0
                in_liquidation = True
                equity_curve.append(0)
                continue

            if position == 1 and not in_liquidation and bool(row["exit_signal"]):
                exit_price = close
                exit_fee = bybit_fee_fn(qty * exit_price, self.config)
                gross = (exit_price - entry_price) * qty  # type: ignore[operator]
                entry_fee = bybit_fee_fn(qty * entry_price, self.config)  # type: ignore[arg-type]
                net_pnl = gross - entry_fee - exit_fee
                balance += net_pnl
                if net_pnl > 0:
                    wins += 1
                    win_sizes.append((net_pnl / self.config.starting_balance) * 100)
                else:
                    losses += 1
                    loss_sizes.append((net_pnl / self.config.starting_balance) * 100)
                if capture_trades and entry_price is not None and entry_time is not None:
                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": row.name,
                            "side": "LONG",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl_value": net_pnl,
                            "pnl_pct": (net_pnl / self.config.starting_balance) * 100,
                            "qty": qty,
                        }
                    )
                position = 0
                entry_price = None
                entry_time = None
                qty = 0

            if position == 1 and entry_price is not None:
                unrealized_pnl = (close - entry_price) * qty
                equity = balance + unrealized_pnl
            elif position == -1 and entry_price is not None:
                unrealized_pnl = (entry_price - close) * qty
                equity = balance + unrealized_pnl
            else:
                equity = balance
            equity_curve.append(max(equity, 0))

        if not equity_curve:
            return BacktestMetrics(0, 0, self.config.starting_balance, 0, 0, 0, None, 0, 0, 0, 0)

        final_balance = equity_curve[-1]
        pnl_value = final_balance - self.config.starting_balance
        pnl_pct = (pnl_value / self.config.starting_balance) * 100
        avg_win = float(np.mean(win_sizes)) if win_sizes else 0
        avg_loss = float(np.mean(loss_sizes)) if loss_sizes else 0
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        rr_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else None
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24 * 60 / self.config.agg_minutes) if returns.std() != 0 else 0

        if capture_trades:
            self._last_trades = trades
        else:
            self._last_trades = []

        return BacktestMetrics(pnl_pct, pnl_value, final_balance, avg_win, avg_loss, win_rate, rr_ratio, sharpe, 0, wins, losses)

    def run_backtest_with_trades(
        self, df_1m: pd.DataFrame, params: StrategyParams, df_5m: pd.DataFrame
    ) -> tuple[BacktestMetrics, pd.DataFrame]:
        metrics = self._run_backtest(df_1m, params, capture_trades=True, df_5m=df_5m)
        trades_df = pd.DataFrame(self._last_trades) if hasattr(self, "_last_trades") else pd.DataFrame()
        return metrics, trades_df

    def grid_search(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        results_long: List[Dict] = []
        results_short: List[Dict] = []

        for swing_lb in self.config.swing_lookback_range:
            for bos_lb in self.config.bos_lookback_range:
                for fib_low in self.config.fib_low_range:
                    for fib_high in self.config.fib_high_range:
                        params_long = StrategyParams(int(swing_lb), int(bos_lb), float(fib_low), float(fib_high), "long")
                        metrics_long = self._run_backtest(df_1m, params_long, df_5m=df_5m)
                        results_long.append({**params_long.__dict__, **metrics_long.__dict__})
        if not results_short:
            results_short.append(self._placeholder_short_row.copy())

        df_long = pd.DataFrame(results_long)
        df_short = pd.DataFrame(results_short)
        return df_long, df_short

    def grid_search_with_progress(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        results_long: List[Dict] = []
        results_short: List[Dict] = []
        total = (
            len(self.config.swing_lookback_range)
            * len(self.config.bos_lookback_range)
            * len(list(self.config.fib_low_range))
            * len(list(self.config.fib_high_range))
        )

        for swing_lb, bos_lb, fib_low, fib_high in tqdm(
            [
                (s, b, fl, fh)
                for s in self.config.swing_lookback_range
                for b in self.config.bos_lookback_range
                for fl in self.config.fib_low_range
                for fh in self.config.fib_high_range
            ],
            total=total,
            desc="Param search",
            ncols=80,
        ):
            params_long = StrategyParams(int(swing_lb), int(bos_lb), float(fib_low), float(fib_high), "long")
            metrics_long = self._run_backtest(df_1m, params_long, df_5m=df_5m)
            results_long.append({**params_long.__dict__, **metrics_long.__dict__})
        if not results_short:
            results_short.append(self._placeholder_short_row.copy())

        return pd.DataFrame(results_long), pd.DataFrame(results_short)


def summarize_long_results(best_long: pd.DataFrame, starting_balance: float) -> Dict[str, float]:
    l = best_long.iloc[0]
    total_trades = int(l["wins"] + l["losses"])
    total_wins = int(l["wins"])
    total_losses = int(l["losses"])
    total_pnl = float(l["pnl_value"])
    combined_final_balance = float(l["final_balance"])
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    avg_win = float(l["avg_win"])
    avg_loss = float(l["avg_loss"])
    return {
        "Total Trades": total_trades,
        "Wins": total_wins,
        "Losses": total_losses,
        "Win Rate %": round(win_rate, 2),
        "Total PnL": round(total_pnl, 2),
        "Combined Final Balance": round(combined_final_balance, 2),
        "Average Win": round(avg_win, 2),
        "Average Loss": round(avg_loss, 2),
    }
