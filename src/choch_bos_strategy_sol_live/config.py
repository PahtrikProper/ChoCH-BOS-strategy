from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np


DEFAULT_AGG_MINUTES = 1


@dataclass
class TraderConfig:
    symbol: str = "SOLUSDT"
    category: str = "spot"
    backtest_days: int = 30
    starting_balance: float = 472
    bybit_fee: float = 0.001
    leverage: int = 10
    agg_minutes: int = DEFAULT_AGG_MINUTES
    spread_bps: int = 2  # simulated spread in basis points (0.02%)
    slippage_bps: int = 3  # additional slippage beyond spread (avg in bps)
    order_reject_prob: float = 0.01  # probability an order is rejected (simulated failure)
    max_fill_latency: float = 0.5  # seconds
    risk_fraction: float = 0.95  # portion of available USDT to deploy per entry

    swing_lookback: int = 20  # count of 15m bars to define swing high/low
    bos_lookback: int = 5  # 1m bars to confirm BOS/ChoCH
    fib_low: float = 0.6
    fib_high: float = 0.7

    swing_lookback_range: Sequence[int] = field(default_factory=lambda: (10, 15, 20, 25))
    bos_lookback_range: Sequence[int] = field(default_factory=lambda: (2, 3, 5, 8))
    fib_low_range: Iterable[float] = field(default_factory=lambda: (0.5, 0.55, 0.6, 0.65))
    fib_high_range: Iterable[float] = field(default_factory=lambda: (0.65, 0.7, 0.75, 0.8))

    # Live loop options
    live_history_days: int = 3
    min_history_padding: int = 10

    def as_log_string(self) -> str:
        return (
            f"Symbol: {self.symbol} | Category: {self.category}\n"
            f"Backtest window (days): {self.backtest_days} | Aggregation: {self.agg_minutes}m\n"
            f"Leverage: {self.leverage}x | Fees: {self.bybit_fee * 100:.2f}% per trade\n"
            f"Spread model: {self.spread_bps} bps | Slippage model: ~{self.slippage_bps} bps\n"
            f"Order reject probability: {self.order_reject_prob * 100:.2f}%\n"
            f"Max simulated latency: {self.max_fill_latency}s\n"
            f"Risk per entry: {self.risk_fraction * 100:.1f}% of available USDT\n"
            "Strategy: 15m swing high/low → fib 60-70% pullback into 1m demand; require 1m ChoCH + BOS before entering on the demand that triggered the ChoCH"
        )
