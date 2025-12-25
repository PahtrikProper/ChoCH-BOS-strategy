from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import TraderConfig


def bybit_fee_fn(trade_value: float, config: TraderConfig) -> float:
    return trade_value * config.bybit_fee


def simulate_order_fill(
    direction: str,
    mid_price: float,
    config: TraderConfig,
    spread_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
    reject_prob: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    spread_bps = spread_bps if spread_bps is not None else config.spread_bps
    slippage_bps = slippage_bps if slippage_bps is not None else config.slippage_bps
    reject_prob = reject_prob if reject_prob is not None else config.order_reject_prob

    if random.random() < reject_prob:
        return None, "rejected"

    spread = mid_price * (spread_bps / 10000)
    slippage = abs(np.random.normal(slippage_bps, slippage_bps / 2))
    slippage_amt = mid_price * (slippage / 10000)

    if direction == "long":
        fill_price = mid_price + spread + slippage_amt
    else:
        fill_price = mid_price - spread - slippage_amt

    time.sleep(random.uniform(0, config.max_fill_latency))
    return fill_price, "filled"


def mark_to_market_equity(
    cash_equity: float,
    position: int,
    entry_price: Optional[float],
    qty: float,
    last_price: float,
    margin_used: float = 0.0,
) -> float:
    total = cash_equity + margin_used
    if position == 1 and entry_price:
        total += (last_price - entry_price) * qty
    if position == -1 and entry_price:
        total += (entry_price - last_price) * qty
    return total


def calc_liq_price_long(entry_price: float, leverage: int) -> float:
    return entry_price * (1 - 1 / leverage)


def calc_liq_price_short(entry_price: float, leverage: int) -> float:
    return entry_price * (1 + 1 / leverage)


@dataclass
class PositionState:
    side: Optional[str] = None  # "long" or "short"
    entry_price: Optional[float] = None
    tp_price: Optional[float] = None
    liq_price: Optional[float] = None
    qty: float = 0.0
    entry_bar_time: Optional[pd.Timestamp] = None
    entry_fee: float = 0.0
    trade_value: float = 0.0
    margin_used: float = 0.0
