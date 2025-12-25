from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from .config import TraderConfig
from .order_utils import PositionState, bybit_fee_fn, calc_liq_price_long, simulate_order_fill


@dataclass
class BuyOrderEngine:
    config: TraderConfig

    def should_enter(self, row: pd.Series, position_side: Optional[str]) -> bool:
        if position_side:
            return False
        return bool(row.get("entry_signal") and row.get("tradable", True))

    def open_position(
        self,
        row: pd.Series,
        available_usdt: float,
    ) -> Tuple[Optional[PositionState], str, float, float, float]:
        entry_price, status = simulate_order_fill("long", row["Close"], self.config)
        if status == "rejected" or entry_price is None:
            return None, status, 0.0, 0.0, 0.0

        if available_usdt <= 0:
            return None, "insufficient_funds", 0.0, 0.0, 0.0

        risk_fraction = min(max(self.config.risk_fraction, 0.0), 1.0)
        if risk_fraction == 0:
            return None, "insufficient_funds", 0.0, 0.0, 0.0

        margin_used = available_usdt * risk_fraction
        trade_value = margin_used * self.config.leverage
        qty = trade_value / entry_price
        entry_fee = bybit_fee_fn(trade_value, self.config)

        liq_price = calc_liq_price_long(entry_price, self.config.leverage)

        position = PositionState(
            side="long",
            entry_price=entry_price,
            liq_price=liq_price,
            qty=qty,
            entry_bar_time=row.name,
            entry_fee=entry_fee,
            trade_value=trade_value,
            margin_used=margin_used,
        )
        return position, status, entry_fee, trade_value, margin_used
