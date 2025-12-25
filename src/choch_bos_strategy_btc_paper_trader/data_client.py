from __future__ import annotations

from datetime import datetime
import time
from typing import Optional

import pandas as pd
import requests

from .config import TraderConfig


class DataClient:
    def __init__(self, config: TraderConfig):
        self.config = config

    def fetch_bybit_bars(
        self,
        symbol: Optional[str] = None,
        category: Optional[str] = None,
        days: Optional[int] = None,
        interval_minutes: Optional[int] = None,
    ) -> pd.DataFrame:
        symbol = symbol or self.config.symbol
        category = category or self.config.category
        days = days or self.config.backtest_days
        interval_minutes = interval_minutes or self.config.agg_minutes

        end = int(datetime.utcnow().timestamp())
        start = end - days * 24 * 60 * 60
        df_list = []

        while start < end:
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                "category": category,
                "symbol": symbol,
                "interval": str(interval_minutes),
                "start": start * 1000,
                "limit": 1000,
            }
            resp = requests.get(url, params=params, timeout=10)
            if not resp.ok:
                raise RuntimeError(f"Bybit API request failed with status {resp.status_code}: {resp.text}")

            payload = resp.json()
            if str(payload.get("retCode")) != "0":
                raise RuntimeError(f"Bybit API returned error code {payload.get('retCode')}: {payload.get('retMsg')}")

            rows = payload["result"].get("list", [])
            if not rows:
                break

            df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "turnover"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            df = df.sort_values("timestamp")
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = df[col].astype(float)
            df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
            df.set_index("timestamp", inplace=True)
            df_list.append(df)
            start = int(df.index[-1].timestamp()) + interval_minutes * 60
            time.sleep(0.2)

        if not df_list:
            raise ValueError("No candle data received from Bybit.")

        return pd.concat(df_list).sort_index()

    def fetch_bybit_1m(self, symbol: Optional[str] = None, category: Optional[str] = None, days: Optional[int] = None) -> pd.DataFrame:
        """Backward compatible wrapper for code paths still requesting 1m bars."""
        return self.fetch_bybit_bars(symbol=symbol, category=category, days=days, interval_minutes=1)
