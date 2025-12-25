#
"""Live Bybit linear futures trader built from the paper-trading prototype.

The script calculates trade signals using native-interval Bybit kline data and
submits real orders (paper trading is blocked). It now applies a Stoch RSI
crossover approach (D% crossing up the lower bound to buy, K% crossing down the
upper bound to sell) and runs as a standalone loop for Bybit *linear futures*
only. It requests 10x leverage on startup (skipping isolated margin changes on
Unified Accounts).

**Safety model**
- Maker-only entries at candle close (PostOnly limit orders)
- Reduce-only exits (PostOnly limit TP, market SL)
- Position truth is pulled from Bybit every loop; no local position assumptions
- Exit size always equals the open size reported by the exchange
- A rejected entry is ignored until the next cycle; exit failures fall back to
  a reduce-only market close

Environment variables
---------------------
BYBIT_API_KEY      API key for the Bybit mainnet account.
BYBIT_API_SECRET   API secret for the Bybit account.
BYBIT_API_URL      Optional. Defaults to ``https://api.bybit.com`` (mainnet only).
BYBIT_POSITION_IDX Optional. Provide 1/2 for hedge mode; omit for one-way.
api_keys.json      Optional. JSON file with the same keys as above; env vars win on conflicts.

**Usage flow**
- Run the paper-trading optimizer via ``python -m v4`` first to derive
  production parameters.
- Apply those tuned ``stoch_params`` below.
- Run this script only on Bybit mainnet; testnet and paper trading are blocked.
"""
from __future__ import annotations

raise SystemExit(
    "Live trading is disabled in choch_bos_strategy_btc_paper_trader. "
    "Use the main engine to run the paper-trading loop instead."
)

import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import math

import numpy as np
import pandas as pd
import requests

# ===== USER CONFIG =====
symbol = "BTCUSDT"
category = "linear"  # linear USDT perpetuals
agg_minutes = 5
backtest_days = 30
leverage = 10
trade_mode = 1  # 0 = cross, 1 = isolated
spread_bps = 2
slippage_bps = 3
order_reject_prob = 0.01  # kept for signal continuity; not used for real orders
max_fill_latency = 0.5

# Strategy parameters (use optimizer output from paper trader as needed)
stoch_params = {
    "rsi_length": 14,
    "stoch_length": 14,
    "smooth_k": 3,
    "smooth_d": 3,
    "lower_bound": 0.2,
    "upper_bound": 0.8,
    "min_up_candles": 4,
    "trend_ema_length": 200,
    "tp_pct": 0.01,
    "sl_pct": 0.005,
}

# Risk sizing
max_usdt_per_trade = 150


class BybitClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = base_url.rstrip("/")

    def _sign(self, payload: str, timestamp: str, recv_window: str) -> str:
        param_str = timestamp + self.api_key + recv_window + payload
        return hmac.new(self.api_secret, param_str.encode(), hashlib.sha256).hexdigest()

    def _request(self, method: str, path: str, params: Optional[Dict] = None, allowed_ret_codes: Optional[set] = None) -> Dict:
        url = f"{self.base_url}{path}"
        params = params or {}
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        method_upper = method.upper()

        if method_upper == "GET":
            # Sort params by key and build query string for signature per Bybit v5 docs
            query_pairs = [f"{k}={v}" for k, v in sorted(params.items())]
            query_str = "&".join(query_pairs)
            body = query_str
        else:
            body = json.dumps(params)

        sign = self._sign(body, timestamp, recv_window)

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": sign,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }

        resp = requests.request(
            method_upper,
            url,
            params=params if method_upper == "GET" else None,
            data=body if method_upper == "POST" else None,
            headers=headers,
            timeout=10,
        )
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        payload = resp.json()
        ret_code = str(payload.get("retCode"))
        allowed = allowed_ret_codes or set()
        if ret_code != "0" and ret_code not in allowed:
            raise RuntimeError(f"Bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
        return payload.get("result", {})

    def get_tickers(self, category: str, symbol: str) -> Dict:
        return self._request("GET", "/v5/market/tickers", {"category": category, "symbol": symbol})

    def set_leverage(
        self,
        category: str,
        symbol: str,
        buy_leverage: int,
        sell_leverage: int,
        trade_mode: int | None = None,
    ) -> None:
        payload = {
            "category": category,
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage),
        }
        if trade_mode is not None:
            payload["tradeMode"] = trade_mode
        # retCode 110043 indicates leverage unchanged; treat as success
        self._request("POST", "/v5/position/set-leverage", payload, allowed_ret_codes={"110043"})

    def get_position(self, category: str, symbol: str, position_idx: Optional[int]) -> Optional[Dict]:
        params = {"category": category, "symbol": symbol}
        if position_idx:
            params["positionIdx"] = position_idx
        result = self._request("GET", "/v5/position/list", params)
        for item in result.get("list", []):
            if item.get("symbol") != symbol:
                continue
            size = float(item.get("size", 0) or 0)
            if size == 0:
                continue
            return {
                "side": item.get("side"),
                "size": size,
                "avg_price": float(item.get("avgPrice", 0) or 0),
                "position_idx": item.get("positionIdx"),
            }
        return None

    def get_open_orders(self, category: str, symbol: str) -> list[Dict]:
        result = self._request("GET", "/v5/order/realtime", {"category": category, "symbol": symbol})
        return result.get("list", [])

    def get_order(self, category: str, symbol: str, order_id: str) -> Optional[Dict]:
        result = self._request(
            "GET", "/v5/order/realtime", {"category": category, "symbol": symbol, "orderId": order_id}
        )
        orders = result.get("list", [])
        if not orders:
            return None
        return orders[0]

    def cancel_order(self, category: str, symbol: str, order_id: str) -> None:
        payload = {"category": category, "symbol": symbol, "orderId": order_id}
        self._request("POST", "/v5/order/cancel", payload)

    def cancel_reduce_only_orders(self, category: str, symbol: str) -> None:
        orders = self.get_open_orders(category, symbol)
        for order in orders:
            if str(order.get("reduceOnly", "false")).lower() == "true":
                try:
                    self.cancel_order(category, symbol, order.get("orderId"))
                except Exception as exc:  # noqa: PIE786
                    print(f"Failed to cancel reduce-only order {order.get('orderId')}: {exc}")

    def place_limit_post_only(
        self,
        category: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        position_idx: Optional[int],
    ) -> str:
        payload: Dict[str, str | float | int] = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "timeInForce": "PostOnly",
            "reduceOnly": False,
            "closeOnTrigger": False,
        }
        if position_idx:
            payload["positionIdx"] = position_idx
        result = self._request("POST", "/v5/order/create", payload)
        return result.get("orderId", "")

    def place_reduce_only_limit(
        self,
        category: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        position_idx: Optional[int],
    ) -> str:
        payload: Dict[str, str | float | int] = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "timeInForce": "PostOnly",
            "reduceOnly": True,
            "closeOnTrigger": True,
        }
        if position_idx:
            payload["positionIdx"] = position_idx
        result = self._request("POST", "/v5/order/create", payload)
        return result.get("orderId", "")

    def place_reduce_only_stop_market(
        self,
        category: str,
        symbol: str,
        side: str,
        qty: float,
        trigger_price: float,
        position_idx: Optional[int],
    ) -> str:
        payload: Dict[str, str | float | int] = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "triggerPrice": str(trigger_price),
            "triggerBy": "MarkPrice",
            "triggerDirection": trigger_direction_for_stop(side),
            "orderFilter": "StopOrder",
            "timeInForce": "ImmediateOrCancel",
            "reduceOnly": True,
            "closeOnTrigger": True,
        }
        if position_idx:
            payload["positionIdx"] = position_idx
        result = self._request("POST", "/v5/order/create", payload)
        return result.get("orderId", "")

    def place_reduce_only_market_exit(
        self,
        category: str,
        symbol: str,
        side: str,
        qty: float,
        position_idx: Optional[int],
    ) -> str:
        payload: Dict[str, str | float | int] = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "ImmediateOrCancel",
            "reduceOnly": True,
        }
        if position_idx:
            payload["positionIdx"] = position_idx
        result = self._request("POST", "/v5/order/create", payload)
        return result.get("orderId", "")

    def get_wallet_balance(self, coin: str = "USDT") -> float:
        result = self._request(
            "GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": coin}
        )
        list_data = result.get("list", [])
        if not list_data:
            return 0.0
        available = list_data[0].get("totalAvailableBalance", 0)
        return float(available)


def fetch_bybit_bars(
    symbol: str,
    category: str,
    interval_minutes: int = agg_minutes,
    limit: int = 1000,
    days: int = backtest_days,
) -> pd.DataFrame:
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
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if str(payload.get("retCode")) != "0":
            raise RuntimeError(f"Bybit API returned error code {payload.get('retCode')}: {payload.get('retMsg')}")

        rows = payload.get("result", {}).get("list", [])
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
        raise ValueError("No candle data fetched from Bybit.")
    return pd.concat(df_list).sort_index()


def calculate_signal(df: pd.DataFrame) -> Dict[str, float | bool]:
    data = df.copy()
    rsi_len = int(stoch_params["rsi_length"])
    stoch_len = int(stoch_params["stoch_length"])
    smooth_k = int(stoch_params["smooth_k"])
    smooth_d = int(stoch_params["smooth_d"])
    lower = float(stoch_params["lower_bound"])
    upper = float(stoch_params["upper_bound"])
    min_up = int(stoch_params["min_up_candles"])
    trend_ema_length = int(stoch_params.get("trend_ema_length", 200))

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))

    rsi_min = data["rsi"].rolling(stoch_len).min()
    rsi_max = data["rsi"].rolling(stoch_len).max()
    stoch_rsi_raw = (data["rsi"] - rsi_min) / (rsi_max - rsi_min)
    stoch_rsi_raw = stoch_rsi_raw.replace([np.inf, -np.inf], np.nan).clip(0, 1)
    data["stoch_k"] = stoch_rsi_raw.rolling(smooth_k).mean()
    data["stoch_d"] = data["stoch_k"].rolling(smooth_d).mean()
    data["d_cross_up_lower"] = (data["stoch_d"].shift(1) <= lower) & (data["stoch_d"] > lower)
    data["k_cross_down_upper"] = (data["stoch_k"].shift(1) >= upper) & (data["stoch_k"] < upper)
    data["uptrend"] = (data["Close"].diff() > 0).rolling(min_up).sum() >= min_up
    data["trend_ema"] = data["Close"].ewm(span=trend_ema_length).mean()
    data["above_trend"] = data["Close"] > data["trend_ema"]

    row = data.iloc[-1]
    entry_cond_long = bool(row["d_cross_up_lower"] and row["uptrend"] and row["above_trend"])
    entry_cond_short = bool(row["k_cross_down_upper"])
    return {
        "long": entry_cond_long,
        "short": entry_cond_short,
        "last_close": float(row["Close"]),
        "stoch_k": float(row["stoch_k"]),
        "stoch_d": float(row["stoch_d"]),
        "d_cross_up_lower": bool(row["d_cross_up_lower"]),
        "k_cross_down_upper": bool(row["k_cross_down_upper"]),
        "uptrend": bool(row["uptrend"]),
        "above_trend": bool(row["above_trend"]),
    }


def indicator_position(signal: Dict[str, float | bool]) -> str:
    return "LONG" if signal["long"] else "FLAT"


def determine_qty(client: BybitClient, last_price: float) -> float:
    balance = client.get_wallet_balance()
    if balance <= 0:
        raise RuntimeError("No available USDT balance to trade.")
    allocation = min(balance, max_usdt_per_trade)
    qty = allocation * leverage / last_price
    return round(qty, 6)


def compute_take_profit_stop_loss(side: str, entry_price: float) -> tuple[float, float]:
    if side == "Buy":
        tp_price = entry_price * (1 + float(stoch_params["tp_pct"]))
        sl_price = entry_price * (1 - float(stoch_params["sl_pct"]))
    else:
        tp_price = entry_price * (1 - float(stoch_params["tp_pct"]))
        sl_price = entry_price * (1 + float(stoch_params["sl_pct"]))
    return round(tp_price, 4), round(sl_price, 4)


def minimum_history_days_for_indicators(agg_minutes: int) -> int:
    bars_needed = (
        stoch_params["rsi_length"]
        + stoch_params["stoch_length"]
        + stoch_params["smooth_k"]
        + stoch_params["smooth_d"]
        + stoch_params["min_up_candles"]
        + 5
    )
    minutes_needed = bars_needed * agg_minutes
    return max(1, math.ceil(minutes_needed / (60 * 24)))


def trigger_direction_for_stop(side: str) -> int:
    """Return Bybit triggerDirection per side for stop orders.

    triggerDirection=1 fires when lastPrice/markPrice rises to or above triggerPrice,
    triggerDirection=2 fires when it falls to or below triggerPrice.
    """
    return 2 if side.lower() == "sell" else 1


def get_current_price(client: BybitClient, category: str, symbol: str, fallback: float) -> float:
    try:
        ticker = client.get_tickers(category, symbol)
        last_price = float(ticker.get("list", [{}])[0].get("lastPrice", fallback))
        return last_price
    except Exception as exc:  # noqa: PIE786
        print(f"Failed to fetch live price; falling back to last close: {exc}")
        return fallback


def poll_for_entry_fill(
    client: BybitClient,
    pending_entry: Dict[str, str | float],
    category: str,
    symbol: str,
    poll_interval: float = 1.5,
    timeout_seconds: float = 12,
) -> Optional[Dict[str, float]]:
    """Poll Bybit for entry fill status and return fill info when available.

    Returns a dict with ``filled_qty`` and ``avg_price`` on successful fill,
    otherwise ``None``. Cancels the pending order on timeout.
    """

    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            order = client.get_order(category, symbol, pending_entry["order_id"])
        except Exception as exc:  # noqa: PIE786
            print(f"Failed to poll order status: {exc}")
            return None

        if order:
            status = order.get("orderStatus")
            if status in {"Filled", "PartiallyFilled"}:
                filled_qty = float(order.get("cumExecQty", 0) or 0)
                avg_price = float(order.get("avgPrice", pending_entry["price"]) or pending_entry["price"])
                if filled_qty > 0:
                    if status == "PartiallyFilled":
                        try:
                            client.cancel_order(category, symbol, pending_entry["order_id"])
                            print("Cancelled remaining entry after partial fill to avoid extra exposure.")
                        except Exception as exc:  # noqa: PIE786
                            print(f"Failed to cancel remaining entry after partial fill: {exc}")
                    return {"filled_qty": filled_qty, "avg_price": avg_price}
            elif status in {"Cancelled", "Rejected", "Expired"}:
                print(f"Entry order ended with status {status}; skipping exits.")
                return None

        time.sleep(poll_interval)

    print("Entry fill polling timed out; cancelling pending order.")
    try:
        client.cancel_order(category, symbol, pending_entry["order_id"])
    except Exception as exc:  # noqa: PIE786
        print(f"Failed to cancel timed-out entry: {exc}")
    return None


def place_exits_for_fill(
    client: BybitClient,
    side: str,
    filled_qty: float,
    avg_price: float,
    position_idx: Optional[int],
    category: str,
    symbol: str,
) -> None:
    exit_side = "Sell" if side == "Buy" else "Buy"
    tp_price, sl_price = compute_take_profit_stop_loss(side, avg_price)

    try:
        client.place_reduce_only_limit(category, symbol, exit_side, filled_qty, tp_price, position_idx)
        print(f"Placed reduce-only TP {exit_side} @ {tp_price} for filled qty {filled_qty}.")
    except Exception as exc:  # noqa: PIE786
        print(f"Failed to place TP limit after fill: {exc}. Trying reduce-only market exit.")
        try:
            client.place_reduce_only_market_exit(category, symbol, exit_side, filled_qty, position_idx)
            print("Issued reduce-only market exit after TP placement error.")
        except Exception as exc2:  # noqa: PIE786
            print(f"Market exit failed after TP placement error: {exc2}")

    try:
        client.place_reduce_only_stop_market(category, symbol, exit_side, filled_qty, sl_price, position_idx)
        print(f"Placed reduce-only SL {exit_side} trigger @ {sl_price} (Mark) for filled qty {filled_qty}.")
    except Exception as exc:  # noqa: PIE786
        print(f"Failed to place SL stop after fill: {exc}. Trying reduce-only market exit.")
        try:
            client.place_reduce_only_market_exit(category, symbol, exit_side, filled_qty, position_idx)
            print("Issued reduce-only market exit after SL placement error.")
        except Exception as exc2:  # noqa: PIE786
            print(f"Market exit failed after SL placement error: {exc2}")


def ensure_exit_orders(
    client: BybitClient,
    position: Dict,
    tp_price: float,
    sl_price: float,
    category: str,
    symbol: str,
) -> None:
    side = position["side"]
    exit_side = "Sell" if side == "Buy" else "Buy"
    qty = position["size"]
    position_idx = position.get("position_idx")
    open_orders = client.get_open_orders(category, symbol)

    existing_tp = next(
        (
            o
            for o in open_orders
            if str(o.get("reduceOnly", "false")).lower() == "true"
            and o.get("orderType") == "Limit"
            and o.get("side") == exit_side
        ),
        None,
    )
    if existing_tp is None:
        try:
            client.place_reduce_only_limit(category, symbol, exit_side, qty, tp_price, position_idx)
            print(f"Placed reduce-only TP {exit_side} @ {tp_price} for qty {qty}.")
        except Exception as exc:  # noqa: PIE786
            print(f"Failed to place TP limit: {exc}. Trying immediate reduce-only market exit.")
            try:
                client.place_reduce_only_market_exit(category, symbol, exit_side, qty, position_idx)
                print("Issued reduce-only market exit after TP failure.")
            except Exception as exc2:  # noqa: PIE786
                print(f"Market exit failed after TP placement error: {exc2}")

    existing_sl = next(
        (
            o
            for o in open_orders
            if str(o.get("reduceOnly", "false")).lower() == "true"
            and o.get("orderType") == "Market"
            and o.get("side") == exit_side
            and o.get("triggerPrice") is not None
        ),
        None,
    )
    if existing_sl is None:
        try:
            client.place_reduce_only_stop_market(category, symbol, exit_side, qty, sl_price, position_idx)
            print(f"Placed reduce-only SL {exit_side} trigger @ {sl_price} (Mark) for qty {qty}.")
        except Exception as exc:  # noqa: PIE786
            print(f"Failed to place SL stop: {exc}. Trying immediate reduce-only market exit.")
            try:
                client.place_reduce_only_market_exit(category, symbol, exit_side, qty, position_idx)
                print("Issued reduce-only market exit after SL placement error.")
            except Exception as exc2:  # noqa: PIE786
                print(f"Market exit failed after SL placement error: {exc2}")


def ensure_mainnet_only(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    mainnet = "https://api.bybit.com"
    if "testnet" in normalized.lower() or normalized != mainnet:
        raise SystemExit(
            "BYBIT_API_URL must point to Bybit mainnet (https://api.bybit.com); testnet and alt endpoints are blocked."
        )
    return normalized


def load_api_credentials(path: str = "api_keys.json") -> Dict[str, str | int | None]:
    defaults = {
        "BYBIT_API_KEY": None,
        "BYBIT_API_SECRET": None,
        "BYBIT_API_URL": "https://api.bybit.com",
        "BYBIT_POSITION_IDX": "0",
    }

    file_creds: Dict[str, str | int | None] = {}
    file_path = Path(path)
    if file_path.exists():
        try:
            file_creds = json.loads(file_path.read_text())
            print(f"Loaded API credentials from {file_path}")
        except Exception as exc:  # noqa: PIE786
            print(f"Failed to load {file_path}: {exc}. Continuing with environment values.")

    creds: Dict[str, str | int | None] = {}
    for key, default in defaults.items():
        env_value = os.getenv(key)
        creds[key] = env_value if env_value is not None else file_creds.get(key, default)

    return creds


# Paper trading is explicitly disabled. The trader always runs live on Bybit
# mainnet; setting DRY_RUN=true or pointing to a testnet URL will abort.
if os.getenv("DRY_RUN", "false").lower() == "true":
    raise SystemExit("Paper trading (DRY_RUN) is disabled; this trader always runs live.")


def run_live_trading() -> None:
    creds = load_api_credentials()
    api_key = creds["BYBIT_API_KEY"]
    api_secret = creds["BYBIT_API_SECRET"]
    base_url = ensure_mainnet_only(str(creds["BYBIT_API_URL"]))
    position_idx = int(creds["BYBIT_POSITION_IDX"] or 0) or None

    if not api_key or not api_secret:
        raise SystemExit("BYBIT_API_KEY and BYBIT_API_SECRET must be set (env or api_keys.json).")

    client = BybitClient(str(api_key), str(api_secret), base_url)
    print(f"Starting live trader for {symbol} ({category}).")

    try:
        client.set_leverage(category, symbol, leverage, leverage, trade_mode=trade_mode)
        print(f"Configured {leverage}x leverage for {symbol} with trade mode={trade_mode} (1=isolated).")
    except Exception as exc:  # noqa: PIE786
        print(f"Leverage setup skipped due to error: {exc}")

    pending_entry: Optional[Dict[str, str | float]] = None
    long_cooldown_until: Optional[pd.Timestamp] = None
    last_long_signal_ts: Optional[pd.Timestamp] = None
    history_days = max(backtest_days, minimum_history_days_for_indicators(agg_minutes))

    while True:
        try:
            try:
                position = client.get_position(category, symbol, position_idx)
            except Exception as exc:  # noqa: PIE786
                print(f"Failed to fetch position; skipping cycle: {exc}")
                time.sleep(agg_minutes * 60)
                continue

            if pending_entry is not None:
                if position is not None:
                    print(
                        "Open position detected while waiting on a pending entry; "
                        "clearing pending entry to manage the live position."
                    )
                    pending_entry = None
                    last_long_signal_ts = None
                else:
                    print(
                        f"Polling for entry fill: id={pending_entry['order_id']} side={pending_entry['side']} "
                        f"price={pending_entry['price']}"
                    )
                    fill_info = poll_for_entry_fill(client, pending_entry, category, symbol)
                    if fill_info:
                        place_exits_for_fill(
                            client,
                            str(pending_entry["side"]),
                            float(fill_info["filled_qty"]),
                            float(fill_info["avg_price"]),
                            position_idx,
                            category,
                            symbol,
                        )
                    else:
                        print("Entry did not fill within limits; no exits will be placed.")
                    pending_entry = None
                    time.sleep(agg_minutes * 60)
                    continue

            try:
                df = fetch_bybit_bars(symbol, category, interval_minutes=agg_minutes, days=history_days)
            except Exception as exc:  # noqa: PIE786
                print(f"Data fetch failed: {exc}")
                time.sleep(agg_minutes * 60)
                continue

            bars_needed = (
                stoch_params["rsi_length"]
                + stoch_params["stoch_length"]
                + stoch_params["smooth_k"]
                + stoch_params["smooth_d"]
                + stoch_params["min_up_candles"]
                + 5
            )
            if len(df) < bars_needed:
                print(f"Waiting for enough bars ({len(df)}/{bars_needed})...")
                time.sleep(60)
                continue

            signal = calculate_signal(df)
            last_close = signal["last_close"]
            current_price = get_current_price(client, category, symbol, last_close)
            last_candle_ts = df.index[-1].strftime("%Y-%m-%d %H:%M")
            nowstr = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            indicator_state = indicator_position(signal)

            if pending_entry is not None:
                heartbeat_state = "ENTRY_PENDING"
            elif position is None:
                if signal["long"]:
                    heartbeat_state = "ENTRY_LONG_READY"
                else:
                    heartbeat_state = "HOLD_WAIT"
            else:
                heartbeat_state = f"HOLD_{position['side'].upper()}"

            print(
                "Heartbeat | "
                f"{symbol} {agg_minutes}m @ {last_candle_ts} UTC "
                f"Close={last_close:.4f} | LivePx={current_price:.4f} | "
                f"StochK/D={signal['stoch_k']:.4f}/{signal['stoch_d']:.4f} | "
                f"Crosses D↑{signal['d_cross_up_lower']} K↓{signal['k_cross_down_upper']} | "
                f"Indicator={indicator_state} | "
                f"State={heartbeat_state}"
            )

            if position is None:
                try:
                    client.cancel_reduce_only_orders(category, symbol)
                except Exception as exc:  # noqa: PIE786
                    print(f"Failed to cancel stale reduce-only orders: {exc}")

                if signal["long"]:
                    last_candle_ts = df.index[-1]
                    if last_long_signal_ts == last_candle_ts:
                        print("Long signal already acted on this candle; waiting for next cross.")
                        time.sleep(agg_minutes * 60)
                        continue
                    if long_cooldown_until and last_candle_ts < long_cooldown_until:
                        print(f"Long signal skipped due to cooldown until {long_cooldown_until}. Last candle {last_candle_ts}.")
                        time.sleep(agg_minutes * 60)
                        continue

                    side = "Buy"
                    entry_price = round(last_close, 4)
                    try:
                        qty = determine_qty(client, last_close)
                    except Exception as exc:  # noqa: PIE786
                        print(f"Skipping entry; sizing failed: {exc}")
                        time.sleep(agg_minutes * 60)
                        continue

                    long_cooldown_until = last_candle_ts + pd.Timedelta(minutes=agg_minutes * 4)
                    last_long_signal_ts = last_candle_ts

                    try:
                        order_id = client.place_limit_post_only(
                            category, symbol, side, qty, entry_price, position_idx
                        )
                        pending_entry = {"order_id": order_id, "side": side, "price": entry_price}
                        print(
                            f"Placed PostOnly entry {side} {qty} {symbol} @ {entry_price} on {nowstr}."
                            " Waiting for fill before placing exits."
                        )
                    except Exception as exc:  # noqa: PIE786
                        print(f"Entry rejected or failed ({side}): {exc}. Waiting for next cycle.")

            else:
                tp_price, sl_price = compute_take_profit_stop_loss(position["side"], position["avg_price"])
                print(
                    f"{nowstr} | Open {position['side']} size={position['size']} avg={position['avg_price']} "
                    f"TP={tp_price} SL={sl_price} (Mark)"
                )
                ensure_exit_orders(client, position, tp_price, sl_price, category, symbol)

            time.sleep(agg_minutes * 60)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as exc:  # noqa: PIE786 (handling loop exceptions)
            print(f"Exception: {exc}")
            time.sleep(5)


if __name__ == "__main__":
    run_live_trading()
