from __future__ import annotations

import asyncio
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Iterable

import httpx
from mcp.server import MCPServer
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

BASE_URL = "https://api.upbit.com"
TIMEOUT = httpx.Timeout(15.0, connect=8.0)
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "UpbitPublicMCP/3.0",
}

mcp = MCPServer("Upbit Public Market Data")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ms_age_seconds(ts_ms: int | float | None) -> float | None:
    if not ts_ms:
        return None
    return max(0.0, time.time() - (float(ts_ms) / 1000.0))


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return (sum(vals) / len(vals)) if vals else None


def _ema(values: list[float], period: int) -> list[float | None]:
    if len(values) < period:
        return [None] * len(values)
    out: list[float | None] = [None] * len(values)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    alpha = 2.0 / (period + 1.0)
    prev = seed
    for i in range(period, len(values)):
        prev = (values[i] - prev) * alpha + prev
        out[i] = prev
    return out


def _sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period <= 0:
        return out
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def _rsi(values: list[float], period: int = 14) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if n <= period:
        return out
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, period + 1):
        d = values[i] - values[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        d = values[i] - values[i - 1]
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def _macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, list[float | None]]:
    ef = _ema(values, fast)
    es = _ema(values, slow)
    macd_line: list[float | None] = [None] * len(values)
    compact: list[float] = []
    compact_idx: list[int] = []
    for i, (a, b) in enumerate(zip(ef, es)):
        if a is not None and b is not None:
            m = a - b
            macd_line[i] = m
            compact.append(m)
            compact_idx.append(i)
    compact_signal = _ema(compact, signal)
    signal_line: list[float | None] = [None] * len(values)
    hist: list[float | None] = [None] * len(values)
    for j, idx in enumerate(compact_idx):
        s = compact_signal[j]
        if s is not None:
            signal_line[idx] = s
            hist[idx] = (macd_line[idx] or 0.0) - s
    return {"macd": macd_line, "signal": signal_line, "histogram": hist}


def _bollinger(values: list[float], period: int = 20, stdev: float = 2.0) -> dict[str, list[float | None]]:
    mid = _sma(values, period)
    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mu = sum(window) / period
        var = sum((x - mu) ** 2 for x in window) / period
        sd = math.sqrt(var)
        upper[i] = mu + stdev * sd
        lower[i] = mu - stdev * sd
    return {"middle": mid, "upper": upper, "lower": lower}


def _atr(candles_asc: list[dict[str, Any]], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(candles_asc)
    if len(candles_asc) <= period:
        return out
    trs: list[float] = []
    for i, c in enumerate(candles_asc):
        high = float(c["high_price"])
        low = float(c["low_price"])
        if i == 0:
            tr = high - low
        else:
            prev_close = float(candles_asc[i - 1]["trade_price"])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    first = sum(trs[1 : period + 1]) / period
    out[period] = first
    prev = first
    for i in range(period + 1, len(trs)):
        prev = ((prev * (period - 1)) + trs[i]) / period
        out[i] = prev
    return out


# One connection pool and shared per-group pacing across all tools.
_http_client: httpx.AsyncClient | None = None
_rate_locks: dict[str, asyncio.Lock] = {}
_rate_next: dict[str, float] = {}
_candle_cache: dict[tuple[str, str, int], tuple[float, list[dict[str, Any]]]] = {}
_candle_locks: dict[tuple[str, str, int], asyncio.Lock] = {}


async def _pace(group: str) -> None:
    async with _rate_locks.setdefault(group, asyncio.Lock()):
        await asyncio.sleep(max(0.0, _rate_next.get(group, 0.0) - time.monotonic()))
        _rate_next[group] = time.monotonic() + 0.115


async def _get(path: str, params: dict[str, Any] | None = None, retries: int = 3) -> tuple[Any, dict[str, str]]:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS)
    group = "candle" if path.startswith("/v1/candles/") else path.split("/")[2]
    for attempt in range(retries + 1):
        await _pace(group)
        resp = await _http_client.get(f"{BASE_URL}{path}", params=params)
        if resp.status_code in {429, 418} and attempt < retries:
            delay = max(float(resp.headers.get("Retry-After", "1")), 0.5 * (attempt + 1))
            _rate_next[group] = max(_rate_next.get(group, 0.0), time.monotonic() + delay)
            await asyncio.sleep(delay)
            continue
        resp.raise_for_status()
        return resp.json(), dict(resp.headers)
    raise RuntimeError("unreachable")


async def _cached_candles(path: str, market: str, count: int) -> tuple[Any, dict[str, str]]:
    key = (path, market, count)
    async with _candle_locks.setdefault(key, asyncio.Lock()):
        cached = _candle_cache.get(key)
        unit = 86400 if path.endswith("days") else int(path.rsplit("/", 1)[1]) * 60
        # Daily history is shared by pattern scanning and full analysis. Never
        # reuse across the 00:00 UTC (09:00 KST) daily boundary.
        if (path.endswith("days") and cached and time.monotonic() - cached[0] < 60
                and cached[1] and cached[1][0]["candle_date_time_utc"][:10] == _utc_now_iso()[:10]):
            return cached[1], {"x-candle-cache": "daily-60s"}
        # Reload both current and preceding bars; larger gaps require full history.
        incremental = cached is not None and time.monotonic() - cached[0] < unit
        raw, headers = await _get(path, {"market": market, "count": 2 if incremental else count})
        if incremental:
            merged = {c["candle_date_time_utc"]: c for c in cached[1]}
            merged.update({c["candle_date_time_utc"]: c for c in raw})
            raw = sorted(merged.values(), key=lambda c: c["candle_date_time_utc"], reverse=True)[:count]
        _candle_cache[key] = (time.monotonic(), raw)
        return raw, headers


def _wrap(data: Any, headers: dict[str, str], endpoint: str) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "received_at_utc": _utc_now_iso(),
        "remaining_req": headers.get("remaining-req"),
        "data": data,
    }


@mcp.tool()
async def health_check() -> dict[str, Any]:
    """Check live connectivity to Upbit Public REST using KRW-BTC ticker."""
    data, headers = await _get("/v1/ticker", {"markets": "KRW-BTC"})
    row = data[0] if data else {}
    return {
        "ok": bool(data),
        "received_at_utc": _utc_now_iso(),
        "source_timestamp_ms": row.get("timestamp"),
        "source_age_seconds": _ms_age_seconds(row.get("timestamp")),
        "remaining_req": headers.get("remaining-req"),
        "sample": row,
    }


@mcp.tool()
async def get_markets(is_details: bool = True) -> dict[str, Any]:
    """List all Upbit trading pairs. Public endpoint; no API key required."""
    data, headers = await _get("/v1/market/all", {"is_details": str(is_details).lower()})
    return _wrap(data, headers, "/v1/market/all")


@mcp.tool()
async def get_all_tickers(quote_currencies: str = "KRW") -> dict[str, Any]:
    """Get current ticker data for every pair in one or more quote markets, e.g. KRW or KRW,BTC,USDT."""
    data, headers = await _get("/v1/ticker/all", {"quote_currencies": quote_currencies})
    ages = [_ms_age_seconds(x.get("timestamp")) for x in data if isinstance(x, dict)]
    ages = [x for x in ages if x is not None]
    return {
        **_wrap(data, headers, "/v1/ticker/all"),
        "rows": len(data),
        "max_source_age_seconds": max(ages) if ages else None,
        "median_source_age_seconds": sorted(ages)[len(ages) // 2] if ages else None,
    }


@mcp.tool()
async def get_ticker(markets: list[str]) -> dict[str, Any]:
    """Get live ticker rows for explicit market codes, e.g. ['KRW-BTC','KRW-ETH']."""
    if not markets:
        raise ValueError("markets must not be empty")
    data, headers = await _get("/v1/ticker", {"markets": ",".join(markets)})
    for row in data:
        row["source_age_seconds"] = _ms_age_seconds(row.get("timestamp"))
    return _wrap(data, headers, "/v1/ticker")


@mcp.tool()
async def get_orderbook(markets: list[str], level: float | None = None) -> dict[str, Any]:
    """Get live order books. Returns Upbit orderbook units plus source-age diagnostics."""
    if not markets:
        raise ValueError("markets must not be empty")
    params: dict[str, Any] = {"markets": ",".join(markets)}
    if level is not None:
        params["level"] = level
    data, headers = await _get("/v1/orderbook", params)
    for row in data:
        row["source_age_seconds"] = _ms_age_seconds(row.get("timestamp"))
        units = row.get("orderbook_units") or []
        if units:
            bid = units[0].get("bid_price")
            ask = units[0].get("ask_price")
            if bid and ask:
                row["best_bid"] = bid
                row["best_ask"] = ask
                row["spread"] = float(ask) - float(bid)
                row["spread_bps"] = (float(ask) - float(bid)) / ((float(ask) + float(bid)) / 2.0) * 10000.0
            bid_size = sum(float(u.get("bid_size", 0) or 0) for u in units[:10])
            ask_size = sum(float(u.get("ask_size", 0) or 0) for u in units[:10])
            denom = bid_size + ask_size
            row["top10_bid_size"] = bid_size
            row["top10_ask_size"] = ask_size
            row["top10_imbalance"] = ((bid_size - ask_size) / denom) if denom else 0.0
    return _wrap(data, headers, "/v1/orderbook")


@mcp.tool()
async def get_candles(
    market: str,
    timeframe: str = "5m",
    count: int = 200,
    to: str | None = None,
) -> dict[str, Any]:
    """Get candles. timeframe: 1s,1m,3m,5m,10m,15m,30m,60m,240m,1d,1w,1M,1y."""
    count = max(1, min(int(count), 200))
    tf = timeframe.strip()
    mapping = {
        "1s": "/v1/candles/seconds",
        "1m": "/v1/candles/minutes/1",
        "3m": "/v1/candles/minutes/3",
        "5m": "/v1/candles/minutes/5",
        "10m": "/v1/candles/minutes/10",
        "15m": "/v1/candles/minutes/15",
        "30m": "/v1/candles/minutes/30",
        "60m": "/v1/candles/minutes/60",
        "240m": "/v1/candles/minutes/240",
        "1d": "/v1/candles/days",
        "1w": "/v1/candles/weeks",
        "1M": "/v1/candles/months",
        "1y": "/v1/candles/years",
    }
    if tf not in mapping:
        raise ValueError(f"unsupported timeframe: {tf}")
    params: dict[str, Any] = {"market": market, "count": count}
    if to:
        params["to"] = to
    path = mapping[tf]
    data, headers = await _get(path, params)
    latest = data[0] if data else {}
    return {
        **_wrap(data, headers, path),
        "market": market,
        "timeframe": tf,
        "latest_candle_utc": latest.get("candle_date_time_utc"),
        "latest_candle_kst": latest.get("candle_date_time_kst"),
        "latest_source_timestamp_ms": latest.get("timestamp"),
        "latest_source_age_seconds": _ms_age_seconds(latest.get("timestamp")),
    }


@mcp.tool()
async def get_recent_trades(market: str, count: int = 200, to: str | None = None) -> dict[str, Any]:
    """Get recent trades for a pair. Maximum count is capped at 200 by this server."""
    params: dict[str, Any] = {"market": market, "count": max(1, min(int(count), 200))}
    if to:
        params["to"] = to
    data, headers = await _get("/v1/trades/ticks", params)
    return _wrap(data, headers, "/v1/trades/ticks")


async def _analyze_tf(market: str, timeframe: str, count: int = 200) -> dict[str, Any]:
    path_map = {
        "5m": "/v1/candles/minutes/5",
        "15m": "/v1/candles/minutes/15",
        "60m": "/v1/candles/minutes/60",
        "240m": "/v1/candles/minutes/240",
        "1d": "/v1/candles/days",
    }
    path = path_map[timeframe]
    raw, headers = await _cached_candles(path, market, max(60, min(count, 200)))
    if len(raw) < 35:
        return {"timeframe": timeframe, "error": "insufficient candles", "count": len(raw)}
    candles = list(reversed(raw))  # ascending
    close = [float(c["trade_price"]) for c in candles]
    volumes = [float(c.get("candle_acc_trade_volume", 0) or 0) for c in candles]
    quote_values = [float(c.get("candle_acc_trade_price", 0) or 0) for c in candles]

    rsi14 = _rsi(close, 14)
    ma5 = _sma(close, 5)
    ma10 = _sma(close, 10)
    ma20 = _sma(close, 20)
    ma60 = _sma(close, 60)
    ma120 = _sma(close, 120)
    ema20 = _ema(close, 20)
    ema60 = _ema(close, 60)
    macd = _macd(close)
    bb = _bollinger(close, 20, 2.0)
    atr14 = _atr(candles, 14)

    last = len(close) - 1
    v20 = _mean(volumes[max(0, last - 19) : last + 1])
    q20 = _mean(quote_values[max(0, last - 19) : last + 1])
    bb_width = None
    bb_width_prev = None
    bb_pctb = None
    if bb["upper"][last] is not None and bb["lower"][last] is not None and bb["middle"][last] is not None:
        up = float(bb["upper"][last])
        lo = float(bb["lower"][last])
        mid = float(bb["middle"][last])
        if mid:
            bb_width = (up - lo) / mid
        if up != lo:
            bb_pctb = (close[last] - lo) / (up - lo)
    if last > 0 and bb["middle"][last - 1]:
        bb_width_prev = (bb["upper"][last - 1] - bb["lower"][last - 1]) / bb["middle"][last - 1]

    def _ret_bars(n: int) -> float | None:
        if last - n < 0 or close[last - n] == 0:
            return None
        return (close[last] / close[last - n] - 1.0) * 100.0

    def _recent_window_stats(n: int) -> dict[str, float | int | None]:
        start = max(0, last - n + 1)
        window = candles[start:last + 1]
        if not window:
            return {"high": None, "low": None, "runup_from_low_pct": None, "drawdown_from_high_pct": None, "max_bar_gain_pct": None, "upper_wick_ratio": None, "up_bars": 0}
        highs = [float(c["high_price"]) for c in window]
        lows = [float(c["low_price"]) for c in window]
        hi, lo = max(highs), min(lows)
        gains = []
        upper_wicks = []
        up_bars = 0
        for c in window:
            o = float(c.get("opening_price", 0) or 0)
            h = float(c.get("high_price", 0) or 0)
            l = float(c.get("low_price", 0) or 0)
            cl = float(c.get("trade_price", 0) or 0)
            if o:
                gains.append((cl / o - 1.0) * 100.0)
            if cl > o:
                up_bars += 1
            rng = h - l
            if rng > 0:
                upper_wicks.append(max(0.0, h - max(o, cl)) / rng)
        return {
            "high": hi,
            "low": lo,
            "runup_from_low_pct": ((close[last] / lo - 1.0) * 100.0) if lo else None,
            "drawdown_from_high_pct": ((close[last] / hi - 1.0) * 100.0) if hi else None,
            "max_bar_gain_pct": max(gains) if gains else None,
            "upper_wick_ratio": _mean(upper_wicks),
            "up_bars": up_bars,
        }

    recent7 = _recent_window_stats(7)
    recent14 = _recent_window_stats(14)

    # Stochastic RSI (14,14): normalized RSI position in its recent 14-value range.
    stoch_rsi = None
    valid_rsi = [x for x in rsi14[max(0, last - 13): last + 1] if x is not None]
    if len(valid_rsi) >= 2:
        rlo, rhi = min(valid_rsi), max(valid_rsi)
        if rhi != rlo and rsi14[last] is not None:
            stoch_rsi = (float(rsi14[last]) - rlo) / (rhi - rlo) * 100.0

    latest_c = candles[last]
    prev_c = candles[last - 1] if last > 0 else latest_c
    candle_range = float(latest_c['high_price']) - float(latest_c['low_price'])
    upper_wick = max(0.0, float(latest_c['high_price']) - max(float(latest_c['opening_price']), float(latest_c['trade_price'])))
    lower_wick = max(0.0, min(float(latest_c['opening_price']), float(latest_c['trade_price'])) - float(latest_c['low_price']))
    latest = raw[0]
    return {
        "timeframe": timeframe,
        "candle_count": len(raw),
        "latest_candle_kst": latest.get("candle_date_time_kst"),
        "latest_source_age_seconds": _ms_age_seconds(latest.get("timestamp")),
        "close": close[last],
        "open": float(latest["opening_price"]),
        "rsi14": rsi14[last],
        "stoch_rsi14": stoch_rsi,
        "ma5": ma5[last],
        "ma10": ma10[last],
        "ma20": ma20[last],
        "ma60": ma60[last],
        "ma120": ma120[last],
        "ma5_slope_pct": ((ma5[last] / ma5[last-1] - 1.0) * 100.0) if last > 0 and ma5[last] and ma5[last-1] else None,
        "ma10_slope_pct": ((ma10[last] / ma10[last-1] - 1.0) * 100.0) if last > 0 and ma10[last] and ma10[last-1] else None,
        "ma20_slope_pct": ((ma20[last] / ma20[last-1] - 1.0) * 100.0) if last > 0 and ma20[last] and ma20[last-1] else None,
        "ma_bullish_stack_5_10_20": bool(ma5[last] and ma10[last] and ma20[last] and ma5[last] > ma10[last] > ma20[last]),
        "ema20": ema20[last],
        "ema60": ema60[last],
        "ema20_above_ema60": (ema20[last] is not None and ema60[last] is not None and ema20[last] > ema60[last]),
        "macd": macd["macd"][last],
        "macd_signal": macd["signal"][last],
        "macd_histogram": macd["histogram"][last],
        "macd_histogram_prev": macd["histogram"][last - 1] if last > 0 else None,
        "macd_bullish": bool(macd["macd"][last] is not None and macd["signal"][last] is not None and macd["macd"][last] > macd["signal"][last]),
        "macd_histogram_rising": bool(last > 0 and macd["histogram"][last] is not None and macd["histogram"][last-1] is not None and macd["histogram"][last] > macd["histogram"][last-1]),
        "macd_histogram_delta": (
            (macd["histogram"][last] - macd["histogram"][last - 1])
            if last > 0 and macd["histogram"][last] is not None and macd["histogram"][last - 1] is not None
            else None
        ),
        "bollinger_upper": bb["upper"][last],
        "bollinger_middle": bb["middle"][last],
        "bollinger_lower": bb["lower"][last],
        "bollinger_percent_b": bb_pctb,
        "bollinger_width": bb_width,
        "bollinger_width_prev": bb_width_prev,
        "atr14": atr14[last],
        "atr14_pct": (atr14[last] / close[last]) if atr14[last] is not None and close[last] else None,
        "volume": volumes[last],
        "volume_sma20": v20,
        "volume_ratio20": (volumes[last] / v20) if v20 else None,
        "quote_volume": quote_values[last],
        "quote_volume_sma20": q20,
        "candle_body_pct": ((float(latest_c["trade_price"]) / float(latest_c["opening_price"]) - 1.0) * 100.0) if float(latest_c["opening_price"]) else None,
        "upper_wick_ratio": (upper_wick / candle_range) if candle_range else 0.0,
        "lower_wick_ratio": (lower_wick / candle_range) if candle_range else 0.0,
        "prev_close": float(prev_c["trade_price"]),
        "recent_high_20": max(float(c["high_price"]) for c in candles[max(0, last - 19): last + 1]),
        "recent_low_20": min(float(c["low_price"]) for c in candles[max(0, last - 19): last + 1]),
        "return_1bar_pct": _ret_bars(1),
        "return_3bar_pct": _ret_bars(3),
        "return_5bar_pct": _ret_bars(5),
        "return_7bar_pct": _ret_bars(7),
        "return_14bar_pct": _ret_bars(14),
        "recent7": recent7,
        "recent14": recent14,
        "remaining_req": headers.get("remaining-req"),
    }


async def _analyze_market_full(
    market: str, ticker_row: dict[str, Any] | None = None,
    precomputed: dict[str, Any] | None = None,
    orderbook_row: dict[str, Any] | None = None,
    orderbook_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    tf_results = dict(precomputed or {})
    missing = [tf for tf in ("5m", "15m", "60m", "240m", "1d") if tf not in tf_results]
    results = await asyncio.gather(*(_analyze_tf(market, tf) for tf in missing))
    tf_results.update(zip(missing, results))
    if any(t.get("error") for t in tf_results.values()):
        raise ValueError("insufficient candles for full analysis")
    ticker_h: dict[str, str] = {}
    if ticker_row is None:
        ticker, ticker_h = await _get("/v1/ticker", {"markets": market})
        ticker_row = ticker[0] if ticker else {}
    if orderbook_row is None:
        orderbook, ob_h = await _get("/v1/orderbook", {"markets": market})
        ob_row = orderbook[0] if orderbook else {}
    else:
        ob_row, ob_h = orderbook_row, orderbook_headers or {}
    units = ob_row.get("orderbook_units") or []
    top10_bid = sum(float(u.get("bid_size", 0) or 0) for u in units[:10])
    top10_ask = sum(float(u.get("ask_size", 0) or 0) for u in units[:10])
    denom = top10_bid + top10_ask

    core = [tf_results.get(k, {}) for k in ["15m", "60m", "240m", "1d"]]
    all_macd_bullish = all(bool(x.get("macd_bullish")) for x in core)
    bullish_count = sum(1 for x in core if x.get("macd_bullish"))
    rising_hist_count = sum(1 for x in core if x.get("macd_histogram_rising"))
    trend_snapshot = {
        "all_15m_1h_4h_1d_macd_bullish": all_macd_bullish,
        "macd_bullish_timeframes": bullish_count,
        "macd_histogram_rising_timeframes": rising_hist_count,
        "15m_price_up": (tf_results.get("15m", {}).get("close", 0) > tf_results.get("15m", {}).get("open", 0)),
        "15m_bb_expanding": ((tf_results.get("15m", {}).get("bollinger_width") or 0) > (tf_results.get("15m", {}).get("bollinger_width_prev") or 0)),
        "15m_volume_ratio20": tf_results.get("15m", {}).get("volume_ratio20"),
        "1h_volume_ratio20": tf_results.get("60m", {}).get("volume_ratio20"),
    }

    return {
        "market": market,
        "received_at_utc": _utc_now_iso(),
        "trend_snapshot": trend_snapshot,
        "ticker": {
            **ticker_row,
            "source_age_seconds": _ms_age_seconds(ticker_row.get("timestamp")),
            "remaining_req": ticker_h.get("remaining-req"),
        },
        "orderbook_summary": {
            "source_age_seconds": _ms_age_seconds(ob_row.get("timestamp")),
            "best_ask": units[0].get("ask_price") if units else None,
            "best_bid": units[0].get("bid_price") if units else None,
            "spread_bps": (
                ((float(units[0].get("ask_price")) - float(units[0].get("bid_price"))) /
                 ((float(units[0].get("ask_price")) + float(units[0].get("bid_price"))) / 2.0) * 10000.0)
                if units and units[0].get("ask_price") and units[0].get("bid_price") else None
            ),
            "top10_bid_size": top10_bid,
            "top10_ask_size": top10_ask,
            "top10_imbalance": ((top10_bid - top10_ask) / denom) if denom else None,
            "remaining_req": ob_h.get("remaining-req"),
        },
        "timeframes": tf_results,
    }



@mcp.tool()
async def analyze_market(market: str) -> dict[str, Any]:
    """Technical analysis for one market using fresh Upbit 5m/15m/60m/240m/1d candles plus ticker and orderbook."""
    return await _analyze_market_full(market)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ticker_prescore(row: dict[str, Any], mode: str = "swing5d") -> float:
    """Fast first-pass screen. Used only to choose which markets get expensive multi-timeframe analysis."""
    change_pct = float(row.get("signed_change_rate", 0) or 0) * 100.0
    turnover = float(row.get("acc_trade_price_24h", 0) or 0)
    high = float(row.get("high_price", 0) or 0)
    low = float(row.get("low_price", 0) or 0)
    price = float(row.get("trade_price", 0) or 0)
    day_range_pct = (((high - low) / price) * 100.0) if price else 0.0
    liquidity = _clamp((math.log10(max(turnover, 1.0)) - 8.5) * 3.0, 0.0, 12.0)

    if mode == "day":
        momentum = change_pct * (1.25 if change_pct >= 0 else 0.45)
        extreme_penalty = max(0.0, change_pct - 18.0) * 0.9
        return liquidity + momentum + min(day_range_pct, 20.0) * 0.20 - extreme_penalty

    if -3.0 <= change_pct <= 8.0:
        momentum = 4.0 + change_pct * 0.45
    elif 8.0 < change_pct <= 15.0:
        momentum = 7.6 - (change_pct - 8.0) * 0.35
    elif change_pct > 15.0:
        momentum = 5.15 - (change_pct - 15.0) * 0.85
    else:
        momentum = 2.5 + change_pct * 0.35
    return liquidity * 1.25 + momentum + min(day_range_pct, 18.0) * 0.10


def _liquidity_score(turnover: float, max_points: float = 15.0) -> float:
    raw = (math.log10(max(turnover, 1.0)) - 9.0) / 3.0
    return _clamp(raw * max_points, 0.0, max_points)


def _ratio_quality(ratio: float | None) -> float:
    if ratio is None:
        return 0.0
    r = float(ratio)
    if 1.2 <= r <= 4.0:
        return 1.0
    if 1.0 <= r < 1.2:
        return 0.75
    if 0.6 <= r < 1.0:
        return 0.45
    if 4.0 < r <= 7.0:
        return 0.65
    if r > 7.0:
        return 0.35
    if 0.3 <= r < 0.6:
        return 0.20
    return 0.05


def _surge_drive_profile(analysis: dict[str, Any], ticker_row: dict[str, Any]) -> dict[str, Any]:
    """Assess recent surge/retrace risk and classify momentum 'drive' style. Not a probability model."""
    tfs = analysis.get("timeframes", {}) if isinstance(analysis, dict) else {}
    ob = analysis.get("orderbook_summary", {}) if isinstance(analysis, dict) else {}

    def tf(name: str) -> dict[str, Any]:
        x = tfs.get(name, {})
        return x if isinstance(x, dict) and not x.get("error") else {}

    d = tf("1d")
    h4 = tf("240m")
    h1 = tf("60m")
    m15 = tf("15m")
    m5 = tf("5m")
    change24 = float(ticker_row.get("signed_change_rate", 0) or 0) * 100.0
    r3 = float(d.get("return_3bar_pct") or 0.0)
    r5 = float(d.get("return_5bar_pct") or 0.0)
    r7 = float(d.get("return_7bar_pct") or 0.0)
    r14 = float(d.get("return_14bar_pct") or 0.0)
    rec7 = d.get("recent7") or {}
    runup7 = float(rec7.get("runup_from_low_pct") or 0.0)
    dd7 = float(rec7.get("drawdown_from_high_pct") or 0.0)
    max_day_gain7 = float(rec7.get("max_bar_gain_pct") or 0.0)
    upper_wick = float(rec7.get("upper_wick_ratio") or 0.0)
    dvol = float(d.get("volume_ratio20") or 0.0)
    imbalance = ob.get("top10_imbalance")
    imbalance = float(imbalance) if imbalance is not None else 0.0

    # Surge intensity 0-100: combines multi-day return, run-up and single-day impulse.
    surge = 0.0
    surge += _clamp(max(r3, 0.0) / 15.0, 0.0, 1.0) * 24.0
    surge += _clamp(max(r5, 0.0) / 24.0, 0.0, 1.0) * 24.0
    surge += _clamp(max(r7, 0.0) / 32.0, 0.0, 1.0) * 18.0
    surge += _clamp(max(runup7, 0.0) / 35.0, 0.0, 1.0) * 18.0
    surge += _clamp(max(max_day_gain7, 0.0) / 16.0, 0.0, 1.0) * 10.0
    surge += _clamp(max(change24, 0.0) / 15.0, 0.0, 1.0) * 6.0
    surge = _clamp(surge, 0.0, 100.0)

    risk = 0.0
    # Recent multi-day verticality.
    risk += _clamp((r3 - 10.0) / 15.0, 0.0, 1.0) * 14.0
    risk += _clamp((r5 - 18.0) / 22.0, 0.0, 1.0) * 16.0
    risk += _clamp((runup7 - 22.0) / 28.0, 0.0, 1.0) * 12.0
    risk += _clamp((max_day_gain7 - 10.0) / 15.0, 0.0, 1.0) * 8.0

    # Overbought/extension.
    d_rsi = float(d.get("rsi14") or 0.0)
    h4_rsi = float(h4.get("rsi14") or 0.0)
    h1_rsi = float(h1.get("rsi14") or 0.0)
    d_b = float(d.get("bollinger_percent_b") or 0.0)
    h4_b = float(h4.get("bollinger_percent_b") or 0.0)
    risk += _clamp((d_rsi - 68.0) / 14.0, 0.0, 1.0) * 12.0
    risk += _clamp((h4_rsi - 72.0) / 16.0, 0.0, 1.0) * 10.0
    risk += _clamp((h1_rsi - 78.0) / 14.0, 0.0, 1.0) * 7.0
    risk += _clamp((d_b - 0.98) / 0.38, 0.0, 1.0) * 8.0
    risk += _clamp((h4_b - 1.02) / 0.38, 0.0, 1.0) * 8.0

    # Blow-off/distribution hints: oversized volume, upper wicks, selling orderbook, momentum rollover.
    if dvol > 4.0:
        risk += _clamp((dvol - 4.0) / 8.0, 0.0, 1.0) * 5.0
    if upper_wick > 0.28:
        risk += _clamp((upper_wick - 0.28) / 0.35, 0.0, 1.0) * 4.0
    if imbalance < -0.20:
        risk += _clamp((-imbalance - 0.20) / 0.45, 0.0, 1.0) * 6.0
    h1_delta = h1.get("macd_histogram_delta")
    h4_delta = h4.get("macd_histogram_delta")
    if surge >= 45 and h1_delta is not None and float(h1_delta) < 0:
        risk += 4.0
    if surge >= 45 and h4_delta is not None and float(h4_delta) < 0:
        risk += 4.0
    risk = _clamp(risk, 0.0, 100.0)

    aligned = sum(1 for x in (h1, h4, d) if x.get("ema20_above_ema60"))
    short_recover = 0
    for x in (m5, m15):
        delta = x.get("macd_histogram_delta")
        rsi = x.get("rsi14")
        if delta is not None and float(delta) > 0 and rsi is not None and 35 <= float(rsi) <= 58:
            short_recover += 1
    mid_positive = sum(1 for x in (h1, h4, d) if (x.get("macd_histogram") is not None and float(x.get("macd_histogram")) > 0))

    if surge >= 60 and risk >= 65:
        drive = "과열 급등형"
    elif surge >= 45 and dd7 <= -5.0 and ((h1_delta is not None and float(h1_delta) < 0) or imbalance < -0.20):
        drive = "분배·되돌림형"
    elif surge >= 35 and -16.0 <= dd7 <= -3.0 and short_recover >= 1 and aligned >= 2:
        drive = "눌림 후 재가속형"
    elif aligned == 3 and mid_positive >= 2 and risk < 55:
        drive = "추세 지속형"
    elif aligned >= 2 and surge < 40 and risk < 45:
        drive = "초기 드라이브형"
    else:
        drive = "혼합·중립형"

    if drive == "눌림 후 재가속형":
        bonus = 4.0
    elif drive == "초기 드라이브형":
        bonus = 3.0
    elif drive == "추세 지속형":
        bonus = 2.0
    elif drive == "분배·되돌림형":
        bonus = -2.0
    elif drive == "과열 급등형":
        bonus = -4.0
    else:
        bonus = 0.0

    return {
        "surge_intensity": round(surge, 1),
        "surge_risk": round(risk, 1),
        "drive_profile": drive,
        "drive_bonus": bonus,
        "recent_return_3d_pct": round(r3, 2),
        "recent_return_5d_pct": round(r5, 2),
        "recent_return_7d_pct": round(r7, 2),
        "recent_return_14d_pct": round(r14, 2),
        "runup_7d_pct": round(runup7, 2),
        "drawdown_from_7d_high_pct": round(dd7, 2),
        "max_daily_gain_7d_pct": round(max_day_gain7, 2),
        "upper_wick_ratio_7d": round(upper_wick, 3),
    }


def _score_market(analysis: dict[str, Any], ticker_row: dict[str, Any], mode: str = "swing5d") -> dict[str, Any]:
    """Return transparent rule-based score components. Scores are not probabilities."""
    tfs = analysis.get("timeframes", {}) if isinstance(analysis, dict) else {}
    ob = analysis.get("orderbook_summary", {}) if isinstance(analysis, dict) else {}
    surge_profile = _surge_drive_profile(analysis, ticker_row)
    change_pct = float(ticker_row.get("signed_change_rate", 0) or 0) * 100.0
    turnover = float(ticker_row.get("acc_trade_price_24h", 0) or 0)
    ticker_age = ticker_row.get("source_age_seconds")

    def tf(name: str) -> dict[str, Any]:
        x = tfs.get(name, {})
        return x if isinstance(x, dict) and not x.get("error") else {}

    trend_raw = 0.0
    volume_raw = 0.0
    entry_raw = 0.0
    heat_penalty = 0.0
    tags: list[str] = []

    if mode == "day":
        for name, w in [("5m", 0.9), ("15m", 1.2), ("60m", 1.5)]:
            x = tf(name)
            if not x:
                continue
            rsi, hist, delta = x.get("rsi14"), x.get("macd_histogram"), x.get("macd_histogram_delta")
            if x.get("ema20_above_ema60"):
                trend_raw += 3.0 * w
            if hist is not None and hist > 0:
                trend_raw += 3.0 * w
            if delta is not None and delta > 0:
                trend_raw += 1.0 * w
            if rsi is not None:
                if 50 <= rsi <= 68:
                    trend_raw += 2.0 * w
                elif 45 <= rsi < 50:
                    trend_raw += 0.8 * w
                elif rsi >= 78:
                    heat_penalty += 1.7 * w

        for name, w in [("5m", 1.0), ("15m", 1.3), ("60m", 1.5)]:
            volume_raw += _ratio_quality(tf(name).get("volume_ratio20")) * 5.0 * w

        for name in ("5m", "15m"):
            x = tf(name)
            if not x:
                continue
            rsi, hist, pctb = x.get("rsi14"), x.get("macd_histogram"), x.get("bollinger_percent_b")
            if rsi is not None and 42 <= rsi <= 62:
                entry_raw += 3.0
            elif rsi is not None and 62 < rsi <= 70:
                entry_raw += 1.2
            if hist is not None and hist > 0:
                entry_raw += 2.0
            if pctb is not None and 0.35 <= pctb <= 0.90:
                entry_raw += 1.5

        if change_pct > 15:
            heat_penalty += min(8.0, (change_pct - 15.0) * 0.7)
        for name in ("15m", "60m"):
            rsi = tf(name).get("rsi14")
            pctb = tf(name).get("bollinger_percent_b")
            if rsi is not None and rsi > 75:
                heat_penalty += min(4.0, (rsi - 75.0) * 0.5)
            if pctb is not None and pctb > 1.08:
                heat_penalty += min(3.0, (pctb - 1.08) * 10.0)

        trend_score = _clamp(trend_raw / 32.0 * 30.0, 0.0, 30.0)
        volume_score = _clamp(volume_raw / 19.0 * 25.0, 0.0, 25.0)
        entry_score = _clamp(entry_raw / 13.0 * 20.0, 0.0, 20.0)
        orderbook_max, liquidity_max = 15.0, 15.0
    else:
        aligned = 0
        for name, w in [("60m", 1.0), ("240m", 1.4), ("1d", 1.6)]:
            x = tf(name)
            if not x:
                continue
            rsi, hist, delta = x.get("rsi14"), x.get("macd_histogram"), x.get("macd_histogram_delta")
            if x.get("ema20_above_ema60"):
                trend_raw += 4.0 * w
                aligned += 1
            if hist is not None and hist > 0:
                trend_raw += 3.0 * w
            if delta is not None and delta > 0:
                trend_raw += 1.0 * w
            if rsi is not None:
                if 50 <= rsi <= 68:
                    trend_raw += 2.0 * w
                elif 45 <= rsi < 50:
                    trend_raw += 0.8 * w
                elif 68 < rsi <= 72:
                    trend_raw += 0.4 * w
        if aligned == 3:
            tags.append("1h·4h·1d 추세정렬")

        for name, w in [("60m", 1.0), ("240m", 1.25), ("1d", 1.5)]:
            volume_raw += _ratio_quality(tf(name).get("volume_ratio20")) * 4.0 * w
        if (tf("1d").get("volume_ratio20") or 0) >= 1.5:
            tags.append("일봉 거래량 증가")
        if (tf("60m").get("volume_ratio20") or 0) >= 1.5:
            tags.append("1h 거래량 증가")

        pullback = False
        for name in ("5m", "15m"):
            x = tf(name)
            if not x:
                continue
            rsi, hist, pctb = x.get("rsi14"), x.get("macd_histogram"), x.get("bollinger_percent_b")
            if rsi is not None and 35 <= rsi <= 55:
                entry_raw += 3.0
                pullback = True
            elif rsi is not None and 55 < rsi <= 65:
                entry_raw += 1.4
            if hist is not None and hist > 0:
                entry_raw += 1.5
            if x.get("ema20_above_ema60"):
                entry_raw += 1.0
            if pctb is not None and 0.25 <= pctb <= 0.80:
                entry_raw += 1.0
        if pullback and aligned >= 2:
            tags.append("중기상승·단기눌림")

        # Stronger anti-chasing rules for a 5-day entry horizon.
        if change_pct > 8:
            heat_penalty += min(10.0, (change_pct - 8.0) * 0.85)
        if change_pct > 12:
            heat_penalty += min(4.0, (change_pct - 12.0) * 0.45)
        if change_pct < -8:
            heat_penalty += min(4.0, abs(change_pct + 8.0) * 0.35)
        for name, rsi_cut, bb_cut, mult in [("60m", 78.0, 1.12, 0.8), ("240m", 70.0, 1.00, 1.25), ("1d", 70.0, 0.98, 1.45)]:
            x = tf(name)
            rsi, pctb = x.get("rsi14"), x.get("bollinger_percent_b")
            if rsi is not None and rsi > rsi_cut:
                heat_penalty += min(8.0, (rsi - rsi_cut) * 0.80 * mult)
            if pctb is not None and pctb > bb_cut:
                heat_penalty += min(7.0, (pctb - bb_cut) * 15.0 * mult)

        # Recent pump risk: catches coins that surged over several days even if today's change is modest.
        surge_risk = float(surge_profile.get("surge_risk", 0.0) or 0.0)
        surge_penalty = _clamp((surge_risk - 28.0) * 0.18, 0.0, 13.0)
        heat_penalty += surge_penalty
        if surge_risk >= 65:
            tags.append("최근급등 고위험")
        elif surge_risk >= 48:
            tags.append("최근급등 주의")
        drive = str(surge_profile.get("drive_profile") or "")
        if drive:
            tags.append(drive)
        if heat_penalty >= 8.0:
            tags.append("과열주의")

        trend_score = _clamp(trend_raw, 0.0, 40.0)
        volume_score = _clamp(volume_raw / 15.0 * 20.0, 0.0, 20.0)
        entry_score = _clamp(entry_raw / 13.0 * 15.0, 0.0, 15.0)
        orderbook_max, liquidity_max = 10.0, 15.0

    imbalance = ob.get("top10_imbalance") if isinstance(ob, dict) else None
    if imbalance is None:
        orderbook_score = orderbook_max * 0.5
    else:
        orderbook_score = _clamp(orderbook_max * (0.5 + float(imbalance) * 1.25), 0.0, orderbook_max)
        if float(imbalance) >= 0.12:
            tags.append("호가 매수우위")
        elif float(imbalance) <= -0.20:
            tags.append("호가 매도우위")

    spread_bps = ob.get("spread_bps") if isinstance(ob, dict) else None
    if spread_bps is not None and float(spread_bps) > 40:
        heat_penalty += min(3.0, (float(spread_bps) - 40.0) / 25.0)

    liquidity_score = _liquidity_score(turnover, liquidity_max)

    freshness_penalty = 0.0
    if ticker_age is not None and float(ticker_age) > 15:
        freshness_penalty += min(8.0, (float(ticker_age) - 15.0) / 10.0)
    candle_ages = [tf(n).get("latest_source_age_seconds") for n in ("5m", "15m", "60m", "240m", "1d")]
    candle_ages = [float(x) for x in candle_ages if x is not None]
    if candle_ages and max(candle_ages) > 120:
        freshness_penalty += min(8.0, (max(candle_ages) - 120.0) / 60.0)

    drive_bonus = float(surge_profile.get("drive_bonus", 0.0) or 0.0) if mode == "swing5d" else 0.0
    total = trend_score + volume_score + entry_score + orderbook_score + liquidity_score + drive_bonus - heat_penalty - freshness_penalty
    total = _clamp(total, 0.0, 100.0)

    return {
        "mode": mode,
        "mode_label": "오늘 단타" if mode == "day" else "5일 스윙",
        "total_score": round(total, 3),
        "trend_score": round(trend_score, 3),
        "volume_score": round(volume_score, 3),
        "entry_score": round(entry_score, 3),
        "orderbook_score": round(orderbook_score, 3),
        "liquidity_score": round(liquidity_score, 3),
        "heat_penalty": round(heat_penalty, 3),
        "freshness_penalty": round(freshness_penalty, 3),
        "surge_intensity": surge_profile.get("surge_intensity"),
        "surge_risk": surge_profile.get("surge_risk"),
        "drive_profile": surge_profile.get("drive_profile"),
        "drive_bonus": round(drive_bonus, 3),
        "recent_surge": surge_profile,
        "tags": tags[:7],
    }


def _fast_shortlist(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"EDGE": [], "AXS": [], "Pre-Breakout": [], "Overheated": []}
    for row in rows:
        change = float(row.get("signed_change_rate") or 0) * 100
        high, low, price = (float(row.get(k) or 0) for k in ("high_price", "low_price", "trade_price"))
        position = (price - low) / (high - low) if high > low else 0.5
        group = "Overheated" if change > 18 else "EDGE" if change >= 3 else "AXS" if position < 0.55 else "Pre-Breakout"
        row["candidate_group"] = group
        row["prescore"] = _ticker_prescore(row, "day") if group == "EDGE" else math.log10(max(float(row.get("acc_trade_price_24h") or 0), 1)) + position * 2 - abs(change) * 0.1
        groups[group].append(row)
    for values in groups.values():
        values.sort(key=lambda r: r["prescore"], reverse=True)
    chosen = []
    # Round-robin guarantees low-change groups participate before momentum fills the pool.
    while len(chosen) < size:
        added = False
        for name in ("EDGE", "AXS", "Pre-Breakout"):
            if groups[name] and len(chosen) < size:
                chosen.append(groups[name].pop(0)); added = True
        if not added:
            break
    return chosen


def _early_score(tfs: dict[str, Any]) -> float:
    score = 0.0
    for t in tfs.values():
        score += 3 * bool(t.get("macd_bullish")) + 3 * bool(t.get("macd_histogram_rising"))
        score += 2 * bool(t.get("ema20_above_ema60"))
        width, prev = t.get("bollinger_width"), t.get("bollinger_width_prev")
        score += 2 if width is not None and width < 0.08 else 0
        score += 1 if width is not None and prev is not None and width > prev else 0
        score += min(float(t.get("volume_ratio20") or 0), 3)
        rsi = t.get("rsi14")
        score += 2 if rsi is not None and 40 <= rsi <= 68 else 0
        score -= 4 if rsi is not None and rsi > 80 else 0
    return score


async def _legacy_scan_krw_market(
    top_n: int = 5, shortlist_size: int = 24,
    min_turnover_krw: float = 1_000_000_000, mode: str = "day",
) -> dict[str, Any]:
    """FAST: diverse ticker candidates -> 15m/1h -> top six full analysis. Swing remains opt-in."""
    started = time.monotonic()
    mode = (mode or "day").strip().lower()
    if mode not in {"day", "swing5d"}:
        raise ValueError("mode must be 'day' or 'swing5d'")
    top_n = max(1, min(int(top_n), 10))
    shortlist_size = max(top_n, min(int(shortlist_size), 30))
    tickers, headers = await _get("/v1/ticker/all", {"quote_currencies": "KRW"})
    rows = [dict(r) for r in tickers if float(r.get("acc_trade_price_24h") or 0) >= min_turnover_krw]
    shortlist = _fast_shortlist(rows, shortlist_size)
    errors = []
    async def early(row: dict[str, Any]) -> dict[str, Any] | None:
        try:
            values = await asyncio.gather(*(_analyze_tf(row["market"], tf) for tf in ("15m", "60m")))
            tfs = dict(zip(("15m", "60m"), values))
            if any(t.get("error") for t in values):
                raise ValueError("insufficient candles")
            return {"row": row, "timeframes": tfs, "early_score": _early_score(tfs)}
        except Exception as exc:
            errors.append({"market": row["market"], "error": str(exc), "stage": "early"})
            return None
    early_results = [r for r in await asyncio.gather(*(early(row) for row in shortlist)) if r is not None]
    early_results.sort(key=lambda r: r["early_score"], reverse=True)
    finalists = early_results[:max(6, top_n)]
    books, book_headers = ([], {})
    if finalists:
        books, book_headers = await _get("/v1/orderbook", {"markets": ",".join(r["row"]["market"] for r in finalists)})
    book_map = {r["market"]: r for r in books}
    async def full(item: dict[str, Any]) -> dict[str, Any] | None:
        row = item["row"]
        try:
            if row["market"] not in book_map:
                raise ValueError("missing batch orderbook")
            a = await _analyze_market_full(row["market"], row, item["timeframes"], book_map[row["market"]], book_headers)
            score = _score_market(a, a["ticker"], mode)
            return {"market": row["market"], "trade_price": row.get("trade_price"),
                    "signed_change_rate": row.get("signed_change_rate"), "acc_trade_price_24h": row.get("acc_trade_price_24h"),
                    "ticker_source_age_seconds": a["ticker"]["source_age_seconds"], "prescore": row["prescore"],
                    "candidate_group": row["candidate_group"], "early_score": item["early_score"],
                    "technical_screen_score": score["total_score"], "score_breakdown": score, "analysis": a}
        except Exception as exc:
            errors.append({"market": row["market"], "error": str(exc), "stage": "full"})
            return None
    valid = [r for r in await asyncio.gather(*(full(item) for item in finalists)) if r is not None]
    valid.sort(key=lambda r: r["technical_screen_score"], reverse=True)
    return {"received_at_utc": _utc_now_iso(), "mode": mode,
            "mode_label": "오늘 단타 FAST" if mode == "day" else "5일 스윙",
            "method": "ticker snapshot -> diverse pools -> 15m/1h -> finalists 5m/4h/1d + batch orderbook",
            "important": "Heuristic screening scores, not probabilities. Ticker groups are proxies, not confirmed patterns.",
            "ticker_rows": len(tickers), "eligible_rows": len(rows), "shortlist_size": len(shortlist),
            "full_analysis_count": len(finalists), "elapsed_seconds": round(time.monotonic() - started, 3),
            "overheated_markets": [r["market"] for r in rows if r.get("candidate_group") == "Overheated"],
            "remaining_req": headers.get("remaining-req"), "top_candidates": valid[:top_n],
            "shortlist_results": valid, "errors": errors,
            "early_results": [{"market": r["row"]["market"], "candidate_group": r["row"]["candidate_group"], "early_score": r["early_score"]} for r in early_results]}


def _daily_pattern(raw: list[dict[str, Any]], as_of: datetime | None = None) -> dict[str, Any]:
    """Causal daily setup: use only bars closed before as_of, never today's high.

    Thresholds are broad screening heuristics, not fitted probabilities. A setup
    is a watch candidate; it does not imply that a breakout will occur tomorrow.
    """
    as_of = as_of or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    cutoff = as_of.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    unique = {c["candle_date_time_utc"]: c for c in raw}
    bars = [c for k, c in sorted(unique.items())
            if datetime.fromisoformat(k).replace(tzinfo=timezone.utc) < cutoff]
    if len(bars) < 60:
        return {"matched": False, "reason": "insufficient_closed_days", "closed_days": len(bars)}
    last_date = datetime.fromisoformat(bars[-1]["candle_date_time_utc"]).replace(tzinfo=timezone.utc)
    if (cutoff - last_date).days > 1:
        return {"matched": False, "reason": "stale_daily_history", "closed_days": len(bars)}
    c = [float(b["trade_price"]) for b in bars]
    h = [float(b["high_price"]) for b in bars]
    l = [float(b["low_price"]) for b in bars]
    v = [float(b.get("candle_acc_trade_volume") or 0) for b in bars]
    ma5, ma20 = _sma(c, 5), _sma(c, 20)
    hist = _macd(c)["histogram"]
    n = len(c) - 1
    # Most recent occurrence of the highest high in the previous 14 sessions.
    peak = max(range(n - 13, n + 1), key=lambda i: (h[i], i))
    days = n - peak
    base = min(l[max(0, peak - 20):peak])
    peak_price = h[peak]
    if base <= 0 or peak_price <= base:
        return {"matched": False, "reason": "no_prior_impulse", "closed_days": len(bars)}
    impulse = (peak_price / base - 1) * 100
    pullback = (1 - c[n] / peak_price) * 100
    correction_low = min(l[peak + 1:]) if days else l[n]
    retracement = (peak_price - correction_low) / (peak_price - base)
    vol_impulse = _mean(v[max(0, peak - 2):peak + 1]) or 0
    vol_pullback = _mean(v[max(peak + 1, n - 2):n + 1]) or 0
    volume_dry_ratio = vol_pullback / vol_impulse if vol_impulse else None
    ma20_slope = (ma20[n] / ma20[n - 3] - 1) * 100 if ma20[n - 3] else 0
    trend = ma20_slope > 0 and c[n] >= ma20[n] * 0.97
    recent_hist = [x for x in hist[max(peak, n - 8):n + 1] if x is not None]
    momentum_reset = len(recent_hist) > 1 and min(recent_hist) < max(recent_hist)
    hist_turn = hist[n] is not None and hist[n - 1] is not None and hist[n] > hist[n - 1]
    ma5_turn = bool(ma5[n] and ma5[n - 1] and ma5[n] > ma5[n - 1])
    matched = bool(2 <= days <= 12 and impulse >= 15 and 3 <= pullback <= 40
                   and retracement <= 0.75 and trend)
    score = (30 + 15 * trend + 15 * (volume_dry_ratio is not None and volume_dry_ratio < 0.7)
             + 10 * momentum_reset + 10 * hist_turn + 10 * ma5_turn
             + 10 * (c[n] >= c[n - 1])) if matched else 0
    return {
        "matched": matched, "reason": "daily_pullback_setup" if matched else "structure_not_met",
        "score": round(score, 2), "closed_days": len(bars),
        "setup_as_of_kst": bars[-1].get("candle_date_time_kst"),
        "peak_date_kst": bars[peak].get("candle_date_time_kst"),
        "impulse_pct": round(impulse, 2), "pullback_pct": round(pullback, 2),
        "pullback_days": days, "retracement_ratio": round(retracement, 3),
        "volume_dry_ratio": round(volume_dry_ratio, 3) if volume_dry_ratio is not None else None,
        "ma20_slope_3d_pct": round(ma20_slope, 3), "ma5_turn_up": ma5_turn,
        "macd_histogram_turn_up": hist_turn, "momentum_reset": momentum_reset,
        "last_closed_price": c[n], "ma5": ma5[n], "ma20": ma20[n],
        "prior_peak": peak_price, "support": correction_low,
        "trigger": max(h[n - 1:n + 1]),
        "basis": "closed daily bars only; excludes current daily bar",
    }


def _pattern_state(pattern: dict[str, Any], row: dict[str, Any]) -> tuple[str, list[str]]:
    price = float(row.get("trade_price") or 0)
    change = float(row.get("signed_change_rate") or 0) * 100
    risks = []
    if change > 18:
        risks.append("당일 18% 초과·추격주의")
    if not pattern.get("matched"):
        return ("extended" if change > 18 else "unmatched"), risks
    if price < pattern["support"]:
        return "invalidated", risks + ["조정 저점 이탈"]
    extension = (price / pattern["ma20"] - 1) * 100 if pattern.get("ma20") else 0
    if change > 18 or extension > 35:
        return "extended", risks + (["20일선 이격 35% 초과"] if extension > 35 else [])
    if price > pattern["trigger"] and price > pattern["ma5"]:
        return "triggered", risks
    return "watch", risks


_pattern_scan_lock = asyncio.Lock()
_daily_setup_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}


async def _setup_daily_history(market: str, as_of: datetime) -> list[dict[str, Any]]:
    """Closed setup bars do not change during a UTC day. Live daily indicators
    still use _cached_candles separately for shortlisted markets.
    """
    day = as_of.astimezone(timezone.utc).date().isoformat()
    key = (market, day)
    if key in _daily_setup_cache:
        return _daily_setup_cache[key]
    raw, _ = await _cached_candles("/v1/candles/days", market, 200)
    closed = [c for c in raw if c["candle_date_time_utc"][:10] < day]
    # Do not retain missing/stale responses until tomorrow; allow a retry.
    previous_day = datetime.fromtimestamp(as_of.timestamp() - 86400, timezone.utc).date().isoformat()
    if closed and closed[0]["candle_date_time_utc"][:10] == previous_day:
        for old in [k for k in _daily_setup_cache if k[0] == market and k != key]:
            del _daily_setup_cache[old]
        _daily_setup_cache[key] = closed
    return closed


@mcp.tool()
async def scan_krw_market(
    top_n: int = 5, shortlist_size: int = 24,
    min_turnover_krw: float = 1_000_000_000, mode: str = "day",
) -> dict[str, Any]:
    """Day: EVERY KRW daily setup -> 4h/1h -> finalists. Includes all setup
    matches and extended movers separately. Scores are not probabilities.
    swing5d retains the legacy opt-in scanner. No orders are submitted.
    """
    mode = (mode or "day").strip().lower()
    if mode == "swing5d":
        return await _legacy_scan_krw_market(top_n, shortlist_size, min_turnover_krw, mode)
    if mode != "day":
        raise ValueError("mode must be 'day' or 'swing5d'")
    async with _pattern_scan_lock:
        return await _scan_daily_first(top_n, shortlist_size, min_turnover_krw)


async def _scan_daily_first(top_n: int, shortlist_size: int, min_turnover_krw: float) -> dict[str, Any]:
    started = time.monotonic()
    as_of = datetime.now(timezone.utc)
    top_n = max(1, min(int(top_n), 10))
    shortlist_size = max(top_n, min(int(shortlist_size), 30))
    min_turnover_krw = max(0, float(min_turnover_krw))
    tickers, headers = await _get("/v1/ticker/all", {"quote_currencies": "KRW"})
    rows = [dict(r) for r in tickers if r["market"].startswith("KRW-")]
    errors, checked, excluded = [], [], []
    details, _ = await _get("/v1/market/all", {"is_details": "true"})
    events = {r["market"]: r.get("market_event", {"warning": r.get("market_warning") == "CAUTION"}) for r in details}
    semaphore = asyncio.Semaphore(6)
    stablecoins = {"KRW-USDT", "KRW-USDC", "KRW-DAI", "KRW-USDE", "KRW-USDS", "KRW-USD1"}

    async def daily(row: dict[str, Any]) -> None:
        async with semaphore:
            try:
                raw = await _setup_daily_history(row["market"], as_of)
                p = _daily_pattern(raw, as_of)
                checked.append({"row": row, "pattern": p})
            except Exception as exc:
                errors.append({"market": row["market"], "stage": "daily", "error": str(exc)})
    await asyncio.gather(*(daily(row) for row in rows))
    patterns, movers = [], []
    labels = {"watch": "재상승 대기", "triggered": "재상승 신호(미확정)",
              "extended": "이미 급등·추격주의", "invalidated": "지지 이탈", "unmatched": "패턴 불일치"}

    def pack(row: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
        state, risks = _pattern_state(p, row)
        event = events.get(row["market"], {})
        turnover = float(row.get("acc_trade_price_24h") or 0)
        if turnover < min_turnover_krw:
            risks.append("거래대금 기준 미달")
        if event.get("warning"):
            risks.append("거래 유의 지정")
        risks += ["주의: " + k for k, value in event.get("caution", {}).items() if value]
        if row["market"] in stablecoins:
            risks.append("스테이블코인 제외")
        if row["market"] not in events:
            risks.append("유의 상태 확인 불가")
        trade_age = _ms_age_seconds(row.get("trade_timestamp") or row.get("timestamp"))
        fresh_trade = trade_age is not None and trade_age <= 120
        if not fresh_trade:
            risks.append("최근 체결 2분 초과 또는 시각 불명")
        return {"market": row["market"], "trade_price": row.get("trade_price"),
                "signed_change_rate": row.get("signed_change_rate"), "acc_trade_price_24h": turnover,
                "ticker_source_age_seconds": _ms_age_seconds(row.get("timestamp")),
                "last_trade_age_seconds": trade_age,
                "last_trade_kst": str(row.get("trade_date_kst", "")) + " " + str(row.get("trade_time_kst", "")),
                "daily_pattern": p, "state": state, "state_label": labels[state], "risk_flags": risks,
                "market_event": event, "entry_eligible": bool(p.get("matched") and state in {"watch", "triggered"}
                    and turnover >= min_turnover_krw and not event.get("warning")
                    and fresh_trade and row["market"] in events and row["market"] not in stablecoins)}

    for item in checked:
        x = pack(item["row"], item["pattern"])
        if item["pattern"].get("matched"):
            patterns.append(x)
        else:
            excluded.append({"market": x["market"], "reason": item["pattern"].get("reason")})
        if float(item["row"].get("signed_change_rate") or 0) > 0.18:
            movers.append(x)
    eligible = sorted([x for x in patterns if x["entry_eligible"]],
                      key=lambda x: (x["daily_pattern"]["score"], x["acc_trade_price_24h"]), reverse=True)
    # Retain both waiting and triggered setups before filling by daily score.
    shortlist = []
    pools = [[x for x in eligible if x["state"] == state] for state in ("watch", "triggered")]
    while len(shortlist) < shortlist_size and any(pools):
        for pool in pools:
            if pool and len(shortlist) < shortlist_size:
                shortlist.append(pool.pop(0))

    async def confirm(x: dict[str, Any]) -> dict[str, Any] | None:
        async with semaphore:
            try:
                values = await asyncio.gather(*(_analyze_tf(x["market"], tf) for tf in ("240m", "60m")))
                if any(t.get("error") for t in values):
                    raise ValueError("insufficient confirmation candles")
                tfs = dict(zip(("240m", "60m"), values))
                confirmation = sum(5 * bool(t.get("macd_bullish")) + 5 * bool(t.get("macd_histogram_rising")) for t in values)
                return {**x, "early_score": x["daily_pattern"]["score"] * 0.8 + confirmation, "tfs": tfs}
            except Exception as exc:
                errors.append({"market": x["market"], "stage": "confirmation", "error": str(exc)})
                return None
    confirmed = [x for x in await asyncio.gather(*(confirm(x) for x in shortlist)) if x]
    confirmed.sort(key=lambda x: x["early_score"], reverse=True)
    finalists = confirmed[:max(6, top_n)]
    books, book_headers = ([], {})
    if finalists:
        books, book_headers = await _get("/v1/orderbook", {"markets": ",".join(x["market"] for x in finalists)})
    book_map = {b["market"]: b for b in books}

    async def full(x: dict[str, Any]) -> dict[str, Any] | None:
        async with semaphore:
            try:
                if x["market"] not in book_map:
                    raise ValueError("missing orderbook")
                a = await _analyze_market_full(x["market"], precomputed=x["tfs"],
                                             orderbook_row=book_map[x["market"]], orderbook_headers=book_headers)
                fresh = pack(a["ticker"], x["daily_pattern"])
                b = _score_market(a, a["ticker"], "day")
                pattern_score = x["daily_pattern"]["score"]
                score = round(0.6 * pattern_score + 0.4 * b["total_score"], 3)
                b.update({"technical_only_score": b["total_score"], "pattern_score": pattern_score,
                          "total_score": score, "mode_label": "일봉 재상승", "tags": fresh["risk_flags"] + b["tags"]})
                return {**fresh, "early_score": x["early_score"], "technical_screen_score": score,
                        "score_breakdown": b, "analysis": a}
            except Exception as exc:
                errors.append({"market": x["market"], "stage": "full", "error": str(exc)})
                return None
    valid = [x for x in await asyncio.gather(*(full(x) for x in finalists)) if x]
    # Refresh the whole ticker snapshot after the scan, so day-stage prices are
    # never presented as current after a potentially long first scan.
    latest, _ = await _get("/v1/ticker/all", {"quote_currencies": "KRW"})
    fresh_map = {r["market"]: r for r in latest}
    def refresh(x: dict[str, Any]) -> dict[str, Any]:
        row = fresh_map.get(x["market"])
        if row is None:
            return {**x, "entry_eligible": False, "risk_flags": x["risk_flags"] + ["현재가 조회 실패"]}
        return {**x, **pack(row, x["daily_pattern"])}
    patterns = [refresh(x) for x in patterns]
    # Every >18% mover remains visible, even if it did not match this pattern.
    pattern_map = {i["row"]["market"]: i["pattern"] for i in checked}
    movers = [pack(r, pattern_map.get(r["market"], {"matched": False, "reason": "daily_failed"}))
              for r in latest if r["market"].startswith("KRW-") and float(r.get("signed_change_rate") or 0) > 0.18]
    valid = [refresh(x) for x in valid]
    valid.sort(key=lambda x: x["technical_screen_score"], reverse=True)
    patterns.sort(key=lambda x: x["daily_pattern"]["score"], reverse=True)
    day_changed = as_of.date() != datetime.now(timezone.utc).date()
    if day_changed:
        for x in valid + patterns:
            x["entry_eligible"] = False
            x["risk_flags"].append("일봉 마감 경계 통과·재조회 필요")
    return {"version": "3.1-daily-first", "received_at_utc": _utc_now_iso(), "scan_started_at_utc": as_of.isoformat(),
            "mode": "day", "mode_label": "일봉 재상승", "method": "all KRW closed daily setup -> 4h/1h -> finalists",
            "important": "Setup scores are not probabilities. Watch is not an entry signal. Live signals can reverse.",
            "ticker_rows": len(rows), "daily_checked_rows": len(checked), "daily_match_count": len(patterns),
            "eligible_rows": sum(x["entry_eligible"] for x in patterns), "shortlist_size": len(shortlist),
            "full_analysis_count": len(valid), "elapsed_seconds": round(time.monotonic() - started, 3),
            "remaining_req": headers.get("remaining-req"), "errors": errors, "excluded": excluded,
            "top_candidates": [x for x in valid if x["entry_eligible"]][:top_n],
            "shortlist_results": valid, "daily_candidates": patterns, "extended_movers": movers,
            "overheated_markets": [x["market"] for x in movers],
            "closed_daily_cache": "until next 09:00 KST; in-memory", "live_daily_cache_ttl_seconds": 60,
            "day_boundary_crossed": day_changed}


def _matches_macd_expansion(t: dict[str, Any], rising_15m: bool = False) -> bool:
    """Golden state means MACD strictly above signal, including an unfinished candle."""
    if t.get("error") or t.get("macd") is None or t.get("macd_signal") is None:
        return False
    if t["macd"] <= t["macd_signal"]:
        return False
    if rising_15m:
        return (t.get("close") is not None and t.get("open") is not None
                and t["close"] > t["open"]
                and t.get("bollinger_width") is not None
                and t.get("bollinger_width_prev") is not None
                and t["bollinger_width"] > t["bollinger_width_prev"])
    return True


@mcp.tool()
async def scan_macd_bb_expansion() -> dict[str, Any]:
    """Check every KRW pair: golden MACD state on 15m/1h/4h/day, rising 15m candle and widening 15m Bollinger width."""
    tickers, _ = await _get("/v1/ticker/all", {"quote_currencies": "KRW"})
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    async def paced_tf(market: str, tf: str) -> dict[str, Any]:
        return await _analyze_tf(market, tf)

    async def check(row: dict[str, Any]) -> None:
        market = row["market"]
        try:
            tfs = {"15m": await paced_tf(market, "15m")}
            if not _matches_macd_expansion(tfs["15m"], rising_15m=True):
                return
            for tf in ("60m", "240m", "1d"):
                tfs[tf] = await paced_tf(market, tf)
                if not _matches_macd_expansion(tfs[tf]):
                    return
            results.append({
                "market": market, "trade_price": row.get("trade_price"),
                "signed_change_rate": row.get("signed_change_rate"),
                "ticker_source_age_seconds": _ms_age_seconds(row.get("timestamp")),
                "timeframes": {tf: {
                    "macd": tfs[tf]["macd"], "signal": tfs[tf]["macd_signal"],
                    "candle_kst": tfs[tf]["latest_candle_kst"],
                    "bb_width": tfs[tf]["bollinger_width"] if tf == "15m" else None,
                    "bb_width_prev": tfs[tf]["bollinger_width_prev"] if tf == "15m" else None,
                } for tf in ("15m", "60m", "240m", "1d")},
            })
        except Exception as exc:
            errors.append({"market": market, "error": f"{type(exc).__name__}: {exc}"})

    semaphore = asyncio.Semaphore(3)

    async def bounded(row: dict[str, Any]) -> None:
        async with semaphore:
            await check(row)

    await asyncio.gather(*(bounded(row) for row in tickers))
    results.sort(key=lambda x: float(x.get("signed_change_rate") or 0), reverse=True)
    return {
        "received_at_utc": _utc_now_iso(), "ticker_rows": len(tickers),
        "checked_rows": len(tickers) - len(errors), "errors": errors,
        "match_count": len(results), "matches": results,
        "criteria": "MACD > signal on 15m/1h/4h/day; current 15m close > open; normalized BB width > preceding 15m width",
    }


MOBILE_HTML = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover" />
  <meta name="theme-color" content="#0b0d12" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
  <title>Upbit Full Check v3</title>
  <style>
    :root{--bg:#0b0d12;--card:#151923;--line:#262c3a;--txt:#f4f7fb;--muted:#9ba7ba;--accent:#5da8ff;--swing:#73e2a7;--good:#35d07f;--bad:#ff6472;--warn:#ffbf5f}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif}
    .wrap{max-width:860px;margin:auto;padding:20px 14px 80px}.head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin:8px 0 18px}
    h1{font-size:24px;margin:0 0 5px}.sub{font-size:13px;color:var(--muted);line-height:1.5}.badge{border:1px solid var(--line);padding:7px 10px;border-radius:999px;font-size:12px;color:var(--muted);white-space:nowrap}
    .panel{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:15px;margin:12px 0}.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}
    button{appearance:none;border:0;border-radius:13px;padding:13px 15px;font-weight:850;font-size:14px;background:var(--accent);color:#05101d;cursor:pointer}button.swing{background:var(--swing);color:#062116}button.secondary{background:#232a38;color:var(--txt);border:1px solid var(--line)}button:disabled{opacity:.55}
    input{flex:1;min-width:160px;background:#0f131b;border:1px solid var(--line);border-radius:12px;padding:13px 14px;color:var(--txt);font-size:15px;text-transform:uppercase}
    .status{font-size:13px;color:var(--muted);margin-top:10px;min-height:20px}.status.good{color:var(--good)}.status.bad{color:var(--bad)}
    .cards{display:grid;grid-template-columns:1fr;gap:10px}.coin{border:1px solid var(--line);border-radius:16px;padding:14px;background:#10141c}
    .coinTop{display:flex;justify-content:space-between;gap:8px;align-items:center}.rank{font-size:13px;color:var(--muted)}.market{font-size:19px;font-weight:900}.price{font-size:18px;font-weight:800;text-align:right}
    .chg.up{color:var(--good)}.chg.down{color:var(--bad)}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:12px}.metric{background:#0b0f16;border:1px solid #202634;border-radius:12px;padding:10px}.k{font-size:11px;color:var(--muted)}.v{font-size:14px;font-weight:800;margin-top:3px;word-break:break-word}
    .scores{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:11px}.score{background:#0d121a;border:1px solid #202634;border-radius:10px;padding:8px}.score .v{font-size:13px}.pen .v{color:var(--warn)}
    .tags{margin-top:9px}.tag{display:inline-block;padding:5px 8px;border-radius:999px;background:#1d2634;color:#d7e3f4;font-size:11px;margin:2px 4px 2px 0}.tag.warn{background:#35291a;color:#ffd18a}
    .tf{margin-top:11px;padding-top:11px;border-top:1px solid var(--line)}.tfline{font-size:12px;color:#c6cfdd;line-height:1.8}.spinner{display:inline-block;width:14px;height:14px;border:2px solid #ffffff33;border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;vertical-align:-2px;margin-right:6px}@keyframes spin{to{transform:rotate(360deg)}}
    .foot{margin-top:16px;color:var(--muted);font-size:11px;line-height:1.6}.pill{display:inline-block;padding:4px 7px;border-radius:999px;background:#202736;color:#cdd8e8;font-size:11px;margin:2px}.modehint{font-size:11px;color:var(--muted);margin-top:8px;line-height:1.55}@media(min-width:680px){.cards{grid-template-columns:1fr 1fr}.grid{grid-template-columns:repeat(4,1fr)}}
  </style>
</head>
<body><div class="wrap">
  <div class="head"><div><h1>업비트 풀체크 v3</h1><div class="sub">V3.1 · 전체 일봉 선별 · 조정 후 재상승 · 급등 종목 별도 표시</div></div><div class="badge" id="clock">-</div></div>
  <div class="panel"><div class="row"><button id="dayBtn" onclick="runScan('day')">일봉 재상승 TOP 5</button><button id="swingBtn" class="swing" onclick="runScan('swing5d')">5일 스윙 TOP 5</button><button id="macdBtn" onclick="runMacdScan()">4개 봉 MACD + 15분 볼밴 확장</button><button class="secondary" onclick="checkHealth()">연결 확인</button></div><div class="modehint">전체 원화 종목의 마감된 일봉에서 1차 상승·조정·상승 추세 유지를 찾습니다. 관찰 후보는 매수 신호가 아닙니다. 4시간·1시간 확인 후 상위 후보를 표시하며, 이미 급등한 종목도 별도로 남깁니다. 첫 전체 조회에는 시간이 걸릴 수 있습니다.</div><div id="scanStatus" class="status">원하는 모드를 누르세요. 첫 호출은 Render 무료 서버가 깨어나느라 오래 걸릴 수 있습니다.</div></div>
  <div class="panel"><div class="row"><input id="market" value="KRW-QUID" placeholder="예: KRW-BTC"/><button class="secondary" onclick="analyzeOne()">현재 모드로 종목 분석</button></div><div id="oneStatus" class="status"></div></div>
  <div id="results" class="cards"></div>
  <div class="foot">점수는 급등 확률이 아닙니다. 일봉 구조 60%와 기술 점수 40%를 합산합니다. 구조는 마감봉, 재상승 신호는 현재가 기준이며 변할 수 있습니다. 두 사례를 확인했으며 전체 시장 예측 성능은 아직 검증하지 않았습니다. 최근 급등 리스크와 드라이브 성향은 3·5·7일 수익률, 7일 런업/고점대비 조정, RSI·볼린저·거래량·호가·MACD 변화로 계산합니다. 현재가·호가·캔들 source age가 오래되면 감점됩니다.</div>
</div>
<script>
const $=id=>document.getElementById(id);let currentMode='day';
const fmt=n=>{if(n===null||n===undefined||Number.isNaN(Number(n)))return '-';const x=Number(n);return new Intl.NumberFormat('ko-KR',{maximumFractionDigits:x<10?6:x<100?3:0}).format(x)};
const won=n=>{if(!n)return '-';const x=Number(n);if(x>=1e12)return(x/1e12).toFixed(2)+'조';if(x>=1e8)return(x/1e8).toFixed(1)+'억';if(x>=1e4)return(x/1e4).toFixed(1)+'만';return fmt(x)};
const pct=n=>n===null||n===undefined?'-':(Number(n)*100).toFixed(2)+'%';
const age=n=>n===null||n===undefined?'-':(Number(n)<5?Number(n).toFixed(1)+'초':Number(n)<60?Math.round(n)+'초':(Number(n)/60).toFixed(1)+'분');
function tfSummary(t){if(!t||t.error)return'데이터 부족';const d=t.macd_histogram_delta;return`RSI ${fmt(t.rsi14)} · MACD H ${fmt(t.macd_histogram)}${d==null?'':(' Δ'+fmt(d))} · EMA20${t.ema20_above_ema60?'>':'<'}60 · BB%B ${fmt(t.bollinger_percent_b)} · Vol× ${fmt(t.volume_ratio20)} · age ${age(t.latest_source_age_seconds)}`}
function scoreGrid(b){if(!b)return'';return`<div class="scores"><div class="score"><div class="k">추세</div><div class="v">${fmt(b.trend_score)}</div></div><div class="score"><div class="k">거래량</div><div class="v">${fmt(b.volume_score)}</div></div><div class="score"><div class="k">진입</div><div class="v">${fmt(b.entry_score)}</div></div><div class="score"><div class="k">호가</div><div class="v">${fmt(b.orderbook_score)}</div></div><div class="score"><div class="k">유동성</div><div class="v">${fmt(b.liquidity_score)}</div></div><div class="score pen"><div class="k">과열 감점</div><div class="v">-${fmt(b.heat_penalty)}</div></div><div class="score pen"><div class="k">최근급등 리스크</div><div class="v">${fmt(b.surge_risk)}/100</div></div><div class="score"><div class="k">드라이브 성향</div><div class="v">${b.drive_profile||'-'}</div></div><div class="score"><div class="k">드라이브 보정</div><div class="v">${Number(b.drive_bonus||0)>=0?'+':''}${fmt(b.drive_bonus)}</div></div></div>`}
function tagHtml(tags){return(tags||[]).length?`<div class="tags">${tags.map(t=>`<span class="tag ${t.includes('주의')||t.includes('매도')?'warn':''}">${t}</span>`).join('')}</div>`:''}
function patternHtml(x){const p=x.daily_pattern;if(!p)return'';return `<div class="tf"><div class="tfline"><b>${x.state_label||'일봉 패턴'}</b> · 일봉 구조 ${fmt(p.score)}점</div><div class="tfline">${p.setup_as_of_kst||'-'} 봉까지 · ${fmt(p.pullback_days)}일 조정 · 고점 대비 -${fmt(p.pullback_pct)}%</div><div class="tfline">조정 거래량 / 상승 구간 ${fmt(p.volume_dry_ratio)}배 · 20일선 변화 ${fmt(p.ma20_slope_3d_pct)}%</div><div class="tfline">재상승 확인선 ${fmt(p.trigger)} · 조정 저점 ${fmt(p.support)} · 이전 고점 ${fmt(p.prior_peak)}</div>${tagHtml(x.risk_flags)}</div>`}
function compactPattern(x){const p=x.daily_pattern||{};return `<div class="coin"><div class="coinTop"><div><b>${x.market}</b><div class="rank">${x.state_label}</div></div><div>₩ ${fmt(x.trade_price)}<div class="chg ${x.signed_change_rate>=0?'up':'down'}">${pct(x.signed_change_rate)}</div></div></div>${p.matched?patternHtml(x):tagHtml(x.risk_flags)}<div class="tfline">24H ${won(x.acc_trade_price_24h)} · 마지막 체결 ${age(x.last_trade_age_seconds)} 전</div></div>`}
function card(x,i){const a=x.analysis||x,t=a.ticker||{},m=x.market||a.market||t.market||'-',ch=Number(x.signed_change_rate??t.signed_change_rate??0),cls=ch>=0?'up':'down',ob=a.orderbook_summary||{},tfs=a.timeframes||{},b=x.score_breakdown||x.score||{},label=b.mode_label||'종목 분석',total=b.total_score??x.technical_screen_score;return`<div class="coin"><div class="coinTop"><div><div class="rank">${i?('#'+i):'종목 분석'} · ${label} ${fmt(total)}점</div><div class="market">${m}</div></div><div><div class="price">₩ ${fmt(x.trade_price??t.trade_price)}</div><div class="chg ${cls}">${pct(ch)}</div></div></div>${patternHtml(x)}${tagHtml(b.tags)}${scoreGrid(b)}<div class="grid"><div class="metric"><div class="k">24H 거래대금</div><div class="v">${won(x.acc_trade_price_24h??t.acc_trade_price_24h)}</div></div><div class="metric"><div class="k">Ticker age</div><div class="v">${age(x.ticker_source_age_seconds??t.source_age_seconds)}</div></div><div class="metric"><div class="k">Best bid / ask</div><div class="v">${fmt(ob.best_bid)} / ${fmt(ob.best_ask)}</div></div><div class="metric"><div class="k">Top10 호가 imbalance</div><div class="v">${ob.top10_imbalance==null?'-':(Number(ob.top10_imbalance)*100).toFixed(1)+'%'}</div></div></div>${b.recent_surge?`<div class="tf"><div class="tfline"><span class="pill">최근급등</span> 3D ${fmt(b.recent_surge.recent_return_3d_pct)}% · 5D ${fmt(b.recent_surge.recent_return_5d_pct)}% · 7D ${fmt(b.recent_surge.recent_return_7d_pct)}% · 7D저점대비 ${fmt(b.recent_surge.runup_7d_pct)}% · 7D고점대비 ${fmt(b.recent_surge.drawdown_from_7d_high_pct)}%</div></div>`:''}<div class="tf"><div class="tfline"><span class="pill">5m</span> ${tfSummary(tfs['5m'])}</div><div class="tfline"><span class="pill">15m</span> ${tfSummary(tfs['15m'])}</div><div class="tfline"><span class="pill">1h</span> ${tfSummary(tfs['60m'])}</div><div class="tfline"><span class="pill">4h</span> ${tfSummary(tfs['240m'])}</div><div class="tfline"><span class="pill">1d</span> ${tfSummary(tfs['1d'])}</div></div></div>`}
async function fetchJSON(url,opt){const r=await fetch(url,opt),tx=await r.text();let j;try{j=JSON.parse(tx)}catch(e){throw new Error(`HTTP ${r.status}: ${tx.slice(0,180)}`)}if(!r.ok)throw new Error(j.error||`HTTP ${r.status}`);return j}
async function checkHealth(){const s=$('scanStatus');s.className='status';s.innerHTML='<span class="spinner"></span>업비트 연결 확인 중...';try{const j=await fetchJSON('/api/health');s.className='status good';s.textContent=`정상 · KRW-BTC ${fmt(j.sample?.trade_price)}원 · 데이터 age ${age(j.source_age_seconds)}`;}catch(e){s.className='status bad';s.textContent='실패: '+e.message}}
async function runScan(mode){currentMode=mode;const db=$('dayBtn'),sb=$('swingBtn'),mb=$('macdBtn'),s=$('scanStatus'),r=$('results');db.disabled=sb.disabled=mb.disabled=true;s.className='status';s.innerHTML='<span class="spinner"></span>'+(mode==='day'?'전체 일봉 확인 → 조정·재상승 후보 선별 → 4시간·1시간 확인 중...':'5일 스윙 후보 확인 중...');r.innerHTML='';try{const j=await fetchJSON('/api/scan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,top_n:5,shortlist_size:24,min_turnover_krw:1000000000})});r.innerHTML=(j.top_candidates||[]).map((x,i)=>card(x,i+1)).join('')||'<div class="coin">현재 진입 검토 기준에 맞는 정밀 후보가 없습니다.</div>';if(j.daily_candidates){r.innerHTML+='<div class="panel" style="grid-column:1/-1"><b>일봉 패턴 전체 '+j.daily_candidates.length+'개</b><div class="sub">정밀분석 여부와 관계없이 모두 표시 · 거래대금 부족·유의·지지 이탈 포함</div></div>'+j.daily_candidates.map(compactPattern).join('');r.innerHTML+='<div class="panel" style="grid-column:1/-1"><b>이미 급등한 종목 '+j.extended_movers.length+'개</b><div class="sub">발견 목록 · 신규 매수 추천이 아닙니다</div></div>'+j.extended_movers.map(compactPattern).join('');}const errs=j.errors||[];s.className=errs.length||j.day_boundary_crossed?'status bad':'status good';s.textContent=j.mode_label+' · 전체 '+j.ticker_rows+'개 / 일봉 확인 '+(j.daily_checked_rows??'-')+'개 / 패턴 '+(j.daily_match_count??'-')+'개 / 정밀 '+j.full_analysis_count+'개 / 오류 '+errs.length+'건 · '+new Date(j.received_at_utc).toLocaleString('ko-KR',{timeZone:'Asia/Seoul'})+' 한국시간';if(errs.length)s.textContent+=' · 미완료 종목: '+errs.map(e=>e.market).join(', ');if(j.day_boundary_crossed)s.textContent+=' · 오전 9시 일봉 전환: 재조회 필요';}catch(e){s.className='status bad';s.textContent='실패: '+e.message}finally{db.disabled=sb.disabled=mb.disabled=false}}
async function runMacdScan(){const btn=$('macdBtn'),s=$('scanStatus'),r=$('results');btn.disabled=$('dayBtn').disabled=$('swingBtn').disabled=true;s.className='status';s.innerHTML='<span class="spinner"></span>원화마켓 전체 15분봉 검사 및 조건 후보의 1시간·4시간·일봉 확인 중...';r.innerHTML='';const st=Date.now();try{const j=await fetchJSON('/api/scan/macd-bb',{method:'POST'});r.innerHTML=(j.matches||[]).map(x=>`<div class="coin"><div class="coinTop"><div class="market">${x.market}</div><div><div class="price">₩ ${fmt(x.trade_price)}</div><div class="chg ${Number(x.signed_change_rate)>=0?'up':'down'}">${pct(x.signed_change_rate)}</div></div></div><div class="tf">${[['15m','15분'],['60m','1시간'],['240m','4시간'],['1d','1일']].map(([k,n])=>`<div class="tfline"><span class="pill">${n}</span> MACD ${fmt(x.timeframes[k].macd)} &gt; Signal ${fmt(x.timeframes[k].signal)}</div>`).join('')}<div class="tfline">15분 볼밴 폭 ${fmt(x.timeframes['15m'].bb_width_prev)} → ${fmt(x.timeframes['15m'].bb_width)}</div></div></div>`).join('')||'<div class="coin">조건에 맞는 코인이 없습니다.</div>';s.className='status good';s.textContent=`전체 ${j.ticker_rows}개 중 ${j.match_count}개 일치 · 조회 실패 ${j.errors.length}개 · ${((Date.now()-st)/1000).toFixed(1)}초`;}catch(e){s.className='status bad';s.textContent='실패: '+e.message}finally{btn.disabled=$('dayBtn').disabled=$('swingBtn').disabled=false}}
async function analyzeOne(){const m=$('market').value.trim().toUpperCase(),s=$('oneStatus'),r=$('results');if(!m)return;s.className='status';s.innerHTML='<span class="spinner"></span>'+m+' 분석 중...';try{const j=await fetchJSON('/api/analyze?market='+encodeURIComponent(m)+'&mode='+encodeURIComponent(currentMode));r.innerHTML=card(j,0);s.className='status good';s.textContent='완료 · '+(j.score_breakdown?.mode_label||'공식 Upbit Public API');}catch(e){s.className='status bad';s.textContent='실패: '+e.message}}
setInterval(()=>{$('clock').textContent=new Date().toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit',second:'2-digit'})},1000);checkHealth();
</script></body></html>'''


@mcp.custom_route("/", methods=["GET"])
async def mobile_home(request: Request) -> Response:
    return HTMLResponse(MOBILE_HTML, headers={"Cache-Control": "no-store"})


@mcp.custom_route("/api/health", methods=["GET"])
async def api_health(request: Request) -> Response:
    try:
        return JSONResponse(await health_check(), headers={"Cache-Control": "no-store"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=502)


@mcp.custom_route("/api/analyze", methods=["GET"])
async def api_analyze(request: Request) -> Response:
    market = (request.query_params.get("market") or "").strip().upper()
    if not market:
        return JSONResponse({"error": "market is required, e.g. KRW-BTC"}, status_code=400)
    mode = (request.query_params.get("mode") or "day").strip().lower()
    if mode not in {"day", "swing5d"}:
        mode = "day"
    try:
        data = await _analyze_market_full(market)
        ticker_row = data.get("ticker", {}) if isinstance(data, dict) else {}
        score = _score_market(data, ticker_row, mode)
        data["score_breakdown"] = score
        data["technical_screen_score"] = score["total_score"]
        return JSONResponse(data, headers={"Cache-Control": "no-store"})
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@mcp.custom_route("/api/tickers", methods=["GET"])
async def api_tickers(request: Request) -> Response:
    quote = (request.query_params.get("quote") or "KRW").strip().upper()
    try:
        return JSONResponse(await get_all_tickers(quote), headers={"Cache-Control": "no-store"})
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@mcp.custom_route("/api/scan", methods=["POST"])
async def api_scan(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        data = await scan_krw_market(
            top_n=int(body.get("top_n", 5)),
            shortlist_size=int(body.get("shortlist_size", 24)),
            min_turnover_krw=float(body.get("min_turnover_krw", 1_000_000_000)),
            mode=str(body.get("mode", "day")),
        )
        return JSONResponse(data, headers={"Cache-Control": "no-store"})
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@mcp.custom_route("/api/scan/macd-bb", methods=["POST"])
async def api_scan_macd_bb(request: Request) -> Response:
    try:
        return JSONResponse(await scan_macd_bb_expansion(), headers={"Cache-Control": "no-store"})
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("MCP_PORT", "8000")))
    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
    )
