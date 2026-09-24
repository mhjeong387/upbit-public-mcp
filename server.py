from __future__ import annotations

import asyncio
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Iterable

import httpx
from mcp.server import MCPServer

BASE_URL = "https://api.upbit.com"
TIMEOUT = httpx.Timeout(15.0, connect=8.0)
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "UpbitPublicMCP/1.0",
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


async def _get(path: str, params: dict[str, Any] | None = None, retries: int = 3) -> tuple[Any, dict[str, str]]:
    url = f"{BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
        for attempt in range(retries + 1):
            resp = await client.get(url, params=params)
            if resp.status_code == 429 and attempt < retries:
                await asyncio.sleep(0.35 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json(), dict(resp.headers)
    raise RuntimeError("unreachable")


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
    raw, headers = await _get(path, {"market": market, "count": max(60, min(count, 200))})
    if len(raw) < 35:
        return {"timeframe": timeframe, "error": "insufficient candles", "count": len(raw)}
    candles = list(reversed(raw))  # ascending
    close = [float(c["trade_price"]) for c in candles]
    volumes = [float(c.get("candle_acc_trade_volume", 0) or 0) for c in candles]
    quote_values = [float(c.get("candle_acc_trade_price", 0) or 0) for c in candles]

    rsi14 = _rsi(close, 14)
    ema20 = _ema(close, 20)
    ema60 = _ema(close, 60)
    macd = _macd(close)
    bb = _bollinger(close, 20, 2.0)
    atr14 = _atr(candles, 14)

    last = len(close) - 1
    v20 = _mean(volumes[max(0, last - 19) : last + 1])
    q20 = _mean(quote_values[max(0, last - 19) : last + 1])
    bb_width = None
    bb_pctb = None
    if bb["upper"][last] is not None and bb["lower"][last] is not None and bb["middle"][last] is not None:
        up = float(bb["upper"][last])
        lo = float(bb["lower"][last])
        mid = float(bb["middle"][last])
        if mid:
            bb_width = (up - lo) / mid
        if up != lo:
            bb_pctb = (close[last] - lo) / (up - lo)

    latest = raw[0]
    return {
        "timeframe": timeframe,
        "candle_count": len(raw),
        "latest_candle_kst": latest.get("candle_date_time_kst"),
        "latest_source_age_seconds": _ms_age_seconds(latest.get("timestamp")),
        "close": close[last],
        "rsi14": rsi14[last],
        "ema20": ema20[last],
        "ema60": ema60[last],
        "ema20_above_ema60": (ema20[last] is not None and ema60[last] is not None and ema20[last] > ema60[last]),
        "macd": macd["macd"][last],
        "macd_signal": macd["signal"][last],
        "macd_histogram": macd["histogram"][last],
        "bollinger_upper": bb["upper"][last],
        "bollinger_middle": bb["middle"][last],
        "bollinger_lower": bb["lower"][last],
        "bollinger_percent_b": bb_pctb,
        "bollinger_width": bb_width,
        "atr14": atr14[last],
        "atr14_pct": (atr14[last] / close[last]) if atr14[last] is not None and close[last] else None,
        "volume": volumes[last],
        "volume_sma20": v20,
        "volume_ratio20": (volumes[last] / v20) if v20 else None,
        "quote_volume": quote_values[last],
        "quote_volume_sma20": q20,
        "remaining_req": headers.get("remaining-req"),
    }


async def _analyze_market_full(market: str) -> dict[str, Any]:
    ticker_task = _get("/v1/ticker", {"markets": market})
    orderbook_task = _get("/v1/orderbook", {"markets": market})
    # Stay below Upbit Candle group's 10 req/s limit.
    tf_results: dict[str, Any] = {}
    for tf in ["5m", "15m", "60m", "240m", "1d"]:
        tf_results[tf] = await _analyze_tf(market, tf)
        await asyncio.sleep(0.12)
    (ticker, ticker_h), (orderbook, ob_h) = await asyncio.gather(ticker_task, orderbook_task)

    ticker_row = ticker[0] if ticker else {}
    ob_row = orderbook[0] if orderbook else {}
    units = ob_row.get("orderbook_units") or []
    top10_bid = sum(float(u.get("bid_size", 0) or 0) for u in units[:10])
    top10_ask = sum(float(u.get("ask_size", 0) or 0) for u in units[:10])
    denom = top10_bid + top10_ask

    return {
        "market": market,
        "received_at_utc": _utc_now_iso(),
        "ticker": {
            **ticker_row,
            "source_age_seconds": _ms_age_seconds(ticker_row.get("timestamp")),
            "remaining_req": ticker_h.get("remaining-req"),
        },
        "orderbook_summary": {
            "source_age_seconds": _ms_age_seconds(ob_row.get("timestamp")),
            "best_ask": units[0].get("ask_price") if units else None,
            "best_bid": units[0].get("bid_price") if units else None,
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


def _ticker_prescore(row: dict[str, Any]) -> float:
    """Liquidity + momentum prescreen only; not a return forecast."""
    change = float(row.get("signed_change_rate", 0) or 0)
    turnover = float(row.get("acc_trade_price_24h", 0) or 0)
    high = float(row.get("high_price", 0) or 0)
    low = float(row.get("low_price", 0) or 0)
    price = float(row.get("trade_price", 0) or 0)
    day_range = ((high - low) / price) if price else 0.0
    liquidity = math.log10(max(turnover, 1.0))
    # Prefer positive momentum with liquidity, but avoid making this a probability claim.
    return (change * 100.0 * 1.7) + (liquidity * 0.8) + (day_range * 100.0 * 0.15)


def _technical_score(analysis: dict[str, Any]) -> float:
    """Heuristic technical screen score; not a predicted probability of future return."""
    score = 0.0
    for tf, weight in [("5m", 1.0), ("15m", 1.2), ("60m", 1.6), ("240m", 1.8), ("1d", 1.2)]:
        x = analysis.get("timeframes", {}).get(tf, {})
        if not isinstance(x, dict) or x.get("error"):
            continue
        rsi = x.get("rsi14")
        hist = x.get("macd_histogram")
        volr = x.get("volume_ratio20")
        pctb = x.get("bollinger_percent_b")
        if x.get("ema20_above_ema60"):
            score += 1.4 * weight
        if hist is not None and hist > 0:
            score += 1.2 * weight
        if rsi is not None:
            if 52 <= rsi <= 68:
                score += 1.4 * weight
            elif 45 <= rsi < 52:
                score += 0.5 * weight
            elif rsi >= 78:
                score -= 1.2 * weight
        if volr is not None:
            if 1.3 <= volr <= 4.0:
                score += 0.9 * weight
            elif volr > 7.0:
                score -= 0.3 * weight
        if pctb is not None and pctb > 1.15:
            score -= 0.8 * weight
    ob = analysis.get("orderbook_summary", {})
    imb = ob.get("top10_imbalance") if isinstance(ob, dict) else None
    if imb is not None:
        score += max(-1.0, min(1.0, float(imb) * 2.0))
    return score


@mcp.tool()
async def scan_krw_market(
    top_n: int = 5,
    shortlist_size: int = 12,
    min_turnover_krw: float = 1_000_000_000,
) -> dict[str, Any]:
    """Full KRW-market technical screen using /ticker/all, then multi-timeframe candles and orderbook for a shortlist. Returns a heuristic ranking, not a guaranteed-return forecast."""
    top_n = max(1, min(int(top_n), 10))
    shortlist_size = max(top_n, min(int(shortlist_size), 20))
    tickers, headers = await _get("/v1/ticker/all", {"quote_currencies": "KRW"})

    fresh_rows = []
    for row in tickers:
        if float(row.get("acc_trade_price_24h", 0) or 0) < float(min_turnover_krw):
            continue
        row = dict(row)
        row["source_age_seconds"] = _ms_age_seconds(row.get("timestamp"))
        row["prescore"] = _ticker_prescore(row)
        fresh_rows.append(row)

    fresh_rows.sort(key=lambda x: x["prescore"], reverse=True)
    shortlist = fresh_rows[:shortlist_size]

    analyzed: list[dict[str, Any]] = []
    # Sequential analysis intentionally respects the shared candle rate-limit group.
    for row in shortlist:
        market = row["market"]
        try:
            a = await _analyze_market_full(market)
            tech_score = _technical_score(a)
            analyzed.append({
                "market": market,
                "trade_price": row.get("trade_price"),
                "signed_change_rate": row.get("signed_change_rate"),
                "acc_trade_price_24h": row.get("acc_trade_price_24h"),
                "ticker_source_age_seconds": row.get("source_age_seconds"),
                "prescore": row.get("prescore"),
                "technical_screen_score": tech_score,
                "analysis": a,
            })
        except Exception as e:
            analyzed.append({"market": market, "error": f"{type(e).__name__}: {e}"})
        await asyncio.sleep(0.15)

    valid = [x for x in analyzed if "error" not in x]
    valid.sort(key=lambda x: x["technical_screen_score"], reverse=True)

    return {
        "received_at_utc": _utc_now_iso(),
        "method": "KRW /ticker/all liquidity+momentum prescreen -> 5m/15m/60m/240m/1d technical analysis -> orderbook summary",
        "important": "technical_screen_score is a deterministic heuristic, not a probability or promise of future gains",
        "ticker_rows": len(tickers),
        "eligible_rows": len(fresh_rows),
        "shortlist_size": len(shortlist),
        "remaining_req": headers.get("remaining-req"),
        "top_candidates": valid[:top_n],
        "shortlist_results": analyzed,
    }


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
