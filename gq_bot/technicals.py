"""Pure technical and risk maths over daily bars (oldest first). No I/O.

A bar is {"date": "YYYY-MM-DD", "close": float, and optionally open/high/low/volume}.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

TRADING_DAYS = 252


def _closes(bars) -> List[float]:
    return [float(b["close"]) for b in bars if b.get("close") is not None]


def sma(xs: Sequence[float], n: int) -> Optional[float]:
    return sum(xs[-n:]) / n if len(xs) >= n else None


def sma_series(xs: Sequence[float], n: int) -> List[Optional[float]]:
    out, run = [], 0.0
    for i, x in enumerate(xs):
        run += x
        if i >= n:
            run -= xs[i - n]
        out.append(run / n if i >= n - 1 else None)
    return out


def ema_series(xs: Sequence[float], n: int) -> List[float]:
    if not xs:
        return []
    k, out = 2 / (n + 1), [xs[0]]
    for x in xs[1:]:
        out.append(x * k + out[-1] * (1 - k))
    return out


def rsi(xs: Sequence[float], n: int = 14) -> Optional[float]:
    """Wilder's RSI."""
    if len(xs) <= n:
        return None
    gains = [max(xs[i] - xs[i - 1], 0) for i in range(1, len(xs))]
    losses = [max(xs[i - 1] - xs[i], 0) for i in range(1, len(xs))]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n
    for g, l in zip(gains[n:], losses[n:]):
        ag, al = (ag * (n - 1) + g) / n, (al * (n - 1) + l) / n
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 2)


def rsi_zone(r: Optional[float]) -> Optional[str]:
    if r is None:
        return None
    return "overbought (>70)" if r > 70 else "oversold (<30)" if r < 30 else "neutral (30-70)"


def macd(xs: Sequence[float]) -> Optional[Dict[str, Any]]:
    if len(xs) < 35:
        return None
    line = [a - b for a, b in zip(ema_series(xs, 12), ema_series(xs, 26))]
    sig = ema_series(line, 9)
    hist = [l - s for l, s in zip(line, sig)]
    last_cross = None
    for i in range(len(hist) - 1, 0, -1):
        if (hist[i] > 0) != (hist[i - 1] > 0):
            last_cross = {"sessions_ago": len(hist) - 1 - i,
                          "type": "bullish (MACD crossed above signal)" if hist[i] > 0
                          else "bearish (MACD crossed below signal)"}
            break
    return {"macd": round(line[-1], 4), "signal": round(sig[-1], 4), "histogram": round(hist[-1], 4),
            "state": "above signal" if hist[-1] > 0 else "below signal", "last_crossover": last_cross}


def daily_returns(xs: Sequence[float]) -> List[float]:
    return [xs[i] / xs[i - 1] - 1 for i in range(1, len(xs)) if xs[i - 1]]


def stdev(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ann_vol(xs: Sequence[float], n: Optional[int] = None) -> Optional[float]:
    r = daily_returns(xs[-(n + 1):] if n else xs)
    s = stdev(r)
    return round(s * math.sqrt(TRADING_DAYS) * 100, 2) if s is not None else None


def max_drawdown(xs: Sequence[float]) -> Optional[Dict[str, Any]]:
    if len(xs) < 2:
        return None
    peak, peak_i, worst, w = xs[0], 0, 0.0, (0, 0)
    for i, x in enumerate(xs):
        if x > peak:
            peak, peak_i = x, i
        dd = x / peak - 1
        if dd < worst:
            worst, w = dd, (peak_i, i)
    return {"pct": round(worst * 100, 2), "peak_index": w[0], "trough_index": w[1]}


def atr_pct(bars, n: int = 14) -> Optional[float]:
    if len(bars) <= n or any(b.get("high") is None or b.get("low") is None for b in bars[-n - 1:]):
        return None
    trs = []
    for prev, b in zip(bars[-n - 1:-1], bars[-n:]):
        pc = float(prev["close"])
        trs.append(max(float(b["high"]) - float(b["low"]), abs(float(b["high"]) - pc), abs(float(b["low"]) - pc)))
    return round(sum(trs) / n / float(bars[-1]["close"]) * 100, 2)


def _last_ma_cross(xs: Sequence[float], dates: Sequence[str]) -> Optional[Dict[str, Any]]:
    s50, s200 = sma_series(xs, 50), sma_series(xs, 200)
    for i in range(len(xs) - 1, 0, -1):
        a, b, pa, pb = s50[i], s200[i], s50[i - 1], s200[i - 1]
        if None in (a, b, pa, pb):
            break
        if (a > b) != (pa > pb):
            return {"type": "golden cross (50-day moved above 200-day)" if a > b
                    else "death cross (50-day moved below 200-day)", "date": dates[i],
                    "sessions_ago": len(xs) - 1 - i}
    return None


def technicals(bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs = _closes(bars)
    if len(xs) < 20:
        return {"ok": False, "reason": "not_enough_history", "sessions": len(xs)}
    dates = [b["date"] for b in bars if b.get("close") is not None]
    px = xs[-1]
    mas = {n: sma(xs, n) for n in (20, 50, 100, 200)}
    year = xs[-TRADING_DAYS:]
    hi, lo = max(year), min(year)
    hi_date = dates[len(xs) - len(year) + year.index(hi)]
    lo_date = dates[len(xs) - len(year) + year.index(lo)]
    vols = [b.get("volume") for b in bars if b.get("close") is not None]
    vol_info = None
    if vols and all(v is not None for v in vols[-51:]):
        v20, v50 = sum(vols[-21:-1]) / 20, sum(vols[-51:-1]) / 50
        vol_info = {"last_session": vols[-1], "avg_20d": round(v20), "avg_50d": round(v50),
                    "last_vs_20d_avg_x": round(vols[-1] / v20, 2) if v20 else None,
                    "20d_vs_50d_avg_x": round(v20 / v50, 2) if v50 else None}

    above = {f"above_sma_{n}": (px > m) if m else None for n, m in mas.items()}
    if mas[50] and mas[200]:
        if px > mas[50] > mas[200]:
            trend = "uptrend (price > 50-day > 200-day)"
        elif px < mas[50] < mas[200]:
            trend = "downtrend (price < 50-day < 200-day)"
        else:
            trend = "mixed / range-bound"
    elif mas[50]:
        trend = "above 50-day average" if px > mas[50] else "below 50-day average"
    else:
        trend = "not enough history for a trend read"
    r = rsi(xs)
    recent = xs[-63:]
    cross = _last_ma_cross(xs, dates)
    if mas[50] and mas[200]:
        state = ("the 50-day average is ABOVE the 200-day" if mas[50] > mas[200]
                 else "the 50-day average is BELOW the 200-day")
        if cross:
            when = (f"the last cross was a {cross['type'].split(' (')[0]} on {cross['date']} "
                    f"({cross['sessions_ago']} sessions ago)")
        else:
            when = f"no 50/200 cross in the last {len(xs) - 199} sessions"
        ma_summary = f"{state}; {when}. Quote this as written."
    else:
        ma_summary = "not enough history for a 50/200-day comparison"
    return {
        "ok": True,
        "as_of": dates[-1], "close": round(px, 4), "sessions_used": len(xs),
        "trend": trend,
        "moving_averages": {f"sma_{n}": round(m, 4) if m else None for n, m in mas.items()},
        "price_vs_ma_pct": {f"vs_sma_{n}": round((px / m - 1) * 100, 2) if m else None for n, m in mas.items()},
        **above,
        "sma_50_above_sma_200": (mas[50] > mas[200]) if mas[50] and mas[200] else None,
        "last_50_200_cross": cross or (
            {"type": "none found", "searched_sessions": max(len(xs) - 199, 0)} if len(xs) >= 200 else None),
        "ma_50_200_summary": ma_summary,
        "rsi_14": r,
        "rsi_zone": rsi_zone(r),
        "macd": macd(xs),
        "range_52w": {"high": round(hi, 4), "high_date": hi_date, "low": round(lo, 4), "low_date": lo_date,
                      "pct_below_high": round((px / hi - 1) * 100, 2),
                      "pct_above_low": round((px / lo - 1) * 100, 2),
                      "position_in_range_pct": round((px - lo) / (hi - lo) * 100, 1) if hi > lo else None},
        "recent_3m_support": round(min(recent), 4), "recent_3m_resistance": round(max(recent), 4),
        "volatility_ann_pct": {"20d": ann_vol(xs, 20), "60d": ann_vol(xs, 60), "1y": ann_vol(year)},
        "atr_14_pct": atr_pct(bars),
        "volume": vol_info,
        "returns_pct": {k: round((px / xs[-n - 1] - 1) * 100, 2) if len(xs) > n else None
                        for k, n in (("5d", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252))},
    }


def align(a: List[Dict[str, Any]], b: List[Dict[str, Any]]):
    bm = {x["date"]: float(x["close"]) for x in b if x.get("close") is not None}
    pairs = [(float(x["close"]), bm[x["date"]]) for x in a if x.get("close") is not None and x["date"] in bm]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def risk(bars: List[Dict[str, Any]], bench: List[Dict[str, Any]], bench_name: str) -> Dict[str, Any]:
    xs = _closes(bars)
    if len(xs) < 30:
        return {"ok": False, "reason": "not_enough_history", "sessions": len(xs)}
    out: Dict[str, Any] = {"ok": True, "from": bars[0]["date"], "to": bars[-1]["date"], "sessions": len(xs)}
    r = daily_returns(xs)
    total = xs[-1] / xs[0] - 1
    years = len(r) / TRADING_DAYS
    vol = stdev(r)
    out["total_return_pct"] = round(total * 100, 2)
    out["annualised_return_pct"] = round(((1 + total) ** (1 / years) - 1) * 100, 2) if years >= 0.9 else None
    out["volatility_ann_pct"] = round(vol * math.sqrt(TRADING_DAYS) * 100, 2) if vol else None
    mean = sum(r) / len(r)
    out["sharpe_no_rf"] = round(mean / vol * math.sqrt(TRADING_DAYS), 2) if vol else None
    down = [x for x in r if x < 0]
    dvol = stdev(down) if len(down) > 1 else None
    out["sortino_no_rf"] = round(mean / dvol * math.sqrt(TRADING_DAYS), 2) if dvol else None
    dd = max_drawdown(xs)
    if dd:
        dated = [b for b in bars if b.get("close") is not None]
        out["max_drawdown"] = {"pct": dd["pct"], "peak_date": dated[dd["peak_index"]]["date"],
                               "trough_date": dated[dd["trough_index"]]["date"]}
    out["worst_day_pct"] = round(min(r) * 100, 2)
    out["best_day_pct"] = round(max(r) * 100, 2)
    out["pct_up_days"] = round(sum(1 for x in r if x > 0) / len(r) * 100, 1)

    a, b = align(bars, bench)
    if len(a) >= 30:
        ra, rb = daily_returns(a), daily_returns(b)
        n = min(len(ra), len(rb))
        ra, rb = ra[-n:], rb[-n:]
        ma, mb = sum(ra) / n, sum(rb) / n
        cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (n - 1)
        vb, va = stdev(rb), stdev(ra)
        btot = b[-1] / b[0] - 1
        up = [(x, y) for x, y in zip(ra, rb) if y > 0]
        dn = [(x, y) for x, y in zip(ra, rb) if y < 0]
        out["vs_benchmark"] = {
            "benchmark": bench_name,
            "benchmark_return_pct": round(btot * 100, 2),
            "excess_return_pct": round((a[-1] / a[0] - btot - 1) * 100, 2),
            "beta": round(cov / vb ** 2, 2) if vb else None,
            "correlation": round(cov / (va * vb), 2) if va and vb else None,
            "upside_capture_pct": round(sum(x for x, _ in up) / sum(y for _, y in up) * 100, 1) if up else None,
            "downside_capture_pct": round(sum(x for x, _ in dn) / sum(y for _, y in dn) * 100, 1) if dn else None,
        }
    return out


def biggest_moves(bars: List[Dict[str, Any]], n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    moves = []
    for prev, b in zip(bars, bars[1:]):
        if prev.get("close") and b.get("close") is not None:
            moves.append({"date": b["date"], "change_pct": round((b["close"] / prev["close"] - 1) * 100, 2),
                          "close": b["close"], "volume": b.get("volume")})
    moves.sort(key=lambda m: m["change_pct"])
    return {"biggest_down_days": moves[:n], "biggest_up_days": list(reversed(moves[-n:]))}
