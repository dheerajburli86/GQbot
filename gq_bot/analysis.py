"""Pure calculations over price bars. No I/O, so every number is testable."""

from __future__ import annotations

from statistics import median
from typing import Dict, Iterable, List, Optional

# A requested window in calendar days -> the returns_latest column the heatmap
# tile shows for that window. Comparing the stock and its sector on the SAME
# stored column keeps the bot's numbers identical to what the dashboard shows.
_WINDOW_COLUMNS = ((1, "ret_1d"), (7, "ret_1w"), (31, "ret_1m"), (92, "ret_3m"),
                   (183, "ret_6m"), (366, "ret_1y"))


def window_column(days: int) -> str:
    for limit, col in _WINDOW_COLUMNS:
        if days <= limit:
            return col
    return "ret_1y"


def pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Percent change from a to b, rounded to 2 dp; None if undefined."""
    if a in (None, 0) or b is None:
        return None
    return round((float(b) - float(a)) / float(a) * 100.0, 2)


def summarise_bars(bars: List[Dict]) -> Dict:
    """Move, range, worst drawdown and biggest single-session moves over the bars.

    `bars` are {"date": "YYYY-MM-DD", "close": float}, any order.
    """
    rows = sorted((b for b in bars if b.get("close") is not None), key=lambda b: b["date"])
    if len(rows) < 2:
        return {"sessions": len(rows), "enough_data": False}

    closes = [float(b["close"]) for b in rows]
    peak, peak_date, max_dd, dd_from, dd_to = closes[0], rows[0]["date"], 0.0, None, None
    for b, c in zip(rows, closes):
        if c > peak:
            peak, peak_date = c, b["date"]
        dd = (c - peak) / peak * 100.0
        if dd < max_dd:
            max_dd, dd_from, dd_to = dd, peak_date, b["date"]

    daily = [{"date": rows[i]["date"], "move_pct": pct(closes[i - 1], closes[i])}
             for i in range(1, len(rows))]
    daily = [d for d in daily if d["move_pct"] is not None]
    hi = max(range(len(closes)), key=closes.__getitem__)
    lo = min(range(len(closes)), key=closes.__getitem__)
    down_days = sum(1 for d in daily if d["move_pct"] < 0)

    return {
        "enough_data": True,
        "sessions": len(rows),
        "from": rows[0]["date"], "to": rows[-1]["date"],
        "start_close": closes[0], "end_close": closes[-1],
        "change_pct": pct(closes[0], closes[-1]),
        "high": {"date": rows[hi]["date"], "close": closes[hi]},
        "low": {"date": rows[lo]["date"], "close": closes[lo]},
        "max_drawdown_pct": round(max_dd, 2),
        "drawdown_window": [dd_from, dd_to] if dd_from else None,
        "down_sessions": down_days, "up_sessions": len(daily) - down_days,
        "biggest_up_days": sorted(daily, key=lambda d: d["move_pct"], reverse=True)[:3],
        "biggest_down_days": sorted(daily, key=lambda d: d["move_pct"])[:3],
    }


def group_stats(values: Iterable[Optional[float]]) -> Dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0}
    return {"count": len(vals), "median_pct": round(median(vals), 2),
            "share_positive": round(sum(v > 0 for v in vals) / len(vals), 2)}


def relative(stock: Optional[float], group_median: Optional[float]) -> Optional[float]:
    if stock is None or group_median is None:
        return None
    return round(float(stock) - float(group_median), 2)
