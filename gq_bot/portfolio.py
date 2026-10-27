"""Whole-watchlist and any-period analytics: performance over a dated window, calendar-year returns,
drawdown episodes, correlation between holdings, and the combined look-through of every fund.

All maths runs here on stored daily closes (split-adjusted, price only - no dividends) so the model
only has to quote the numbers.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from . import technicals as tech

MAX_CORR = 15
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PRICE_ONLY = "Price return from split-adjusted closes; dividends are not included."


def _pct(a: float, b: float) -> Optional[float]:
    return round((a / b - 1) * 100, 2) if b else None


def _closes(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [b for b in bars if b.get("close") is not None and float(b["close"]) > 0]


def drawdown_episodes(bars: List[Dict[str, Any]], top: int = 5, min_pct: float = 5.0) -> Dict[str, Any]:
    """Peak -> trough -> recovery episodes, deepest first, plus where the price sits now."""
    bs = _closes(bars)
    if len(bs) < 2:
        return {"episodes": [], "current": None}
    eps: List[Dict[str, Any]] = []
    peak_i = 0
    trough_i = 0
    for i in range(1, len(bs)):
        c, pk = float(bs[i]["close"]), float(bs[peak_i]["close"])
        if c >= pk:
            if trough_i > peak_i and (float(bs[trough_i]["close"]) / pk - 1) * 100 <= -min_pct:
                eps.append((peak_i, trough_i, i))
            peak_i = trough_i = i
        elif c < float(bs[trough_i]["close"]) or trough_i == peak_i:
            trough_i = i
    open_ep = None
    if trough_i > peak_i and (float(bs[trough_i]["close"]) / float(bs[peak_i]["close"]) - 1) * 100 <= -min_pct:
        open_ep = (peak_i, trough_i, None)
    rows = []
    for p, t, r in eps + ([open_ep] if open_ep else []):
        pk, tr = float(bs[p]["close"]), float(bs[t]["close"])
        d0, d1 = date.fromisoformat(bs[p]["date"]), date.fromisoformat(bs[t]["date"])
        row = {"peak_date": bs[p]["date"], "trough_date": bs[t]["date"], "depth_pct": _pct(tr, pk),
               "days_peak_to_trough": (d1 - d0).days}
        if r is None:
            row.update(recovered=False, recovery_date=None)
        else:
            row.update(recovered=True, recovery_date=bs[r]["date"],
                       days_trough_to_recovery=(date.fromisoformat(bs[r]["date"]) - d1).days)
        rows.append(row)
    rows.sort(key=lambda x: x["depth_pct"])
    run_peak = max(float(b["close"]) for b in bs)
    pk_b = next(b for b in bs if float(b["close"]) == run_peak)
    last = bs[-1]
    return {"episodes": rows[:top], "episodes_found": len(rows),
            "current": {"as_of": last["date"], "close": float(last["close"]),
                        "pct_below_window_high": _pct(float(last["close"]), run_peak),
                        "window_high": run_peak, "window_high_date": pk_b["date"]}}


def calendar_stats(bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calendar-year and month-by-month returns from daily closes."""
    bs = _closes(bars)
    if len(bs) < 2:
        return {"years": [], "months": {}}
    by_year: Dict[int, Tuple[float, str]] = {}
    by_month: Dict[Tuple[int, int], float] = {}
    for b in bs:
        y, m = int(b["date"][:4]), int(b["date"][5:7])
        by_year[y] = (float(b["close"]), b["date"])
        by_month[(y, m)] = float(b["close"])
    first_y = int(bs[0]["date"][:4])
    years = []
    prev = float(bs[0]["close"])
    for y in sorted(by_year):
        end, d = by_year[y]
        partial = (y == first_y and bs[0]["date"][5:] > "01-10") or (y == int(bs[-1]["date"][:4]) and d[5:] < "12-24")
        years.append({"year": y, "return_pct": _pct(end, prev), "to": d, "partial": bool(partial)})
        prev = end
    mkeys = sorted(by_month)
    monthly = []
    for a, b in zip(mkeys, mkeys[1:]):
        monthly.append({"month": f"{MONTHS[b[1] - 1]} {b[0]}", "return_pct": _pct(by_month[b], by_month[a])})
    seasonal: Dict[str, List[float]] = {}
    for a, b in zip(mkeys, mkeys[1:]):
        seasonal.setdefault(MONTHS[b[1] - 1], []).append(by_month[b] / by_month[a] - 1)
    avg_month = {m: round(sum(v) / len(v) * 100, 2) for m, v in seasonal.items() if len(v) >= 3}
    out: Dict[str, Any] = {"years": years}
    if monthly:
        srt = sorted(monthly, key=lambda x: x["return_pct"])
        out["months"] = {"count": len(monthly), "pct_positive": round(sum(1 for m in monthly if m["return_pct"] > 0)
                                                                      / len(monthly) * 100, 1),
                         "best": list(reversed(srt[-3:])), "worst": srt[:3]}
    if avg_month:
        out["average_return_by_calendar_month_pct"] = avg_month
    return out


def _corr(a: List[float], b: List[float]) -> Optional[float]:
    n = len(a)
    if n < 30:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if not va or not vb:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def correlation_matrix(series: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Daily-return correlations between several price series on their common dates."""
    maps = {t: {b["date"]: float(b["close"]) for b in _closes(bs)} for t, bs in series.items()}
    maps = {t: m for t, m in maps.items() if len(m) >= 31}
    if len(maps) < 2:
        return {"ok": False, "reason": "not_enough_history"}
    common = sorted(set.intersection(*(set(m) for m in maps.values())))
    if len(common) < 31:
        return {"ok": False, "reason": "not_enough_common_history", "common_sessions": len(common)}
    rets = {t: [m[common[i]] / m[common[i - 1]] - 1 for i in range(1, len(common))] for t, m in maps.items()}
    names = list(rets)
    pairs, matrix = [], {t: {} for t in names}
    for i, a in enumerate(names):
        matrix[a][a] = 1.0
        for b in names[i + 1:]:
            c = _corr(rets[a], rets[b])
            c = round(c, 2) if c is not None else None
            matrix[a][b] = matrix[b][a] = c
            if c is not None:
                pairs.append({"a": a, "b": b, "correlation": c})
    pairs.sort(key=lambda p: -p["correlation"])
    avg = {t: round(sum(v for k, v in matrix[t].items() if k != t and v is not None)
                    / max(1, sum(1 for k, v in matrix[t].items() if k != t and v is not None)), 2) for t in names}
    allc = [p["correlation"] for p in pairs]
    return {"ok": True, "from": common[0], "to": common[-1], "sessions": len(common) - 1, "tickers": names,
            "average_pair_correlation": round(sum(allc) / len(allc), 2) if allc else None,
            "most_correlated": pairs[:5], "least_correlated": list(reversed(pairs[-5:])),
            "average_correlation_by_ticker": dict(sorted(avg.items(), key=lambda kv: kv[1])),
            "matrix": matrix}


class PortfolioMixin:
    """Mixed into GQData; relies on its _scoped, _instrument, _bars, _map, _window and fund helpers."""

    def _price_bars(self, t: str, start: date, end: date):
        inst = self._instrument(t)
        if not inst:
            return None, []
        return inst, self._bars(inst["id"], start=start, end=end)

    # ── performance over any dated period, with a what-if ────────────────────
    def performance(self, user_id, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                    amount: Optional[float] = None, monthly_amount: Optional[float] = None,
                    benchmark: str = "SPY") -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        b, bno = self._scoped_or_benchmark(user_id, benchmark or "SPY")
        if bno:
            return bno
        start, end = self._window(365, start_date, end_date, default_days=365)

        def run():
            inst, bars = self._price_bars(t, start, end)
            if not inst:
                return {"ok": False, "reason": "no_price_data", "ticker": t,
                        "message": f"The heatmap has no active US price data for {t}."}
            bs = _closes(bars)
            if len(bs) < 2:
                return {"ok": False, "reason": "not_enough_history", "ticker": t, "sessions": len(bs)}
            first, last = bs[0], bs[-1]
            p0, p1 = float(first["close"]), float(last["close"])
            days = (date.fromisoformat(last["date"]) - date.fromisoformat(first["date"])).days
            out: Dict[str, Any] = {
                "ok": True, "ticker": t, "requested": {"from": start.isoformat(), "to": end.isoformat()},
                "first_session": first["date"], "last_session": last["date"], "start_close": p0, "end_close": p1,
                "price_return_pct": _pct(p1, p0), "calendar_days": days, "note": PRICE_ONLY}
            if first["date"] > (start + timedelta(days=7)).isoformat():
                out["history_starts_late"] = f"Stored history for {t} only begins {first['date']}."
            if days >= 365:
                out["annualised_return_pct"] = round(((p1 / p0) ** (365.0 / days) - 1) * 100, 2)
            dd = tech.max_drawdown([float(x["close"]) for x in bs])
            if dd:
                out["max_drawdown"] = {"pct": dd["pct"], "peak_date": bs[dd["peak_index"]]["date"],
                                       "trough_date": bs[dd["trough_index"]]["date"]}
            r = tech.daily_returns([float(x["close"]) for x in bs])
            if r:
                out["best_day_pct"], out["worst_day_pct"] = round(max(r) * 100, 2), round(min(r) * 100, 2)
                vol = tech.stdev(r)
                if vol is not None and len(r) >= 20:
                    out["volatility_ann_pct"] = round(vol * math.sqrt(tech.TRADING_DAYS) * 100, 2)
            if amount is not None and float(amount) > 0:
                amt = float(amount)
                out["lump_sum"] = {"invested": round(amt, 2), "value_now": round(amt * p1 / p0, 2),
                                   "gain": round(amt * (p1 / p0 - 1), 2)}
            if monthly_amount is not None and float(monthly_amount) > 0:
                m = float(monthly_amount)
                seen, shares, spent = set(), 0.0, 0.0
                for x in bs:
                    ym = x["date"][:7]
                    if ym not in seen:
                        seen.add(ym)
                        shares += m / float(x["close"])
                        spent += m
                value = shares * p1
                out["monthly_investing"] = {"monthly_amount": round(m, 2), "months": len(seen),
                                            "total_invested": round(spent, 2), "value_now": round(value, 2),
                                            "gain": round(value - spent, 2), "gain_pct": _pct(value, spent),
                                            "average_cost": round(spent / shares, 2),
                                            "method": "buys on the first session of each month"}
            if b != t:
                binst, bb = self._price_bars(b, start, end)
                bbs = [x for x in _closes(bb) if first["date"] <= x["date"] <= last["date"]]
                if len(bbs) >= 2:
                    br = _pct(float(bbs[-1]["close"]), float(bbs[0]["close"]))
                    out["vs_benchmark"] = {"benchmark": b, "benchmark_return_pct": br,
                                           "excess_return_pct": round(out["price_return_pct"] - br, 2)}
            return out
        return self._safe("performance", run)

    # ── calendar years and months ────────────────────────────────────────────
    def calendar_returns(self, user_id, ticker: str, years: int = 10, benchmark: str = "SPY") -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        b, bno = self._scoped_or_benchmark(user_id, benchmark or "SPY")
        if bno:
            return bno
        years = max(1, min(int(years or 10), 10))
        end = self.today()
        start = date(end.year - years, 1, 1) - timedelta(days=10)   # a few days of the prior year to anchor Jan

        def run():
            inst, bars = self._price_bars(t, start, end)
            if not inst:
                return {"ok": False, "reason": "no_price_data", "ticker": t,
                        "message": f"The heatmap has no active US price data for {t}."}
            stats = calendar_stats(bars)
            if not stats["years"]:
                return {"ok": False, "reason": "not_enough_history", "ticker": t}
            out = {"ok": True, "ticker": t, "note": PRICE_ONLY, **stats}
            best = max((y for y in stats["years"] if not y["partial"]), key=lambda y: y["return_pct"], default=None)
            worst = min((y for y in stats["years"] if not y["partial"]), key=lambda y: y["return_pct"], default=None)
            out["best_full_year"], out["worst_full_year"] = best, worst
            if b != t:
                binst, bb = self._price_bars(b, start, end)
                if binst:
                    by = {y["year"]: y["return_pct"] for y in calendar_stats(bb)["years"]}
                    for y in out["years"]:
                        if by.get(y["year"]) is not None and y["return_pct"] is not None:
                            y[f"vs_{b}_pts"] = round(y["return_pct"] - by[y["year"]], 2)
                    out["benchmark"] = b
            return out
        return self._safe("calendar returns", run)

    # ── drawdowns ────────────────────────────────────────────────────────────
    def drawdowns(self, user_id, ticker: str, days: int = 1825, top: int = 5) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        start, end = self._window(days, None, None, default_days=1825)
        top = max(1, min(int(top or 5), 10))

        def run():
            inst, bars = self._price_bars(t, start, end)
            if not inst:
                return {"ok": False, "reason": "no_price_data", "ticker": t,
                        "message": f"The heatmap has no active US price data for {t}."}
            res = drawdown_episodes(bars, top)
            if res["current"] is None:
                return {"ok": False, "reason": "not_enough_history", "ticker": t}
            return {"ok": True, "ticker": t, "window": {"from": start.isoformat(), "to": end.isoformat()},
                    "note": "Only falls of 5% or more from a prior peak are listed. " + PRICE_ONLY, **res}
        return self._safe("drawdowns", run)

    # ── correlation between holdings ─────────────────────────────────────────
    def watchlist_correlation(self, user_id, tickers: Optional[List[str]] = None, days: int = 365) -> Dict[str, Any]:
        wl = self.watchlist(user_id)
        wanted: List[str] = []
        refused: List[str] = []
        for raw in tickers or wl:
            t, no = self._scoped(user_id, raw)
            if no:
                refused.append(str(raw).upper())
            elif t not in wanted:
                wanted.append(t)
        if tickers and not wanted:
            return {"ok": False, "reason": "not_on_watchlist", "tickers": refused}
        kinds = self._kinds(wanted)
        no_prices = [t for t in wanted if kinds.get(t) == "mutual_fund"]
        pool = [t for t in wanted if t not in no_prices]
        start, end = self._window(days, None, None, default_days=365)

        def run():
            insts = self._instruments_for(pool)
            usable = [t for t in pool if t in insts]
            skipped_cap = usable[MAX_CORR:]
            usable = usable[:MAX_CORR]
            series = dict(zip(usable, self._map(lambda t: self._bars(insts[t]["id"], start=start, end=end), usable)))
            res = correlation_matrix(series)
            if not res.get("ok"):
                return {**res, "tickers": usable}
            res["interpretation"] = ("0.7 or more is high (little diversification), 0.3 to 0.7 moderate, "
                                     "below 0.3 low, negative means they tend to move opposite ways.")
            if no_prices:
                res["excluded_mutual_funds_no_prices"] = no_prices
            missing = [t for t in pool if t not in insts]
            if missing:
                res["not_covered_by_heatmap"] = missing
            if skipped_cap:
                res["not_included_cap_reached"] = skipped_cap
            if refused:
                res["not_on_watchlist"] = refused
            return res
        return self._safe("correlation", run)

    # ── combined look-through ────────────────────────────────────────────────
    def portfolio_exposure(self, user_id, tickers: Optional[List[str]] = None,
                           weights: Optional[Dict[str, float]] = None, top_n: int = 15) -> Dict[str, Any]:
        """What the funds hold in total: shared names, a combined sector mix, and how much funds repeat each other.
        Weights are the user's own (percent per ticker); without them every fund counts equally."""
        wl = self.watchlist(user_id)
        wanted: List[str] = []
        refused: List[str] = []
        for raw in tickers or wl:
            t, no = self._scoped(user_id, raw)
            if no:
                refused.append(str(raw).upper())
            elif t not in wanted:
                wanted.append(t)
        if tickers and not wanted:
            return {"ok": False, "reason": "not_on_watchlist", "tickers": refused}
        top_n = max(3, min(int(top_n or 15), 30))
        kinds = self._kinds(wanted)
        funds = [t for t in wanted if kinds.get(t) in ("etf", "mutual_fund")]
        if not funds:
            return {"ok": False, "reason": "no_funds", "message": "None of those tickers is an ETF or a mutual fund."}

        w: Dict[str, float] = {}
        given = {str(k).upper(): float(v) for k, v in (weights or {}).items() if v is not None}
        custom = any(t in given for t in funds)
        for t in funds:
            w[t] = given.get(t, 0.0) if custom else 1.0
        total_w = sum(w.values())
        if total_w <= 0:
            return {"ok": False, "reason": "bad_weights", "message": "The weights must add up to more than zero."}
        w = {t: v / total_w for t, v in w.items()}

        def run():
            names: Dict[str, str] = {}
            combined: Dict[str, float] = {}
            held_by: Dict[str, List[str]] = {}
            sectors: Dict[str, float] = {}
            assets: Dict[str, float] = {}
            used, missing, dates = [], [], {}
            for t in funds:
                if w.get(t, 0) <= 0:
                    continue
                kind = kinds[t]
                hs, as_of = self._fund_holdings(t, kind)
                if not hs:
                    missing.append(t)
                    continue
                used.append(t)
                dates[t] = as_of
                for h in hs:
                    key = h["symbol"] or h["key"]
                    if not key or h["weight"] is None:
                        continue
                    combined[key] = combined.get(key, 0.0) + w[t] * h["weight"]
                    names.setdefault(key, h["name"])
                    held_by.setdefault(key, []).append(t)
                bd = self._safe("breakdown", lambda t=t, kind=kind, hs=hs: self._breakdown(t, kind, hs), default={}) or {}
                for src, dst in (("by_sector", sectors), ("by_asset_class", assets)):
                    for row in bd.get(src) or []:
                        if row.get("name") and row.get("weight_pct") is not None:
                            dst[row["name"]] = dst.get(row["name"], 0.0) + w[t] * float(row["weight_pct"])
            if not used:
                return {"ok": False, "reason": "no_holdings", "funds": funds}
            rank = sorted(combined.items(), key=lambda kv: -kv[1])
            top = [{"symbol": k, "name": names.get(k), "combined_weight_pct": round(v, 3),
                    "held_by": held_by[k], "funds_holding": len(held_by[k])} for k, v in rank[:top_n]]
            shared = sorted(((k, v) for k, v in held_by.items() if len(v) >= 2 and len(used) >= 2),
                            key=lambda kv: (-len(kv[1]), -combined[kv[0]]))
            top10 = round(sum(v for _, v in rank[:10]), 2)
            fmt = lambda d: [{"name": k, "weight_pct": round(v, 2)} for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:10]]
            out: Dict[str, Any] = {
                "ok": True, "funds_used": used, "weights_pct": {t: round(w[t] * 100, 1) for t in used},
                "weighting": "your weights" if custom else "equal weight across funds (no weights given)",
                "distinct_holdings": len(combined), "top_10_combined_weight_pct": top10,
                "top_combined_holdings": top, "holdings_in_two_or_more_funds": len(shared),
                "most_repeated": [{"symbol": k, "name": names.get(k), "funds_holding": len(v), "held_by": v,
                                   "combined_weight_pct": round(combined[k], 3)} for k, v in shared[:8]],
                "sector_mix_pct": fmt(sectors), "asset_class_mix_pct": fmt(assets),
                "holdings_as_of": dates,
                "note": "Look-through of each fund's latest stored holdings; mutual funds are as of their last N-PORT "
                        "filing. Sector and asset mixes only cover funds that publish them."}
            if missing:
                out["funds_without_holdings"] = missing
            if refused:
                out["not_on_watchlist"] = refused
            return out
        return self._safe("portfolio exposure", run)
