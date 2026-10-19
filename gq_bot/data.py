"""
Read-only data layer.

SCOPE RULE: a user can ask only about tickers on THEIR OWN FinXray watchlist.
Every public lookup that takes a ticker goes through `_scoped()`, so the LLM
cannot reach an out-of-scope ticker by forgetting a rule. It lives here and not
in the prompt because a prompt instruction is a request, and a function that
refuses is a guarantee. It also matches what the data can support: FinXray only
ingests events for watchlisted tickers, so outside a watchlist there would be a
price move with no "why" behind it.

The one deliberate exception is CONTEXT inside `explain_move`: the stock's
sector median and the S&P 500 (SPY) return over the same window. They are
aggregates used to say "the whole market fell too"; no tool answers a
standalone question about a non-watchlisted ticker.

Sources
  FinXray US   users, watchlists, alerts, etf_master, etf_company_data,
               etf_expenses, etf_holdings, etf_holdings_changes, etf_fund_events
  Heatmap      instruments, returns_latest, price_history, us_stock_fundamentals

Every source read is wrapped so that one missing table or renamed column turns
into a `source_error` for that piece, never an exception that kills the answer.
Nothing here writes.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import analysis

log = logging.getLogger(__name__)

US_UNIVERSES = ("us_stocks", "us_etfs")
RET_COLUMNS = ("ret_1d", "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_ytd",
               "ret_1y", "ret_5y", "ret_10y", "ret_all")
MARKET_BENCHMARK = "SPY"
MAX_EVENT_DAYS = 90
MAX_EVENTS = 20
MAX_HISTORY_DAYS = 400
MAX_COMPARE = 5
CHUNK = 100

_FUND_KEYS = {
    "income_statement": ("revenue", "gross_profit", "operating_income", "net_income",
                         "eps", "eps_diluted", "ebitda"),
    "balance_sheet": ("cash", "cash_and_equivalents", "total_assets", "total_liabilities",
                      "total_debt", "total_equity", "shareholders_equity"),
    "cash_flow": ("operating_cash_flow", "capex", "capital_expenditure", "free_cash_flow"),
}


def norm(ticker: str) -> str:
    return (ticker or "").strip().upper().lstrip("$")


def refusal(ticker: str, reason: str, message: str, **extra) -> Dict[str, Any]:
    return {"ok": False, "ticker": ticker, "reason": reason, "message": message, **extra}


class _TTLCache:
    def __init__(self, ttl_s: float, clock: Callable[[], float]):
        self.ttl, self.clock, self.items = ttl_s, clock, {}

    def get(self, key):
        hit = self.items.get(key)
        if hit and self.clock() - hit[0] < self.ttl:
            return hit[1]
        return None

    def put(self, key, value):
        self.items[key] = (self.clock(), value)
        return value

    def drop(self, key):
        self.items.pop(key, None)


class GQData:
    def __init__(self, finxray, heatmap=None, clock: Callable[[], float] = time.monotonic,
                 today: Callable[[], date] = date.today):
        self.fx = finxray
        self.hm = heatmap or finxray
        self.today = today
        self._wl = _TTLCache(60, clock)
        self._inst = _TTLCache(600, clock)
        self._sector = _TTLCache(900, clock)

    # ── plumbing ──────────────────────────────────────────────────────────────
    @staticmethod
    def _rows(query) -> List[Dict[str, Any]]:
        return query.execute().data or []

    def _safe(self, label: str, fn: Callable[[], Any], default=None):
        try:
            return fn()
        except Exception as e:  # one broken source must not sink the answer
            log.warning("[data] %s failed: %s", label, e)
            return default if default is not None else {"ok": False, "reason": "source_error",
                                                          "message": f"{label} is unavailable right now."}

    # ── identity + scope ──────────────────────────────────────────────────────
    def get_user(self, chat_id, username: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """The FinXray user behind a Telegram chat.

        A private chat's id equals the Telegram user id, which is the same for
        every bot, so the chat id the FinXray bot stored resolves here too. The
        username is the fallback for a user whose row predates chat-id capture.
        """
        rows = self._rows(self.fx.table("users").select("id, telegram_username")
                          .eq("telegram_chat_id", str(chat_id)).limit(1))
        if not rows and username:
            # ILIKE for case-insensitivity only. `_` and `%` are wildcards in
            # LIKE, and a Telegram username can contain `_`, so an unescaped
            # pattern would let "d_12312" match someone else's "dj12312".
            # Escape them AND confirm an exact match, so this can never resolve
            # to a different person.
            wanted = username.lstrip("@").lower()
            pattern = wanted.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = [r for r in self._rows(self.fx.table("users").select("id, telegram_username")
                                          .ilike("telegram_username", pattern).limit(5))
                    if (r.get("telegram_username") or "").lstrip("@").lower() == wanted][:1]
        return rows[0] if rows else None

    def watchlist(self, user_id, force: bool = False) -> List[str]:
        if not force:
            hit = self._wl.get(user_id)
            if hit is not None:
                return hit
        rows = self._rows(self.fx.table("watchlists").select("ticker").eq("user_id", user_id))
        return self._wl.put(user_id, sorted({norm(r["ticker"]) for r in rows if r.get("ticker")}))

    def watchlist_details(self, user_id) -> List[Dict[str, Any]]:
        """Watchlist tickers with name and type, for the prompt's ticker resolution."""
        tickers = self.watchlist(user_id)
        found = {}
        for i in range(0, len(tickers), CHUNK):
            part = tickers[i:i + CHUNK]
            for r in self._safe("instruments", lambda p=part: self._rows(
                    self.hm.table("instruments").select("symbol, name, universe")
                    .in_("symbol", p).in_("universe", list(US_UNIVERSES)).eq("is_active", True)),
                    default=[]):
                found[norm(r["symbol"])] = r
        return [{"ticker": t,
                 "name": (found.get(t) or {}).get("name"),
                 "type": {"us_etfs": "etf", "us_stocks": "stock"}.get(
                     (found.get(t) or {}).get("universe"), "unknown")} for t in tickers]

    def _scoped(self, user_id, ticker: str) -> Tuple[Optional[str], Optional[Dict]]:
        t = norm(ticker)
        if not t:
            return None, refusal(t, "no_ticker", "Which ticker do you mean?")
        if t in self.watchlist(user_id):
            return t, None
        return None, refusal(
            t, "not_on_watchlist",
            f"{t} is not on your watchlist. Add it in the GQ FinXray bot with /add {t} "
            f"and then ask me again.", watchlist=self.watchlist(user_id))

    # ── heatmap: instruments + price ──────────────────────────────────────────
    def _instrument(self, ticker: str) -> Optional[Dict[str, Any]]:
        hit = self._inst.get(ticker)
        if hit is not None:
            return hit or None
        rows = self._rows(self.hm.table("instruments")
                          .select("id, symbol, name, universe, sector, industry, market_cap_usd")
                          .eq("symbol", ticker).in_("universe", list(US_UNIVERSES))
                          .eq("is_active", True).limit(1))
        return self._inst.put(ticker, rows[0] if rows else {}) or None

    def _returns(self, instrument_id) -> Optional[Dict[str, Any]]:
        rows = self._rows(self.hm.table("returns_latest")
                          .select("price_usd, as_of, " + ", ".join(RET_COLUMNS))
                          .eq("instrument_id", instrument_id).limit(1))
        return rows[0] if rows else None

    def _snapshot(self, t: str) -> Dict[str, Any]:
        inst = self._instrument(t)
        if not inst:
            return refusal(t, "no_price_data", f"The heatmap has no active US price data for {t}.")
        r = self._returns(inst["id"])
        base = {"ticker": t, "name": inst.get("name"),
                "type": "etf" if inst.get("universe") == "us_etfs" else "stock",
                "sector": inst.get("sector"), "industry": inst.get("industry"),
                "market_cap_usd": inst.get("market_cap_usd")}
        if not r:
            return {"ok": False, "reason": "no_returns", **base,
                    "message": f"{t} has no computed returns yet."}
        return {"ok": True, **base, "price_usd": r.get("price_usd"),
                "as_of": r.get("as_of"), "returns_pct": {k: r.get(k) for k in RET_COLUMNS}}

    def snapshot(self, user_id, ticker: str) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        return self._safe("price snapshot", lambda: self._snapshot(t))

    def _bars(self, instrument_id, days: int) -> List[Dict[str, Any]]:
        since = (self.today() - timedelta(days=days)).isoformat()
        rows = self._rows(self.hm.table("price_history").select("date, close")
                          .eq("instrument_id", instrument_id).gte("date", since)
                          .order("date", desc=True).limit(days + 10))
        return sorted(({"date": str(b["date"])[:10], "close": b["close"]} for b in rows),
                      key=lambda b: b["date"])

    def price_history(self, user_id, ticker: str, days: int = 30) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        days = max(2, min(int(days), MAX_HISTORY_DAYS))

        def run():
            inst = self._instrument(t)
            if not inst:
                return refusal(t, "no_price_data", f"The heatmap has no active US price data for {t}.")
            bars = self._bars(inst["id"], days)
            out = {"ok": True, "ticker": t, "days": days, "summary": analysis.summarise_bars(bars)}
            # Long windows are summarised, not dumped: 400 closes cost tokens
            # and tell the model nothing the summary does not.
            out["bars"] = bars if len(bars) <= 60 else bars[-20:]
            out["bars_note"] = None if len(bars) <= 60 else "last 20 sessions shown; summary covers all"
            return out
        return self._safe("price history", run)

    # ── heatmap: fundamentals (stocks) ────────────────────────────────────────
    @staticmethod
    def _trim_period(p: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: p.get(k) for k in ("period", "fiscal_year", "filing_date", "date") if p.get(k)}
        for section, keys in _FUND_KEYS.items():
            block = p.get(section)
            if isinstance(block, dict):
                picked = {k: block[k] for k in keys if block.get(k) is not None}
                out[section] = picked or dict(list(block.items())[:12])
        return out

    def fundamentals(self, user_id, ticker: str) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no

        def run():
            rows = self._rows(self.hm.table("us_stock_fundamentals").select("data, updated_at")
                              .eq("ticker", t).limit(1))
            if not rows:
                return refusal(t, "no_fundamentals",
                               f"No fundamentals are stored for {t} (ETFs have none).")
            data = rows[0].get("data") or {}
            return {"ok": True, "ticker": t, "updated_at": rows[0].get("updated_at"),
                    "quarterly": [self._trim_period(p) for p in (data.get("quarterly") or [])[:4]],
                    "annual": [self._trim_period(p) for p in (data.get("annual") or [])[:2]]}
        return self._safe("fundamentals", run)

    # ── FinXray: events ───────────────────────────────────────────────────────
    def _events(self, t: str, days: int, limit: int = MAX_EVENTS,
                min_impact: Optional[str] = None) -> List[Dict[str, Any]]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._rows(self.fx.table("alerts")
                          .select("id, summary, impact, source, filing_type, link, created_at, extra")
                          .eq("ticker", t).gte("created_at", since)
                          .order("created_at", desc=True).limit(limit))
        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        floor = rank.get((min_impact or "").upper(), 0)
        return [{"when": r.get("created_at"), "summary": r.get("summary"),
                 "impact": r.get("impact"), "type": r.get("filing_type"),
                 "feature": (r.get("extra") or {}).get("feature_name"),
                 "source": r.get("source"), "link": r.get("link")}
                for r in rows if rank.get((r.get("impact") or "LOW").upper(), 0) >= floor]

    def events(self, user_id, ticker: str, days: int = 14,
               min_impact: Optional[str] = None) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        days = max(1, min(int(days), MAX_EVENT_DAYS))
        return self._safe("FinXray events", lambda: {
            "ok": True, "ticker": t, "days": days,
            "events": self._events(t, days, min_impact=min_impact)})

    # ── the join: why did it move? ────────────────────────────────────────────
    def _sector_stats(self, sector: str, col: str) -> Dict[str, Any]:
        key = (sector, col)
        hit = self._sector.get(key)
        if hit is not None:
            return hit
        ids, page = [], 0
        while True:
            chunk = self._rows(self.hm.table("instruments").select("id")
                               .eq("universe", "us_stocks").eq("is_active", True)
                               .eq("sector", sector).range(page * 1000, page * 1000 + 999))
            ids.extend(r["id"] for r in chunk)
            if len(chunk) < 1000:
                break
            page += 1
        vals = []
        for i in range(0, len(ids), CHUNK):
            vals.extend(r.get(col) for r in self._rows(
                self.hm.table("returns_latest").select(f"instrument_id, {col}")
                .in_("instrument_id", ids[i:i + CHUNK])))
        return self._sector.put(key, {"sector": sector, **analysis.group_stats(vals)})

    def explain_move(self, user_id, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Everything needed to explain a move: the price path, the events in
        (and just before) that window, and the sector + market over it."""
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        days = max(1, min(int(days), MAX_EVENT_DAYS))
        col = analysis.window_column(days)
        snap = self._safe("price snapshot", lambda: self._snapshot(t))
        out: Dict[str, Any] = {"ok": bool(snap.get("ok")), "ticker": t, "window_days": days,
                               "compared_on": col, "snapshot": snap}

        inst = self._safe("instrument", lambda: self._instrument(t), default={}) or {}
        if inst:
            def path():
                bars = self._bars(inst["id"], days)
                if len(bars) < 2:  # a 1-day window over a weekend has one bar
                    bars = self._bars(inst["id"], days + 7)
                return analysis.summarise_bars(bars)
            out["price_path"] = self._safe("price history", path)

        # News often lands a day or two before the session it moves.
        lead = min(days + 3, MAX_EVENT_DAYS)
        out["events"] = self._safe("FinXray events", lambda: self._events(t, lead), default=[])
        out["events_window_days"] = lead

        stock_ret = (snap.get("returns_pct") or {}).get(col) if snap.get("ok") else None
        if snap.get("type") == "stock" and inst.get("sector"):
            sec = self._safe("sector context", lambda: self._sector_stats(inst["sector"], col))
            if isinstance(sec, dict) and sec.get("count"):
                sec["stock_minus_sector_pct"] = analysis.relative(stock_ret, sec.get("median_pct"))
            out["sector_context"] = sec

        def market():
            m = self._instrument(MARKET_BENCHMARK)
            r = self._returns(m["id"]) if m else None
            v = (r or {}).get(col)
            return {"benchmark": MARKET_BENCHMARK, "return_pct": v,
                    "stock_minus_market_pct": analysis.relative(stock_ret, v)}
        out["market_context"] = self._safe("market context", market)
        return out

    def compare(self, user_id, tickers: List[str]) -> Dict[str, Any]:
        seen, results = [], []
        for raw in tickers or []:
            t = norm(raw)
            if t and t not in seen:
                seen.append(t)
        for t in seen[:MAX_COMPARE]:
            results.append(self.snapshot(user_id, t))
        out = {"ok": any(r.get("ok") for r in results), "results": results}
        if len(seen) > MAX_COMPARE:
            out["note"] = f"Compared the first {MAX_COMPARE} tickers only."
        return out

    # ── FinXray: ETFs ─────────────────────────────────────────────────────────
    def etf_profile(self, user_id, ticker: str) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no

        def first(table):
            rows = self._rows(self.fx.table(table).select("*").eq("ticker", t).limit(1))
            return rows[0] if rows else None

        master = self._safe("etf_master", lambda: first("etf_master"), default={})
        if not master:
            return refusal(t, "not_an_etf", f"{t} is not in the FinXray ETF universe.")
        drop = ("updated_at", "created_at", "id")
        clean = lambda d: {k: v for k, v in (d or {}).items() if k not in drop and v not in (None, "", [])}
        return {"ok": True, "ticker": t, "fund": clean(master),
                "company_data": clean(self._safe("etf_company_data", lambda: first("etf_company_data"), default={})),
                "expenses": clean(self._safe("etf_expenses", lambda: first("etf_expenses"), default={})),
                "price": self._safe("price snapshot", lambda: self._snapshot(t))}

    def etf_holdings(self, user_id, ticker: str, top_n: int = 10) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        top_n = max(1, min(int(top_n), 25))

        def run():
            rows = self._rows(self.fx.table("etf_holdings")
                              .select("security_symbol, security_name, portfolio_pct, as_of")
                              .eq("ticker", t).order("portfolio_pct", desc=True).limit(top_n))
            if not rows:
                return refusal(t, "no_holdings", f"No holdings are stored for {t}.")
            return {"ok": True, "ticker": t, "as_of": rows[0].get("as_of"),
                    "top_holdings": [{"symbol": r.get("security_symbol"), "name": r.get("security_name"),
                                      "weight_pct": r.get("portfolio_pct")} for r in rows]}
        return self._safe("ETF holdings", run)

    def etf_changes(self, user_id, ticker: str, days: int = 30) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        days = max(1, min(int(days), MAX_EVENT_DAYS))
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        def changes():
            return [{"action": r.get("action"), "symbol": r.get("security_symbol"),
                     "name": r.get("security_name"), "weight_prev": r.get("weight_prev"),
                     "weight_new": r.get("weight_new"), "detected_at": r.get("detected_at")}
                    for r in self._rows(self.fx.table("etf_holdings_changes")
                                        .select("action, security_symbol, security_name, weight_prev, "
                                                "weight_new, detected_at")
                                        .eq("ticker", t).gte("detected_at", since)
                                        .order("detected_at", desc=True).limit(30))]

        def fund_events():
            skip = {"NO_MATCH", "NO_DOC", "ERROR", "NO_SERIES"}
            return [{"filing_date": r.get("filing_date"), "form": r.get("form"),
                     "event_type": r.get("event_type"), "excerpt": (r.get("excerpt") or "")[:500],
                     "url": r.get("filing_url")}
                    for r in self._rows(self.fx.table("etf_fund_events")
                                        .select("filing_date, form, event_type, excerpt, filing_url, status")
                                        .contains("tickers", [t]).gte("created_at", since)
                                        .order("created_at", desc=True).limit(10))
                    if (r.get("status") or "") not in skip]

        return {"ok": True, "ticker": t, "days": days,
                "holdings_changes": self._safe("ETF holdings changes", changes, default=[]),
                "fund_events": self._safe("ETF fund events", fund_events, default=[]),
                "alerts": self._safe("FinXray events", lambda: self._events(t, days), default=[])}
