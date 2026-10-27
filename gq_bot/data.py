"""
Read-only data layer.

SCOPE RULE: a user can ask only about tickers on THEIR OWN FinXray watchlist.
Every public lookup that takes a ticker goes through `_scoped()`, so the LLM
cannot reach an out-of-scope ticker by forgetting a rule. It lives here and not
in the prompt because a prompt instruction is a request, and a function that
refuses is a guarantee.

CONTEXT is the deliberate exception. To answer a question about a watchlisted
ticker properly the bot needs reference points: the S&P 500 and other market
benchmarks, the sector median, and industry peers for valuation. Those figures
appear only inside an answer about a watchlisted ticker; no tool answers a
standalone question about a non-watchlisted one.

Sources
  FinXray US   users, watchlists, alerts, etf_master, etf_company_data,
               etf_expenses, etf_holdings, etf_holdings_changes, etf_fund_events;
               mutual funds and fund reference data are read in funds.py
  Heatmap      instruments, returns_latest, price_history, fundamentals
               (raw FMP statements + profile), us_stock_fundamentals (legacy
               fallback), tech_trend_identification (GQ dashboard signals)

Every source read is wrapped so that one missing table or renamed column turns
into a `source_error` for that piece, never an exception that kills the answer.
Nothing here writes.
"""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import analysis, financials as fin, technicals as tech

from .funds import FundsMixin
from .portfolio import PortfolioMixin

log = logging.getLogger(__name__)

US_UNIVERSES = ("us_stocks", "us_etfs")
RET_COLUMNS = ("ret_1d", "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_ytd",
               "ret_1y", "ret_5y", "ret_10y", "ret_all")
MARKET_BENCHMARK = "SPY"
BENCHMARKS = ("SPY", "QQQ", "IWM", "DIA")   # context only, never answered about on their own
MAX_EVENT_DAYS = 365
MAX_EVENTS = 25
MAX_HISTORY_DAYS = 3650
MAX_COMPARE = 5
CHUNK = 100
PAGE = 1000
TECH_LOOKBACK_DAYS = 1100         # ~755 sessions: a 200-day average with ~2 years of cross history
RANK_TECH_DAYS = 560              # rankings only need RSI / 200-day / 52-week, so a shorter read
PEERS = 10

INST_COLS = ("id, symbol, name, universe, sector, industry, market_cap_usd, "
             "shares_outstanding, expense_ratio, etf_category")
FUND_COLS = ("income_statement, balance_sheet, cash_flow, income_statement_q, balance_sheet_q, "
             "cash_flow_q, reported_currency, updated_at")
FUND_Q_COLS = "instrument_id, income_statement_q, balance_sheet_q, cash_flow_q, income_statement, reported_currency"

# Event categories -> words matched against the alert's type, source and feature.
EVENT_CATEGORIES = {
    "earnings": ("earning", "transcript", "eps", "guidance", "result"),
    "insider": ("insider", "form 4", "form4", "form_4"),
    "analyst": ("rating", "analyst", "target", "upgrade", "downgrade"),
    "sec_filing": ("sec", "8-k", "10-k", "10-q", "13d", "13g", "s-1", "edgar", "filing"),
    "news": ("news", "press"),
    "technical": ("technical", "cross", "rsi", "volume", "breakout", "52", "macd", "gmma"),
}

RANK_METRICS = {
    # returns_latest
    "return_1d": "ret_1d", "return_1w": "ret_1w", "return_1m": "ret_1m", "return_3m": "ret_3m",
    "return_6m": "ret_6m", "return_ytd": "ret_ytd", "return_1y": "ret_1y", "return_5y": "ret_5y",
    "market_cap": "market_cap",
    # technicals (computed from bars)
    "rsi_14": "tech", "pct_below_52w_high": "tech", "volatility_1y": "tech", "vs_sma_200": "tech",
    # fundamentals (stocks)
    "pe_ttm": "fund", "price_to_sales": "fund", "ev_to_ebitda": "fund", "fcf_yield_pct": "fund",
    "dividend_yield_pct": "fund", "revenue_growth_yoy": "fund", "net_margin": "fund", "roe": "fund",
    "debt_to_equity": "fund",
}


def norm(ticker: str) -> str:
    return (ticker or "").strip().upper().lstrip("$")


def refusal(ticker: str, reason: str, message: str, **extra) -> Dict[str, Any]:
    return {"ok": False, "ticker": ticker, "reason": reason, "message": message, **extra}


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


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


_NOISE = re.compile(r"\bCASH\b|MONEY MARKET|TRSY|TREASURY (OBLIG|BILLS?)|\bT-BILLS?\b|"
                    r"GOVERNMENT & AGENCY|GOVT.*PORTFOLIO|\bFUTURES?\b|\bFUT\b|\bFX\b|\bSWAPS?\b|"
                    r"\bFORWARDS?\b|COLLATERAL|\bREPO\b|MKTLIQ|LIQUIDITY FUND|CASH MGMT|"
                    r"^-[A-Z]{3} CASH-$", re.I)


def net_holding_changes(rows: List[Dict[str, Any]], top: int = 25) -> Dict[str, Any]:
    """Collapse raw add/remove events (oldest first) into what actually changed over the window."""
    per: Dict[str, Dict[str, Any]] = {}
    noise = 0
    for r in sorted(rows, key=lambda x: str(x.get("detected_at") or "")):
        name = (r.get("security_name") or "").strip()
        sym = (r.get("security_symbol") or "").strip()
        if _NOISE.search(name) or _NOISE.search(sym):
            noise += 1
            continue
        act = (r.get("action") or "").lower()
        kind = "add" if "add" in act else "remove" if "remov" in act else None
        if not kind:
            continue
        key = re.sub(r"[^a-z0-9]", "", name.lower()) or sym.lower()   # renames keep one key
        e = per.setdefault(key, {"name": name or sym, "symbol": sym, "first": kind, "last": kind,
                                 "first_at": r.get("detected_at")})
        e.update(last=kind, last_at=r.get("detected_at"), symbol=sym or e["symbol"])
        if r.get("weight_new") is not None:
            e["weight_new"] = r.get("weight_new")
    added = [e for e in per.values() if e["first"] == "add" and e["last"] == "add"]
    removed = [e for e in per.values() if e["first"] == "remove" and e["last"] == "remove"]
    churn = [e for e in per.values() if e["first"] != e["last"]]

    def show(es):
        es = sorted(es, key=lambda e: -(e.get("weight_new") or 0))
        return [{"name": e["name"], "symbol": e["symbol"], "date": str(e.get("last_at") or "")[:10],
                 **({"weight_pct": e["weight_new"]} if e.get("weight_new") is not None else {})} for e in es[:top]]
    return {"net_added_count": len(added), "net_removed_count": len(removed),
            "churned_count": len(churn), "cash_and_derivative_lines_ignored": noise,
            "net_added": show(added), "net_removed": show(removed),
            "tracked_from": str(rows[0].get("detected_at"))[:10] if rows else None}


class GQData(FundsMixin, PortfolioMixin):
    BENCH = BENCHMARKS

    def __init__(self, finxray, heatmap=None, clock: Callable[[], float] = time.monotonic,
                 today: Callable[[], date] = date.today, workers: int = 6):
        self.fx = finxray
        self.hm = heatmap or finxray
        self.today = today
        self.workers = workers
        self._wl = _TTLCache(60, clock)
        self._inst = _TTLCache(600, clock)
        self._sector = _TTLCache(900, clock)
        self._fund = _TTLCache(1800, clock)
        self._bars_cache = _TTLCache(600, clock)
        self._peers = _TTLCache(1800, clock)
        self._mfc = _TTLCache(1800, clock)

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

    def _map(self, fn, items):
        if self.workers <= 1 or len(items) <= 1:
            return [fn(i) for i in items]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(fn, items))

    def _window(self, days: Optional[int], start_date: Optional[str], end_date: Optional[str],
                default_days: int = 30) -> Tuple[date, date]:
        end = _parse_date(end_date) or self.today()
        start = _parse_date(start_date)
        if start is None:
            d = max(1, min(int(days or default_days), MAX_HISTORY_DAYS))
            start = end - timedelta(days=d)
        if start > end:
            start, end = end, start
        return start, end

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
            # LIKE, and a Telegram username can contain `_`, so escape them AND
            # confirm an exact match: this can never resolve to a different person.
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

    def _instruments_for(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        found: Dict[str, Dict[str, Any]] = {}
        missing = []
        for t in tickers:
            hit = self._inst.get(t)
            if hit is not None:
                if hit:
                    found[t] = hit
            else:
                missing.append(t)
        for i in range(0, len(missing), CHUNK):
            part = missing[i:i + CHUNK]
            rows = self._safe("instruments", lambda p=part: self._rows(
                self.hm.table("instruments").select(INST_COLS)
                .in_("symbol", p).in_("universe", list(US_UNIVERSES)).eq("is_active", True)), default=[])
            got = {norm(r["symbol"]): r for r in rows}
            for t in part:
                self._inst.put(t, got.get(t) or {})
                if t in got:
                    found[t] = got[t]
        return found

    def watchlist_details(self, user_id) -> List[Dict[str, Any]]:
        """Watchlist tickers with name and type, for the prompt's ticker resolution."""
        tickers = self.watchlist(user_id)
        found = self._instruments_for(tickers)
        mfs = self._safe("mf_master", lambda: self._mf_many([t for t in tickers if t not in found]), default={})
        out = []
        for t in tickers:
            if t in found:
                out.append({"ticker": t, "name": found[t].get("name"),
                            "type": {"us_etfs": "etf", "us_stocks": "stock"}.get(found[t].get("universe"), "unknown")})
            elif t in mfs:
                m = mfs[t]
                out.append({"ticker": t, "type": "mutual fund",
                            "name": " ".join(x for x in (m.get("series_name"), m.get("class_name")) if x) or None})
            else:
                out.append({"ticker": t, "name": None, "type": "unknown"})
        return out

    def _scoped(self, user_id, ticker: str) -> Tuple[Optional[str], Optional[Dict]]:
        t = norm(ticker)
        if not t:
            return None, refusal(t, "no_ticker", "Which ticker do you mean?")
        if t in self.watchlist(user_id):
            return t, None
        return None, refusal(
            t, "not_on_watchlist",
            f"{t} is not on your watchlist. Add it in the GQ FinXray bot with /add {t} "
            f"and then ask me again.")

    def _scoped_or_benchmark(self, user_id, ticker: str):
        t = norm(ticker)
        if t in BENCHMARKS:
            return t, None
        return self._scoped(user_id, t)

    # ── heatmap: instruments + price ──────────────────────────────────────────
    def _instrument(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self._instruments_for([ticker]).get(ticker)

    def _returns(self, instrument_id) -> Optional[Dict[str, Any]]:
        rows = self._rows(self.hm.table("returns_latest")
                          .select("price_usd, market_cap, as_of, " + ", ".join(RET_COLUMNS))
                          .eq("instrument_id", instrument_id).limit(1))
        return rows[0] if rows else None

    def _returns_many(self, ids: List[str], cols: str) -> Dict[str, Dict[str, Any]]:
        out = {}
        for i in range(0, len(ids), CHUNK):
            for r in self._rows(self.hm.table("returns_latest").select(f"instrument_id, {cols}")
                                .in_("instrument_id", ids[i:i + CHUNK])):
                out[r["instrument_id"]] = r
        return out

    def _snapshot(self, t: str) -> Dict[str, Any]:
        inst = self._instrument(t)
        if not inst:
            return refusal(t, "no_price_data", f"The heatmap has no active US stock or ETF data for {t} "
                                               f"(mutual funds and some funds are not covered).")
        r = self._returns(inst["id"])
        base = {"ticker": t, "name": inst.get("name"),
                "type": "etf" if inst.get("universe") == "us_etfs" else "stock",
                "sector": inst.get("sector"), "industry": inst.get("industry"),
                "market_cap_usd": inst.get("market_cap_usd") or (r or {}).get("market_cap")}
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

    def _bars(self, instrument_id, days: Optional[int] = None, start: Optional[date] = None,
              end: Optional[date] = None, ohlcv: bool = False) -> List[Dict[str, Any]]:
        """Daily bars oldest first, paged past PostgREST's 1,000-row cap."""
        end = end or self.today()
        start = start or end - timedelta(days=int(days or 30))
        key = (instrument_id, start.isoformat(), end.isoformat(), ohlcv)
        hit = self._bars_cache.get(key)
        if hit is not None:
            return hit
        cols = "date, open, high, low, close, volume" if ohlcv else "date, close"
        rows, page = [], 0
        while True:
            chunk = self._rows(self.hm.table("price_history").select(cols)
                               .eq("instrument_id", instrument_id)
                               .gte("date", start.isoformat()).lte("date", end.isoformat())
                               .order("date", desc=False).range(page * PAGE, page * PAGE + PAGE - 1))
            rows.extend(chunk)
            if len(chunk) < PAGE:
                break
            page += 1
        bars = sorted(({"date": str(b["date"])[:10], **{k: b.get(k) for k in
                        ("open", "high", "low", "close", "volume") if k in b}} for b in rows),
                      key=lambda b: b["date"])
        return self._bars_cache.put(key, bars)

    def price_history(self, user_id, ticker: str, days: int = 30, start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        start, end = self._window(days, start_date, end_date)

        def run():
            inst = self._instrument(t)
            if not inst:
                return refusal(t, "no_price_data", f"The heatmap has no active US price data for {t}.")
            bars = self._bars(inst["id"], start=start, end=end)
            out = {"ok": True, "ticker": t, "from": start.isoformat(), "to": end.isoformat(),
                   "summary": analysis.summarise_bars(bars)}
            if len(bars) <= 30:
                out["bars"] = bars
            else:  # long windows are summarised plus sampled, not dumped
                step = max(1, len(bars) // 24)
                out["sampled_closes"] = bars[::step][-24:] + ([bars[-1]] if bars[::step][-1] != bars[-1] else [])
            return out
        return self._safe("price history", run)

    # ── heatmap: fundamentals ─────────────────────────────────────────────────
    def _fund_store(self, t: str, inst: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        hit = self._fund.get(t)
        if hit is not None:
            return hit or None

        def primary():
            rows = self._rows(self.hm.table("fundamentals").select(FUND_COLS)
                              .eq("instrument_id", inst["id"]).limit(1))
            return rows[0] if rows and (rows[0].get("income_statement") or rows[0].get("income_statement_q")) else None

        def legacy():
            rows = self._rows(self.hm.table("us_stock_fundamentals").select("data, updated_at")
                              .eq("ticker", t).limit(1))
            if not rows:
                return None
            return {**(rows[0].get("data") or {}), "updated_at": rows[0].get("updated_at"),
                    "reported_currency": "USD"}

        row = self._safe("fundamentals", primary, default={}) or self._safe("legacy fundamentals", legacy, default={})
        return self._fund.put(t, row or {}) or None

    def _profile(self, inst: Dict[str, Any]) -> Dict[str, Any]:
        rows = self._rows(self.hm.table("fundamentals").select("profile, description")
                          .eq("instrument_id", inst["id"]).limit(1))
        return rows[0] if rows else {}

    def _stock_fin(self, user_id, ticker: str):
        t, no = self._scoped(user_id, ticker)
        if no:
            return None, None, None, no
        inst = self._instrument(t)
        if not inst:
            return t, None, None, refusal(t, "no_price_data", f"The heatmap has no active US data for {t}.")
        if inst.get("universe") == "us_etfs":
            return t, inst, None, refusal(t, "is_etf", f"{t} is an ETF, so it has no company financial "
                                          f"statements. Use the ETF profile, holdings or risk tools instead.")
        store = self._fund_store(t, inst)
        if not store:
            return t, inst, None, refusal(t, "no_fundamentals", f"No financial statements are stored for {t}.")
        return t, inst, store, None

    def financials(self, user_id, ticker: str) -> Dict[str, Any]:
        """Profitability, growth, balance sheet, cash flow, returns on capital, trends."""
        t, inst, store, no = self._stock_fin(user_id, ticker)
        if no:
            return no

        def run():
            res = fin.analyse(fin.periods_from_store(store))
            res.update({"ticker": t, "name": inst.get("name"), "sector": inst.get("sector"),
                        "industry": inst.get("industry"),
                        "reported_currency": store.get("reported_currency") or "USD",
                        "data_updated_at": store.get("updated_at"),
                        "units": "absolute amounts in reported currency; ratios as noted"})
            return res
        return self._safe("financials", run)

    # kept for the old tool name
    fundamentals = financials

    def _valuation_of(self, inst: Dict[str, Any], store: Dict[str, Any],
                      price: Optional[float], mcap: Optional[float]) -> Dict[str, Any]:
        cur = (store.get("reported_currency") or "USD").upper()
        if cur != "USD":
            return {"ok": False, "reason": "non_usd_statements",
                    "message": f"Statements are reported in {cur}, so USD price multiples would be wrong."}
        return fin.valuation(fin.periods_from_store(store), mcap, price)

    def valuation(self, user_id, ticker: str) -> Dict[str, Any]:
        """Multiples, set against industry peers and the stock's own sector."""
        t, inst, store, no = self._stock_fin(user_id, ticker)
        if no:
            return no

        def run():
            r = self._returns(inst["id"]) or {}
            mcap = inst.get("market_cap_usd") or r.get("market_cap")
            v = self._valuation_of(inst, store, r.get("price_usd"), mcap)
            out = {"ok": bool(v.get("ok")), "ticker": t, "name": inst.get("name"),
                   "price_usd": r.get("price_usd"), "as_of": r.get("as_of"), "valuation": v}
            peers = self._safe("peer set", lambda: self._peer_set(t, inst), default={})
            if peers.get("peers"):
                out["peer_comparison"] = {
                    "peer_group": peers.get("group"),
                    "peers_used": [p["ticker"] for p in peers["peers"]],
                    "peer_medians": peers.get("medians"),
                    "note": "Peers are the closest-sized companies in the same industry; context only."}
            return out
        return self._safe("valuation", run)

    def _peer_set(self, t: str, inst: Dict[str, Any]) -> Dict[str, Any]:
        """The PEERS companies nearest in size within the same industry (sector as fallback)."""
        hit = self._peers.get(t)
        if hit is not None:
            return hit
        group, rows = None, []
        for field in ("industry", "sector"):
            if not inst.get(field):
                continue
            rows = [r for r in self._rows(self.hm.table("instruments")
                                          .select("id, symbol, name, market_cap_usd")
                                          .eq("universe", "us_stocks").eq("is_active", True)
                                          .eq(field, inst[field]).limit(400))
                    if norm(r["symbol"]) != t and (r.get("market_cap_usd") or 0) > 0]
            if len(rows) >= 4:
                group = {"by": field, "name": inst[field], "size": len(rows) + 1}
                break
        if not rows:
            return self._peers.put(t, {"peers": []})
        own = inst.get("market_cap_usd") or 0
        if own > 0:
            rows.sort(key=lambda r: abs(math.log(r["market_cap_usd"] / own)))
        else:
            rows.sort(key=lambda r: -r["market_cap_usd"])
        chosen = rows[:PEERS]
        ids = [r["id"] for r in chosen]
        rets = self._returns_many(ids, "price_usd, market_cap, ret_1m, ret_3m, ret_ytd, ret_1y")
        funds = {f["instrument_id"]: f for f in self._rows(
            self.hm.table("fundamentals").select(FUND_Q_COLS).in_("instrument_id", ids))}
        peers = []
        for r in chosen:
            ret = rets.get(r["id"]) or {}
            row = {"ticker": norm(r["symbol"]), "name": r.get("name"), "market_cap_usd": r.get("market_cap_usd"),
                   **{k: ret.get(k) for k in ("ret_1m", "ret_3m", "ret_ytd", "ret_1y")}}
            f = funds.get(r["id"])
            if f:
                periods = fin.periods_from_store(f)
                if (f.get("reported_currency") or "USD").upper() == "USD":
                    v = fin.valuation(periods, r.get("market_cap_usd"), ret.get("price_usd"))
                    row.update({k: v.get(k) for k in ("pe_ttm", "price_to_sales", "ev_to_ebitda",
                                                      "fcf_yield_pct", "dividend_yield_pct")})
                a = fin.analyse(periods)
                if a.get("ok"):
                    row["net_margin"] = a["profitability_pct"]["net_margin"]
                    row["operating_margin"] = a["profitability_pct"]["operating_margin"]
                    row["revenue_growth_yoy"] = a["growth_yoy_pct"]["revenue"]
                    row["roe"] = a["returns_on_capital_pct"]["roe"]
            peers.append(row)
        keys = ("pe_ttm", "price_to_sales", "ev_to_ebitda", "fcf_yield_pct", "dividend_yield_pct",
                "net_margin", "operating_margin", "revenue_growth_yoy", "roe",
                "ret_1m", "ret_3m", "ret_ytd", "ret_1y")
        return self._peers.put(t, {"group": group, "peers": peers, "medians": fin.summarise_peers(peers, keys)})

    def peer_comparison(self, user_id, ticker: str) -> Dict[str, Any]:
        """The stock beside its closest-sized industry peers: size, returns, valuation, margins, growth."""
        t, inst, store, no = self._stock_fin(user_id, ticker)
        if no and (not no.get("reason") == "no_fundamentals"):
            return no

        def run():
            peers = self._peer_set(t, inst)
            if not peers.get("peers"):
                return refusal(t, "no_peers", f"I couldn't find industry peers for {t}.")
            r = self._returns(inst["id"]) or {}
            subject = {"ticker": t, "name": inst.get("name"), "market_cap_usd": inst.get("market_cap_usd"),
                       **{k: r.get(k) for k in ("ret_1m", "ret_3m", "ret_ytd", "ret_1y")}}
            if store:
                periods = fin.periods_from_store(store)
                v = self._valuation_of(inst, store, r.get("price_usd"), inst.get("market_cap_usd"))
                if v.get("ok"):
                    subject.update({k: v.get(k) for k in ("pe_ttm", "price_to_sales", "ev_to_ebitda",
                                                          "fcf_yield_pct", "dividend_yield_pct")})
                a = fin.analyse(periods)
                if a.get("ok"):
                    subject.update({"net_margin": a["profitability_pct"]["net_margin"],
                                    "operating_margin": a["profitability_pct"]["operating_margin"],
                                    "revenue_growth_yoy": a["growth_yoy_pct"]["revenue"],
                                    "roe": a["returns_on_capital_pct"]["roe"]})
            ranks = {}
            for k in ("pe_ttm", "ev_to_ebitda", "net_margin", "revenue_growth_yoy", "roe", "ret_ytd", "ret_1y"):
                ranks[k] = fin.percentile_rank(subject.get(k), [p.get(k) for p in peers["peers"]])
            return {"ok": True, "ticker": t, "peer_group": peers["group"], "subject": subject,
                    "peers": peers["peers"], "peer_medians": peers["medians"],
                    "subject_percentile_vs_peers": ranks,
                    "note": "Percentile = share of peers below the subject on that metric. Peers are context "
                            "for the subject only."}
        return self._safe("peer comparison", run)

    def company_profile(self, user_id, ticker: str) -> Dict[str, Any]:
        """What the company or fund is: business description, sector, industry, size, key facts."""
        t, no = self._scoped(user_id, ticker)
        if no:
            return no

        def run():
            inst = self._instrument(t)
            if not inst:
                return refusal(t, "no_data", f"The heatmap has no active US data for {t}.")
            out = {"ok": True, "ticker": t, "name": inst.get("name"),
                   "type": "etf" if inst.get("universe") == "us_etfs" else "stock",
                   "sector": inst.get("sector"), "industry": inst.get("industry"),
                   "market_cap_usd": inst.get("market_cap_usd"),
                   "shares_outstanding": inst.get("shares_outstanding")}
            if out["type"] == "etf":
                out.update({"expense_ratio": inst.get("expense_ratio"), "etf_category": inst.get("etf_category")})
            about = self._safe("company notes", lambda: self._rows(
                self.hm.table("instruments").select("company_about, about_source_year")
                .eq("id", inst["id"]).limit(1)), default=[])
            if about and about[0].get("company_about"):
                out["gq_company_notes"] = about[0]["company_about"][:3000]
                out["gq_notes_year"] = about[0].get("about_source_year")
            prof = self._safe("profile", lambda: self._profile(inst), default={})
            p = prof.get("profile") or {}
            if p:
                pick = ("ceo", "fullTimeEmployees", "ipoDate", "website", "country", "exchange",
                        "exchangeShortName", "beta", "lastDividend", "lastDiv", "range",
                        "averageVolume", "volAvg", "isAdr")
                out["profile"] = {k: p.get(k) for k in pick if p.get(k) not in (None, "")}
            desc = prof.get("description") or p.get("description")
            if desc and "gq_company_notes" not in out:
                out["description"] = desc[:2000]
            out["price"] = self._snapshot(t)
            return out
        return self._safe("company profile", run)

    # ── technicals + risk ─────────────────────────────────────────────────────
    def _gq_signals(self, inst_id) -> Dict[str, Any]:
        # The table holds one row per timeframe (D/W/M/Q/Y); a daily setup reads the daily row only.
        rows = self._rows(self.hm.table("tech_trend_identification").select("*")
                          .eq("instrument_id", inst_id).eq("timeframe", "D").limit(1))
        drop = {"instrument_id", "symbol", "name", "universe", "market_cap_usd", "id", "created_at",
                "updated_at", "timeframe", "isin"}
        out = [{k: v for k, v in r.items() if k not in drop and v not in (None, "")} for r in rows]
        relabel = {"golden cross": "50-day above 200-day (state, not a dated cross)",
                   "death cross": "50-day below 200-day (state, not a dated cross)"}
        for r in out:
            v = str(r.get("dma50_vs_200dma") or "")
            if v.lower() in relabel:
                r["dma50_vs_200dma"] = relabel[v.lower()]
            r["note"] = ("GQ dashboard daily STATES only, as of the as_of date. They carry no cross "
                         "dates; dates come from last_50_200_cross / ma_50_200_summary.")
        return out

    def _gq_indicators(self, inst_id) -> Dict[str, Any]:
        """Extra daily signals the dashboard computes: RSI trend/divergence, candlestick patterns,
        all-time-high distance and GMMA compression/breakouts."""
        out: Dict[str, Any] = {}

        def first(table, cols, **eqs):
            q = self.hm.table(table).select(cols).eq("instrument_id", inst_id)
            for k, v in eqs.items():
                q = q.eq(k, v)
            return self._rows(q.limit(5))
        r = self._safe("RSI table", lambda: first("tech_indicators_rsi", "rsi_trend, rsi_divergence, as_of_date",
                                                  timeframe="D"), default=[])
        if r and (r[0].get("rsi_trend") or r[0].get("rsi_divergence")):
            out["rsi"] = {k: v for k, v in r[0].items() if v}
        c = self._safe("candlestick table", lambda: first("tech_indicators_candlestick",
                                                          "pattern_name, pattern_type, confidence, as_of_date",
                                                          timeframe="D"), default=[])
        if c:
            latest = max(str(x.get("as_of_date") or "") for x in c)
            out["candlestick_patterns"] = [{k: v for k, v in x.items() if v is not None}
                                           for x in c if str(x.get("as_of_date") or "") == latest]
        a = self._safe("ATH table", lambda: first("tech_indicators_ath", "ath, dist_from_ath_pct, is_new_ath, as_of_date"),
                       default=[])
        if a:
            out["all_time_high"] = {k: v for k, v in a[0].items() if v is not None}
        g = self._safe("GMMA table", lambda: self._rows(
            self.hm.table("gmma_alerts").select("timeframe, grp, strength, breakout, breakout_dir, bars_compressed, as_of_date")
            .eq("instrument_id", inst_id).order("as_of_date", desc=True).limit(4)), default=[])
        if g:
            out["gmma"] = [{k: v for k, v in x.items() if v not in (None, False)} for x in g]
        return out

    def technicals(self, user_id, ticker: str) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no

        def run():
            inst = self._instrument(t)
            if not inst:
                return refusal(t, "no_price_data", f"The heatmap has no active US price data for {t}.")
            bars = self._bars(inst["id"], days=TECH_LOOKBACK_DAYS, ohlcv=True)
            res = tech.technicals(bars)
            res["ticker"] = t
            sig = self._safe("GQ trend signals", lambda: self._gq_signals(inst["id"]), default=[])
            if sig:
                res["gq_dashboard_trend_signals"] = sig
            extra = self._safe("GQ indicator tables", lambda: self._gq_indicators(inst["id"]), default={})
            if extra:
                res["gq_dashboard_indicators"] = extra
            return res
        return self._safe("technicals", run)

    def risk(self, user_id, ticker: str, benchmark: str = MARKET_BENCHMARK, days: int = 365,
             start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        b, no = self._scoped_or_benchmark(user_id, benchmark or MARKET_BENCHMARK)
        if no:
            return no
        start, end = self._window(days, start_date, end_date, default_days=365)

        def run():
            inst, binst = self._instrument(t), self._instrument(b)
            if not inst:
                return refusal(t, "no_price_data", f"The heatmap has no active US price data for {t}.")
            bars = self._bars(inst["id"], start=start, end=end)
            bench = self._bars(binst["id"], start=start, end=end) if binst else []
            res = tech.risk(bars, bench, b)
            res["ticker"] = t
            return res
        return self._safe("risk", run)

    # ── FinXray: events ───────────────────────────────────────────────────────
    def _events(self, t: str, start: datetime, end: Optional[datetime] = None, limit: int = MAX_EVENTS,
                min_impact: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        q = (self.fx.table("alerts")
             .select("id, summary, impact, source, filing_type, link, created_at, extra")
             .eq("ticker", t).gte("created_at", start.isoformat()))
        if end is not None:
            q = q.lte("created_at", end.isoformat())
        rows = self._rows(q.order("created_at", desc=True).limit(limit * (3 if category else 1)))
        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        floor = rank.get((min_impact or "").upper(), 0)
        words = EVENT_CATEGORIES.get((category or "").lower())
        out = []
        for r in rows:
            if rank.get((r.get("impact") or "LOW").upper(), 0) < floor:
                continue
            feature = (r.get("extra") or {}).get("feature_name") if isinstance(r.get("extra"), dict) else None
            if words:
                hay = " ".join(str(x or "") for x in (r.get("filing_type"), r.get("source"), feature)).lower()
                if not any(w in hay for w in words):
                    continue
            out.append({"when": r.get("created_at"), "summary": r.get("summary"),
                        "impact": r.get("impact"), "type": r.get("filing_type"),
                        "feature": feature, "source": r.get("source"), "link": r.get("link")})
            if len(out) >= limit:
                break
        return out

    def _now(self) -> datetime:
        """'Now' from the injected calendar (end of that UTC day), never the wall clock."""
        d = self.today()
        return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)

    def _dt_window(self, days, start_date, end_date, default_days):
        start, end = self._window(days, start_date, end_date, default_days)
        s = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        e = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)
        return min(s, e), e

    def events(self, user_id, ticker: str, days: int = 14, min_impact: Optional[str] = None,
               category: Optional[str] = None, start_date: Optional[str] = None,
               end_date: Optional[str] = None) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        days = max(1, min(int(days or 14), MAX_EVENT_DAYS))
        s, e = self._dt_window(days, start_date, end_date, 14)
        now = self._now()
        if s < now - timedelta(days=MAX_EVENT_DAYS):
            s = now - timedelta(days=MAX_EVENT_DAYS)
        return self._safe("FinXray events", lambda: {
            "ok": True, "ticker": t, "from": s.date().isoformat(), "to": e.date().isoformat(),
            "category": category,
            "events": self._events(t, s, e if (start_date or end_date) else None,
                                   min_impact=min_impact, category=category)})

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
                               .eq("sector", sector).range(page * PAGE, page * PAGE + PAGE - 1))
            ids.extend(r["id"] for r in chunk)
            if len(chunk) < PAGE:
                break
            page += 1
        vals = [r.get(col) for r in self._returns_many(ids, col).values()]
        return self._sector.put(key, {"sector": sector, **analysis.group_stats(vals)})

    def explain_move(self, user_id, ticker: str, days: int = 7, start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict[str, Any]:
        """Everything needed to explain a move: the price path, the biggest sessions with the
        events around each, and the sector + market over the same dates."""
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        custom = bool(start_date or end_date)
        start, end = self._window(days, start_date, end_date, default_days=7)
        span = (end - start).days or 1
        snap = self._safe("price snapshot", lambda: self._snapshot(t))
        out: Dict[str, Any] = {"ok": bool(snap.get("ok")), "ticker": t,
                               "window": {"from": start.isoformat(), "to": end.isoformat()},
                               "snapshot": snap}
        inst = self._safe("instrument", lambda: self._instrument(t), default={}) or {}
        bars: List[Dict[str, Any]] = []
        if inst:
            def path():
                nonlocal bars
                bars = self._bars(inst["id"], start=start, end=end, ohlcv=True)
                if len(bars) < 2:  # a 1-day window over a weekend has one bar
                    bars = self._bars(inst["id"], start=start - timedelta(days=7), end=end, ohlcv=True)
                return analysis.summarise_bars(bars)
            out["price_path"] = self._safe("price history", path)

        # News often lands a day or two before the session it moves.
        s = datetime(start.year, start.month, start.day, tzinfo=timezone.utc) - timedelta(days=3)
        # an open end for recent windows: an alert stamped in UTC can fall after the ET date
        e = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc) if custom else None
        events = self._safe("FinXray events", lambda: self._events(t, s, e, limit=30), default=[])
        out["events"] = events
        out["events_from"] = s.date().isoformat()

        # The biggest sessions, each with the events dated within a day before it.
        if len(bars) >= 2:
            big = tech.biggest_moves(bars, 3)

            def market_days():
                m = self._instrument(MARKET_BENCHMARK)
                if not m or t == MARKET_BENCHMARK:
                    return {}
                mb = self._bars(m["id"], start=date.fromisoformat(bars[0]["date"]) - timedelta(days=7), end=end)
                return {b["date"]: round((b["close"] / p["close"] - 1) * 100, 2)
                        for p, b in zip(mb, mb[1:]) if p.get("close") and b.get("close") is not None}
            spy_day = self._safe("market day moves", market_days, default={}) or {}
            for side in big.values():
                for m in side:
                    if m["date"] in spy_day:
                        m[f"{MARKET_BENCHMARK.lower()}_same_day_pct"] = spy_day[m["date"]]
                    d = date.fromisoformat(m["date"])
                    m["events_near"] = [ev["summary"] for ev in events
                                        if ev.get("when") and timedelta(0) <= d - date.fromisoformat(str(ev["when"])[:10])
                                        <= timedelta(days=2)][:3]
            out["key_sessions"] = big
            vols = [b.get("volume") for b in bars if b.get("volume")]
            if vols:
                out["avg_volume_in_window"] = round(sum(vols) / len(vols))

        stock_ret = (out.get("price_path") or {}).get("change_pct")

        def market():
            m = self._instrument(MARKET_BENCHMARK)
            mb = self._bars(m["id"], start=start if len(bars) < 2 else date.fromisoformat(bars[0]["date"]),
                            end=end) if m else []
            v = analysis.pct(mb[0]["close"], mb[-1]["close"]) if len(mb) >= 2 else None
            return {"benchmark": MARKET_BENCHMARK, "return_pct": v,
                    "stock_minus_market_pct": analysis.relative(stock_ret, v)}
        out["market_context"] = self._safe("market context", market)

        if snap.get("type") == "stock" and inst.get("sector") and not custom:
            col = analysis.window_column(span)
            sec = self._safe("sector context", lambda: self._sector_stats(inst["sector"], col))
            if isinstance(sec, dict) and sec.get("count"):
                sec["window_used"] = col
                sec["stock_on_same_basis_pct"] = (snap.get("returns_pct") or {}).get(col)
                sec["stock_minus_sector_pct"] = analysis.relative(sec["stock_on_same_basis_pct"],
                                                                  sec.get("median_pct"))
            out["sector_context"] = sec
        return out

    def compare(self, user_id, tickers: List[str]) -> Dict[str, Any]:
        seen, results = [], []
        for raw in tickers or []:
            t = norm(raw)
            if t and t not in seen:
                seen.append(t)
        for t in seen[:MAX_COMPARE]:
            res = self.snapshot(user_id, t)
            if res.get("ok") and res.get("type") == "stock":
                v = self.valuation(user_id, t)
                if v.get("ok"):
                    res["valuation"] = {k: v["valuation"].get(k) for k in
                                        ("pe_ttm", "price_to_sales", "ev_to_ebitda", "fcf_yield_pct",
                                         "dividend_yield_pct")}
            results.append(res)
        out = {"ok": any(r.get("ok") for r in results), "results": results}
        if len(seen) > MAX_COMPARE:
            out["note"] = f"Compared the first {MAX_COMPARE} tickers only."
        return out

    # ── the watchlist as a whole ──────────────────────────────────────────────
    def watchlist_overview(self, user_id, window: str = "1d") -> Dict[str, Any]:
        """How the whole watchlist is doing: movers, breadth, sector mix, recent high-impact events."""
        col = "ret_" + (window or "1d").lower().replace("ret_", "")
        if col not in RET_COLUMNS:
            col = "ret_1d"

        def run():
            tickers = self.watchlist(user_id)
            insts = self._instruments_for(tickers)
            rets = self._returns_many([i["id"] for i in insts.values()], "price_usd, as_of, " + col)
            rows = []
            for t, i in insts.items():
                r = rets.get(i["id"]) or {}
                if r.get(col) is not None:
                    rows.append({"ticker": t, "name": i.get("name"), "return_pct": r.get(col),
                                 "type": "etf" if i.get("universe") == "us_etfs" else "stock",
                                 "sector": i.get("sector"), "as_of": r.get("as_of")})
            rows.sort(key=lambda x: x["return_pct"], reverse=True)
            vals = [x["return_pct"] for x in rows]
            sectors: Dict[str, int] = {}
            for x in rows:
                if x["type"] == "stock":
                    sectors[x["sector"] or "Unknown"] = sectors.get(x["sector"] or "Unknown", 0) + 1
            since = self._now() - timedelta(days=7)
            covered = list(insts)
            hi = []
            for i in range(0, len(covered), CHUNK):
                hi.extend(self._safe("watchlist events", lambda p=covered[i:i + CHUNK]: self._rows(
                    self.fx.table("alerts").select("ticker, summary, impact, created_at, filing_type")
                    .in_("ticker", p).eq("impact", "HIGH").gte("created_at", since.isoformat())
                    .order("created_at", desc=True).limit(15)), default=[]))
            return {"ok": bool(rows), "window": col, "as_of": rows[0]["as_of"] if rows else None,
                    "covered": len(rows), "on_watchlist": len(tickers),
                    "not_covered": [t for t in tickers if t not in insts],
                    "breadth": {"up": sum(v > 0 for v in vals), "down": sum(v < 0 for v in vals),
                                "average_pct": round(sum(vals) / len(vals), 2) if vals else None,
                                "median_pct": analysis.group_stats(vals).get("median_pct")},
                    "top_gainers": rows[:5], "top_losers": list(reversed(rows[-5:])),
                    "stock_sector_mix": sectors,
                    "etf_count": sum(1 for x in rows if x["type"] == "etf"),
                    "high_impact_events_last_7d": [{"ticker": h.get("ticker"), "when": h.get("created_at"),
                                                    "summary": h.get("summary"), "type": h.get("filing_type")}
                                                   for h in hi[:15]]}
        return self._safe("watchlist overview", run)

    def rank_watchlist(self, user_id, metric: str, order: str = "desc", limit: int = 10,
                       type_filter: Optional[str] = None,
                       tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Rank the user's watchlist, or just the tickers named, by any supported metric."""
        metric = (metric or "").lower()
        kind = RANK_METRICS.get(metric)
        if not kind:
            return {"ok": False, "reason": "unknown_metric", "supported": sorted(RANK_METRICS)}
        wanted: List[str] = []
        refused: List[str] = []
        for raw in tickers or []:
            t, no = self._scoped(user_id, raw)
            if no:
                refused.append(str(raw).upper())
            elif t not in wanted:
                wanted.append(t)
        if tickers and not wanted:
            return {"ok": False, "reason": "not_on_watchlist", "tickers": refused}
        # a named set is ranked in full; the whole watchlist is cut to the top N
        limit = len(wanted) if wanted else max(1, min(int(limit or 10), 30))

        def run():
            insts = self._instruments_for(wanted or self.watchlist(user_id))
            if type_filter in ("stock", "etf"):
                want = "us_etfs" if type_filter == "etf" else "us_stocks"
                insts = {t: i for t, i in insts.items() if i.get("universe") == want}
            if kind == "fund":
                insts = {t: i for t, i in insts.items() if i.get("universe") == "us_stocks"}
            values: Dict[str, Any] = {}
            if kind not in ("tech", "fund"):
                rets = self._returns_many([i["id"] for i in insts.values()], kind)
                for t, i in insts.items():
                    values[t] = (rets.get(i["id"]) or {}).get(kind)
                    if kind == "market_cap" and i.get("market_cap_usd"):
                        values[t] = i["market_cap_usd"]
            elif kind == "tech":
                def one(item):
                    t, i = item
                    tr = tech.technicals(self._bars(i["id"], days=RANK_TECH_DAYS))
                    if not tr.get("ok"):
                        return t, None
                    return t, {"rsi_14": tr["rsi_14"], "pct_below_52w_high": tr["range_52w"]["pct_below_high"],
                               "volatility_1y": tr["volatility_ann_pct"]["1y"],
                               "vs_sma_200": tr["price_vs_ma_pct"]["vs_sma_200"]}[metric]
                values = dict(self._map(lambda it: self._safe("tech rank", lambda: one(it), default=(it[0], None)),
                                        list(insts.items())))
            else:
                ids = {i["id"]: t for t, i in insts.items()}
                id_list = list(ids)
                rets = self._returns_many(id_list, "price_usd, market_cap")
                for k in range(0, len(id_list), 20):
                    for f in self._rows(self.hm.table("fundamentals").select(FUND_Q_COLS)
                                        .in_("instrument_id", id_list[k:k + 20])):
                        t = ids.get(f["instrument_id"])
                        i, r = insts[t], rets.get(f["instrument_id"]) or {}
                        periods = fin.periods_from_store(f)
                        if metric in ("pe_ttm", "price_to_sales", "ev_to_ebitda", "fcf_yield_pct", "dividend_yield_pct"):
                            if (f.get("reported_currency") or "USD").upper() != "USD":
                                continue
                            values[t] = fin.valuation(periods, i.get("market_cap_usd") or r.get("market_cap"),
                                                      r.get("price_usd")).get(metric)
                        else:
                            a = fin.analyse(periods)
                            if a.get("ok"):
                                values[t] = {"revenue_growth_yoy": a["growth_yoy_pct"]["revenue"],
                                             "net_margin": a["profitability_pct"]["net_margin"],
                                             "roe": a["returns_on_capital_pct"]["roe"],
                                             "debt_to_equity": a["balance_sheet"]["debt_to_equity"]}[metric]
            ranked = sorted(((t, v) for t, v in values.items() if v is not None),
                            key=lambda x: x[1], reverse=(order != "asc"))
            out = {"ok": bool(ranked), "metric": metric, "order": order,
                   "ranked": [{"rank": n + 1, "ticker": t, "name": insts[t].get("name"), "value": v,
                               **({"zone": tech.rsi_zone(v)} if metric == "rsi_14" else {})}
                              for n, (t, v) in enumerate(ranked[:limit])],
                   "count_with_data": len(ranked), "count_considered": len(insts),
                   "no_data": [t for t in insts if values.get(t) is None][:20]}
            if wanted:
                out["requested"] = wanted
                out["not_covered"] = [t for t in wanted if t not in insts]
                out["not_on_watchlist"] = refused
            if metric == "rsi_14":
                out["rsi_zones"] = "overbought only above 70, oversold only below 30, otherwise neutral"
            return out
        return self._safe("watchlist ranking", run)

    # ── charts ────────────────────────────────────────────────────────────────
    def chart(self, user_id, tickers: List[str], days: int = 180, start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> Dict[str, Any]:
        wanted, refused = [], []
        for raw in (tickers or [])[:5]:
            t, no = self._scoped_or_benchmark(user_id, raw)
            (refused.append(no) if no else wanted.append(t))
        if not wanted:
            return refused[0] if refused else refusal("", "no_ticker", "Which ticker should I chart?")
        start, end = self._window(days, start_date, end_date, default_days=180)

        def run():
            from .charts import price_chart
            series = {}
            # a single ticker gets ~200 extra sessions so its averages are drawn from the first day
            warm = timedelta(days=300) if len(wanted) == 1 else timedelta(0)
            for t in wanted:
                inst = self._instrument(t)
                if inst:
                    bars = self._bars(inst["id"], start=start - warm, end=end)
                    if bars:
                        series[t] = bars
            if not series:
                return {"ok": False, "reason": "no_price_data", "message": "No price data to chart."}
            multi = len(series) > 1
            title = (" vs ".join(series) + " (rebased to 100)") if multi else f"{wanted[0]} price"
            png = price_chart(series, f"{title}, {start.isoformat()} to {end.isoformat()}", normalise=multi,
                              plot_from=start.isoformat())
            in_window = {t: [b for b in bars if b["date"] >= start.isoformat()] for t, bars in series.items()}
            changes = {t: analysis.pct(b[0]["close"], b[-1]["close"]) for t, b in in_window.items() if len(b) >= 2}
            return {"ok": True, "charted": list(series), "from": start.isoformat(), "to": end.isoformat(),
                    "change_pct": changes, "refused": refused, "_image": png,
                    "note": "Mention the chart in one line; do not describe how it is delivered."}
        return self._safe("chart", run)

    # ── FinXray: ETFs ─────────────────────────────────────────────────────────
    def etf_profile(self, user_id, ticker: str) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no

        def first(table):
            rows = self._rows(self.fx.table(table).select("*").eq("ticker", t).limit(1))
            return rows[0] if rows else None

        master = self._safe("etf_master", lambda: first("etf_master"), default={})
        inst = self._safe("instrument", lambda: self._instrument(t), default={}) or {}
        if not master and inst.get("universe") != "us_etfs":
            return refusal(t, "not_an_etf", f"{t} is not in the ETF universe.")
        drop = ("updated_at", "created_at", "id")
        clean = lambda d: {k: v for k, v in (d or {}).items() if k not in drop and v not in (None, "", [], {})}
        return {"ok": True, "ticker": t, "fund": clean(master),
                "heatmap": {k: inst.get(k) for k in ("name", "etf_category", "expense_ratio") if inst.get(k)},
                "company_data": clean(self._safe("etf_company_data", lambda: first("etf_company_data"), default={})),
                "expenses": clean(self._safe("etf_expenses", lambda: first("etf_expenses"), default={})),
                "price": self._safe("price snapshot", lambda: self._snapshot(t))}

    def _holdings(self, t: str, limit: int) -> List[Dict[str, Any]]:
        return self._rows(self.fx.table("etf_holdings")
                          .select("security_symbol, security_name, portfolio_pct, as_of")
                          .eq("ticker", t).order("portfolio_pct", desc=True).limit(limit))

    def etf_holdings(self, user_id, ticker: str, top_n: int = 10) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        top_n = max(1, min(int(top_n), 25))

        def run():
            rows = self._holdings(t, top_n)
            if not rows:
                return refusal(t, "no_holdings", f"No holdings are stored for {t}.")
            top = [{"symbol": r.get("security_symbol"), "name": r.get("security_name"),
                    "weight_pct": r.get("portfolio_pct")} for r in rows]
            return {"ok": True, "ticker": t, "as_of": rows[0].get("as_of"), "top_holdings": top,
                    "top_n_combined_weight_pct": round(sum(float(x["weight_pct"] or 0) for x in top), 2)}
        return self._safe("ETF holdings", run)

    def etf_overlap(self, user_id, ticker_a: str, ticker_b: str) -> Dict[str, Any]:
        """How much two ETFs on the watchlist hold in common."""
        a, no = self._scoped(user_id, ticker_a)
        if no:
            return no
        b, no = self._scoped(user_id, ticker_b)
        if no:
            return no

        def run():
            ha = {(r.get("security_symbol") or r.get("security_name")): float(r.get("portfolio_pct") or 0)
                  for r in self._holdings(a, 1000)}
            hb = {(r.get("security_symbol") or r.get("security_name")): float(r.get("portfolio_pct") or 0)
                  for r in self._holdings(b, 1000)}
            if not ha or not hb:
                return {"ok": False, "reason": "no_holdings",
                        "message": f"Holdings are missing for {a if not ha else b}."}
            common = set(ha) & set(hb)
            shared = sorted(({"symbol": s, f"weight_in_{a}": ha[s], f"weight_in_{b}": hb[s],
                              "overlap_weight": min(ha[s], hb[s])} for s in common),
                            key=lambda x: -x["overlap_weight"])
            return {"ok": True, "etfs": [a, b], "holdings_count": {a: len(ha), b: len(hb)},
                    "common_holdings": len(common),
                    "weight_overlap_pct": round(sum(x["overlap_weight"] for x in shared), 2),
                    "pct_of_a_names_in_b": round(len(common) / len(ha) * 100, 1),
                    "pct_of_b_names_in_a": round(len(common) / len(hb) * 100, 1),
                    "top_shared": shared[:10]}
        return self._safe("ETF overlap", run)

    def etf_exposure(self, user_id, symbol: str) -> Dict[str, Any]:
        """Which ETFs on the user's watchlist hold a given stock, and at what weight."""
        s = norm(symbol)

        def run():
            etfs = [t for t, i in self._instruments_for(self.watchlist(user_id)).items()
                    if i.get("universe") == "us_etfs"]
            if not etfs:
                return {"ok": False, "reason": "no_etfs", "message": "There are no ETFs on your watchlist."}
            rows = []
            for k in range(0, len(etfs), CHUNK):
                rows.extend(self._rows(self.fx.table("etf_holdings")
                                       .select("ticker, security_symbol, security_name, portfolio_pct, as_of")
                                       .in_("ticker", etfs[k:k + CHUNK]).eq("security_symbol", s)))
            hits = sorted(({"etf": r.get("ticker"), "weight_pct": r.get("portfolio_pct"), "as_of": r.get("as_of")}
                           for r in rows), key=lambda x: -float(x["weight_pct"] or 0))
            return {"ok": True, "stock": s, "held_by": hits, "etfs_checked": len(etfs),
                    "name": rows[0].get("security_name") if rows else None}
        return self._safe("ETF exposure", run)

    def _change_rows(self, tickers: List[str], since: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """Every holdings change in the window, oldest first, paged; not just the newest few."""
        out: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tickers}
        for k in range(0, len(tickers), CHUNK):
            page = 0
            while page < 20:
                chunk = self._rows(self.fx.table("etf_holdings_changes")
                                   .select("ticker, action, security_symbol, security_name, weight_prev, "
                                           "weight_new, detected_at")
                                   .in_("ticker", tickers[k:k + CHUNK]).gte("detected_at", since.isoformat())
                                   .order("detected_at", desc=False).range(page * PAGE, page * PAGE + PAGE - 1))
                for r in chunk:
                    out.setdefault(r.get("ticker"), []).append(r)
                if len(chunk) < PAGE:
                    break
                page += 1
        return out

    def watchlist_etf_changes(self, user_id, days: int = 90) -> Dict[str, Any]:
        """Net holdings changes for every ETF on the watchlist, one row per ETF."""
        days = max(1, min(int(days or 90), MAX_EVENT_DAYS))
        since = self._now() - timedelta(days=days)

        def run():
            etfs = [t for t, i in self._instruments_for(self.watchlist(user_id)).items()
                    if i.get("universe") == "us_etfs"]
            if not etfs:
                return {"ok": False, "reason": "no_etfs", "message": "There are no ETFs on your watchlist."}
            raw = {t: rows for t, rows in self._change_rows(etfs, since).items() if t in etfs}
            nets = {t: net_holding_changes(rows, top=5) for t, rows in raw.items() if rows}
            changed = sorted(((t, n) for t, n in nets.items() if n["net_added_count"] or n["net_removed_count"]),
                             key=lambda x: -(x[1]["net_added_count"] + x[1]["net_removed_count"]))
            firsts = [str(r[0].get("detected_at"))[:10] for r in raw.values() if r]
            recording_since = min(firsts) if firsts else None
            out = {"ok": True, "days": days, "window_from": since.date().isoformat(),
                   "recording_since": recording_since, "etfs_checked": len(etfs),
                   "changed": [{"etf": t, **n} for t, n in changed],
                   "changes_cancelled_out": sorted(t for t, n in nets.items()
                                                   if not (n["net_added_count"] or n["net_removed_count"])),
                   "nothing_recorded": sorted(t for t, r in raw.items() if not r),
                   "note": "Net of the window: a name added then removed counts as churn, and cash, "
                           "money-market, currency and futures lines are excluded."}
            if recording_since and recording_since > out["window_from"]:
                out["coverage_warning"] = (
                    f"FinXray has only recorded holdings changes since {recording_since}, so this covers "
                    f"{recording_since} onward, not the full {days} days. ETFs in nothing_recorded may not "
                    f"be tracked at all; do not describe them as unchanged.")
            return out
        return self._safe("watchlist ETF changes", run)

    def etf_changes(self, user_id, ticker: str, days: int = 30) -> Dict[str, Any]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return no
        days = max(1, min(int(days), MAX_EVENT_DAYS))
        since = self._now() - timedelta(days=days)

        def changes():
            return net_holding_changes(self._change_rows([t], since)[t])

        def fund_events():
            skip = {"NO_MATCH", "NO_DOC", "ERROR", "NO_SERIES"}
            return [{"filing_date": r.get("filing_date"), "form": r.get("form"),
                     "event_type": r.get("event_type"), "excerpt": (r.get("excerpt") or "")[:500],
                     "url": r.get("filing_url")}
                    for r in self._rows(self.fx.table("etf_fund_events")
                                        .select("filing_date, form, event_type, excerpt, filing_url, status")
                                        .contains("tickers", [t]).gte("created_at", since.isoformat())
                                        .order("created_at", desc=True).limit(10))
                    if (r.get("status") or "") not in skip]

        return {"ok": True, "ticker": t, "days": days,
                "holdings_changes": self._safe("ETF holdings changes", changes, default={}),
                "fund_events": self._safe("ETF fund events", fund_events, default=[]),
                "alerts": self._safe("FinXray events", lambda: self._events(t, since), default=[])}
