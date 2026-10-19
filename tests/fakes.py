"""In-memory stand-ins for Supabase and the LLM. No network, no credentials."""

from __future__ import annotations

import fnmatch
import json
from datetime import date, datetime, timedelta, timezone

from gq_bot.llm import LLMReply, ToolCall


class _Res:
    def __init__(self, data):
        self.data = data


class Query:
    def __init__(self, db, name):
        self.db, self.name = db, name
        self.filters, self._order, self._limit, self._range, self._insert = [], None, None, None, None

    def select(self, *_a, **_k): return self
    def eq(self, c, v): self.filters.append(lambda r: str(r.get(c)) == str(v)); return self
    def in_(self, c, vs): s = {str(v) for v in vs}; self.filters.append(lambda r: str(r.get(c)) in s); return self
    def gte(self, c, v): self.filters.append(lambda r: r.get(c) is not None and str(r.get(c)) >= str(v)); return self
    def contains(self, c, vs): self.filters.append(lambda r: set(vs) <= set(r.get(c) or [])); return self
    def ilike(self, c, p): self.filters.append(lambda r: fnmatch.fnmatch(str(r.get(c) or "").lower(), p.lower().replace("%", "*"))); return self
    def order(self, c, desc=False): self._order = (c, desc); return self
    def limit(self, n): self._limit = n; return self
    def range(self, a, b): self._range = (a, b); return self
    def insert(self, row): self._insert = row; return self

    def execute(self):
        self.db.calls.append(self.name)
        if self.name in self.db.broken:
            raise RuntimeError(f"relation {self.name} does not exist")
        if self._insert is not None:
            self.db.tables.setdefault(self.name, []).append(self._insert)
            return _Res([self._insert])
        rows = [r for r in self.db.tables.get(self.name, []) if all(f(r) for f in self.filters)]
        if self._order:
            c, desc = self._order
            rows.sort(key=lambda r: (r.get(c) is None, r.get(c) if r.get(c) is not None else 0), reverse=desc)
        if self._range:
            rows = rows[self._range[0]:self._range[1] + 1]
        if self._limit is not None:
            rows = rows[:self._limit]
        return _Res([dict(r) for r in rows])


class FakeSupabase:
    def __init__(self, tables=None, broken=()):
        self.tables = tables or {}
        self.broken = set(broken)
        self.calls = []

    def table(self, name):
        return Query(self, name)


TODAY = date(2026, 10, 16)


def _iso_days_ago(n):
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def world(**overrides):
    """A small but complete FinXray + heatmap world."""
    bars = [{"instrument_id": "i-aapl", "date": (TODAY - timedelta(days=d)).isoformat(), "close": c}
            for d, c in [(6, 200.0), (5, 198.0), (4, 190.0), (3, 185.0), (2, 186.0), (1, 180.0), (0, 181.0)]]
    tables = {
        "users": [{"id": "u1", "telegram_chat_id": "42", "telegram_username": "dj", "is_active": True},
                  {"id": "u2", "telegram_chat_id": "43", "telegram_username": "gone", "is_active": False},
                  {"id": "u3", "telegram_chat_id": None, "telegram_username": "late", "is_active": True}],
        "watchlists": [{"user_id": "u1", "ticker": "aapl"}, {"user_id": "u1", "ticker": "QQQ"},
                       {"user_id": "u1", "ticker": "MSFT"}, {"user_id": "u9", "ticker": "TSLA"}],
        "alerts": [
            {"id": 1, "ticker": "AAPL", "summary": "Apple cut iPhone guidance.", "impact": "HIGH",
             "filing_type": "8-K", "source": "SEC_EDGAR", "created_at": _iso_days_ago(3),
             "extra": {"feature_name": "SEC EDGAR Filings"}, "link": "https://sec.gov/x"},
            {"id": 2, "ticker": "AAPL", "summary": "Analyst lowered target.", "impact": "LOW",
             "filing_type": "RATING", "source": "FMP", "created_at": _iso_days_ago(1), "extra": {}},
            {"id": 3, "ticker": "AAPL", "summary": "Old news.", "impact": "MEDIUM",
             "filing_type": "NEWS", "created_at": _iso_days_ago(60), "extra": {}},
            {"id": 4, "ticker": "TSLA", "summary": "Someone else's ticker.", "impact": "HIGH",
             "created_at": _iso_days_ago(1), "extra": {}},
        ],
        "etf_master": [{"ticker": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ", "id": 9}],
        "etf_company_data": [{"ticker": "QQQ", "aum": 300e9, "sector_weights": {"Technology": 50}}],
        "etf_expenses": [{"ticker": "QQQ", "expense_ratio": 0.2}],
        "etf_holdings": [{"ticker": "QQQ", "security_symbol": s, "security_name": s, "portfolio_pct": w, "as_of": "2026-10-15"}
                         for s, w in [("MSFT", 8.5), ("AAPL", 8.9), ("NVDA", 8.1)]],
        "etf_holdings_changes": [{"ticker": "QQQ", "action": "ADDED", "security_symbol": "PLTR",
                                  "security_name": "Palantir", "weight_new": 0.4, "detected_at": _iso_days_ago(5)}],
        "etf_fund_events": [{"tickers": ["QQQ"], "status": "QUEUED", "form": "497", "event_type": "INDEX_CHANGE",
                             "excerpt": "Index methodology change", "created_at": _iso_days_ago(2),
                             "filing_date": "2026-10-14", "filing_url": "https://sec.gov/y"},
                            {"tickers": ["QQQ"], "status": "NO_MATCH", "created_at": _iso_days_ago(2)}],
    }
    heat = {
        "instruments": [
            {"id": "i-aapl", "symbol": "AAPL", "name": "Apple Inc", "universe": "us_stocks", "sector": "Technology", "is_active": True},
            {"id": "i-msft", "symbol": "MSFT", "name": "Microsoft", "universe": "us_stocks", "sector": "Technology", "is_active": True},
            {"id": "i-nvda", "symbol": "NVDA", "name": "NVIDIA", "universe": "us_stocks", "sector": "Technology", "is_active": True},
            {"id": "i-qqq", "symbol": "QQQ", "name": "Invesco QQQ Trust", "universe": "us_etfs", "is_active": True},
            {"id": "i-spy", "symbol": "SPY", "name": "SPDR S&P 500", "universe": "us_etfs", "is_active": True},
            {"id": "i-tsla", "symbol": "TSLA", "name": "Tesla", "universe": "us_stocks", "sector": "Consumer", "is_active": True},
        ],
        "returns_latest": [
            {"instrument_id": "i-aapl", "price_usd": 181.0, "as_of": "2026-10-16", "ret_1d": 0.56, "ret_1w": -9.5, "ret_1m": -12.0},
            {"instrument_id": "i-msft", "price_usd": 400.0, "as_of": "2026-10-16", "ret_1d": 0.1, "ret_1w": -1.0},
            {"instrument_id": "i-nvda", "price_usd": 120.0, "as_of": "2026-10-16", "ret_1d": 1.0, "ret_1w": 2.0},
            {"instrument_id": "i-qqq", "price_usd": 480.0, "as_of": "2026-10-16", "ret_1d": 0.3, "ret_1w": -0.8},
            {"instrument_id": "i-spy", "price_usd": 560.0, "as_of": "2026-10-16", "ret_1d": 0.2, "ret_1w": -0.5},
        ],
        "price_history": bars,
        "us_stock_fundamentals": [{"ticker": "AAPL", "updated_at": "2026-10-01", "data": {
            "quarterly": [{"period": f"Q{i}", "fiscal_year": "2026",
                           "income_statement": {"revenue": 90e9 + i, "net_income": 20e9, "noise": 1},
                           "balance_sheet": {"cash": 30e9}, "cash_flow": {"free_cash_flow": 25e9}} for i in range(6)],
            "annual": [{"period": "FY", "fiscal_year": y, "income_statement": {"revenue": 380e9}} for y in ("2025", "2024", "2023")]}}],
    }
    tables.update(overrides.pop("fx", {}))
    heat.update(overrides.pop("hm", {}))
    return FakeSupabase(tables, overrides.get("fx_broken", ())), FakeSupabase(heat, overrides.get("hm_broken", ()))


class ScriptedLLM:
    """Replays a list of replies; records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def chat(self, messages, tools=None):
        self.sent.append({"messages": json.loads(json.dumps(messages, default=str)), "tools": tools})
        if not self.replies:
            return LLMReply(content="(no more script)")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def call(name, **args):
    return LLMReply(tool_calls=[ToolCall(id=f"c-{name}", name=name, arguments=json.dumps(args))],
                    prompt_tokens=100, completion_tokens=10)


def say(text):
    return LLMReply(content=text, prompt_tokens=200, completion_tokens=50)
