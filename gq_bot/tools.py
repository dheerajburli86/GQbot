"""Tool schemas the LLM sees, and the dispatcher that runs them for ONE user.

The user id is bound by the dispatcher, never taken from the model: a tool call
cannot name a different user, so it cannot read another person's watchlist.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List

from .data import GQData

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 9000

_T = {"type": "string", "description": "US ticker symbol, e.g. AAPL or SPY."}


def _fn(name: str, description: str, props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS: List[Dict[str, Any]] = [
    _fn("explain_move",
        "Best first call for any 'why is X up/down/falling/rallying' question. Returns the "
        "price path over the window, every GQ FinXray event for the ticker in and just "
        "before that window, the sector median and the S&P 500 (SPY) over the same window.",
        {"ticker": _T, "days": {"type": "integer", "description": "Calendar days to look back (1-90). "
                                "1 = today, 7 = this week, 30 = this month.", "default": 7}},
        ["ticker"]),
    _fn("get_snapshot",
        "Latest price and returns (1D, 1W, 1M, 3M, 6M, YTD, 1Y, 5Y, 10Y, ALL) as shown on "
        "the heatmap dashboard, plus name, type, sector and market cap.",
        {"ticker": _T}, ["ticker"]),
    _fn("get_price_history",
        "Daily closes over a window with a summary: change, high, low, max drawdown, "
        "biggest up and down sessions.",
        {"ticker": _T, "days": {"type": "integer", "description": "Calendar days (2-400).", "default": 30}},
        ["ticker"]),
    _fn("get_events",
        "GQ FinXray events for a ticker: SEC filings, news, earnings, analyst ratings, insider "
        "trades, technical alerts, transcripts. Each has a summary, impact and date.",
        {"ticker": _T,
         "days": {"type": "integer", "description": "Calendar days to look back (1-90).", "default": 14},
         "min_impact": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"],
                        "description": "Only events at or above this impact."}},
        ["ticker"]),
    _fn("get_fundamentals",
        "Latest 4 quarters and 2 fiscal years of income statement, balance sheet and cash "
        "flow. Stocks only.",
        {"ticker": _T}, ["ticker"]),
    _fn("compare_tickers",
        "Side-by-side snapshot for up to 5 tickers from the user's watchlist.",
        {"tickers": {"type": "array", "items": {"type": "string"}, "maxItems": 5}}, ["tickers"]),
    _fn("get_etf_profile",
        "ETF facts: fund details, expense ratio, AUM and sector/country weights where stored, "
        "plus its price snapshot. ETFs only.",
        {"ticker": _T}, ["ticker"]),
    _fn("get_etf_holdings",
        "Top holdings of an ETF by portfolio weight. ETFs only.",
        {"ticker": _T, "top_n": {"type": "integer", "description": "1-25.", "default": 10}},
        ["ticker"]),
    _fn("get_etf_changes",
        "What changed in an ETF: holdings added or removed, SEC fund actions (liquidation, "
        "merger, fee, index change) and FinXray alerts. ETFs only.",
        {"ticker": _T, "days": {"type": "integer", "description": "1-90.", "default": 30}},
        ["ticker"]),
]


def _compact(result: Any) -> str:
    text = json.dumps(result, default=str, separators=(",", ":"))
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + '..."[truncated]"'


class ToolRunner:
    """Runs tool calls for one user and records what was called."""

    def __init__(self, data: GQData, user_id):
        self.data, self.user_id = data, user_id
        self.calls: List[Dict[str, Any]] = []
        uid = user_id
        self._table: Dict[str, Callable[..., Any]] = {
            "explain_move": lambda ticker, days=7: data.explain_move(uid, ticker, days),
            "get_snapshot": lambda ticker: data.snapshot(uid, ticker),
            "get_price_history": lambda ticker, days=30: data.price_history(uid, ticker, days),
            "get_events": lambda ticker, days=14, min_impact=None: data.events(uid, ticker, days, min_impact),
            "get_fundamentals": lambda ticker: data.fundamentals(uid, ticker),
            "compare_tickers": lambda tickers: data.compare(uid, tickers),
            "get_etf_profile": lambda ticker: data.etf_profile(uid, ticker),
            "get_etf_holdings": lambda ticker, top_n=10: data.etf_holdings(uid, ticker, top_n),
            "get_etf_changes": lambda ticker, days=30: data.etf_changes(uid, ticker, days),
        }

    @property
    def tickers(self) -> List[str]:
        out = []
        for c in self.calls:
            args = c.get("args") or {}
            for t in ([args.get("ticker")] if args.get("ticker") else []) + list(args.get("tickers") or []):
                t = str(t).upper()
                if t not in out:
                    out.append(t)
        return out

    def run(self, name: str, raw_args: Any) -> str:
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
        except (TypeError, ValueError):
            args = {}
        fn = self._table.get(name)
        if fn is None:
            result = {"ok": False, "reason": "unknown_tool", "message": f"No tool named {name}."}
        else:
            allowed = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            args = {k: v for k, v in args.items() if k in allowed and v is not None}
            try:
                result = fn(**args)
            except TypeError as e:
                result = {"ok": False, "reason": "bad_arguments", "message": str(e)}
            except Exception as e:
                log.exception("[tools] %s crashed", name)
                result = {"ok": False, "reason": "tool_error", "message": f"{name} failed: {e}"}
        self.calls.append({"name": name, "args": args,
                           "ok": bool(result.get("ok")) if isinstance(result, dict) else True})
        return _compact(result)
