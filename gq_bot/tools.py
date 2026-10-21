"""Tool schemas the LLM sees, and the dispatcher that runs them for ONE user.

The user id is bound by the dispatcher, never taken from the model: a tool call
cannot name a different user, so it cannot read another person's watchlist.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Any, Callable, Dict, List

from .data import EVENT_CATEGORIES, RANK_METRICS, GQData

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 14000

_T = {"type": "string", "description": "US ticker symbol, e.g. AAPL or SPY."}


def _fn(name: str, description: str, props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props,
                       **({"required": required} if required else {})}}}


_DAYS = lambda d, hi: {"type": "integer", "description": f"Calendar days to look back (1-{hi}).", "default": d}
_START = {"type": "string", "description": "Optional start date YYYY-MM-DD for a specific period "
          "(e.g. 'in March' -> 2026-03-01). Overrides days."}
_END = {"type": "string", "description": "Optional end date YYYY-MM-DD. Defaults to today."}

TOOLS: List[Dict[str, Any]] = [
    _fn("explain_move",
        "Best first call for any 'why did X move / what happened to X' question, for a recent "
        "window or a specific period. Returns the price path, the biggest up/down sessions each "
        "with the FinXray events dated just before them, all FinXray events in the window, the "
        "S&P 500 over the same dates and (for recent windows) the sector median.",
        {"ticker": _T, "days": _DAYS(7, 365), "start_date": _START, "end_date": _END}, ["ticker"]),
    _fn("get_snapshot",
        "Latest price and returns (1D, 1W, 1M, 3M, 6M, YTD, 1Y, 5Y, 10Y, ALL) exactly as on the "
        "heatmap dashboard, plus name, type, sector, industry and market cap.",
        {"ticker": _T}, ["ticker"]),
    _fn("get_company_profile",
        "What the company or fund IS: GQuants' own business notes (from annual reports) or the "
        "business description, sector, industry, market cap, CEO, employees, IPO date, beta. "
        "Use for 'what does X do', 'tell me about X', business model and segment questions.",
        {"ticker": _T}, ["ticker"]),
    _fn("get_financials",
        "Company financial analysis computed from its statements (stocks only): margins, "
        "revenue/EPS/FCF growth (TTM YoY and multi-year CAGR), ROE/ROA/ROIC, debt, liquidity, "
        "interest coverage, cash flow, dividends and buybacks, plus an 8-quarter and 4-year trend. "
        "Use for earnings, growth, profitability, balance-sheet and 'is it financially healthy' "
        "questions.",
        {"ticker": _T}, ["ticker"]),
    _fn("get_valuation",
        "Valuation multiples (stocks only): P/E TTM, P/S, P/B, EV/EBITDA, EV/Sales, P/FCF, "
        "FCF and earnings yield, dividend and shareholder yield, PEG, set against the median of "
        "its closest-sized industry peers. Use for 'is X expensive/cheap', 'what is its P/E'.",
        {"ticker": _T}, ["ticker"]),
    _fn("get_peer_comparison",
        "The stock beside its ~10 closest-sized industry peers: returns, valuation, margins, "
        "growth, ROE, peer medians and the stock's percentile rank on each. Use for 'how does X "
        "compare with its peers/competitors' and relative-strength questions.",
        {"ticker": _T}, ["ticker"]),
    _fn("get_technicals",
        "Technical picture from daily bars: trend (vs 20/50/100/200-day averages), last golden/death "
        "cross, RSI(14), MACD, 52-week range and position, 3-month support/resistance, volatility, "
        "ATR, volume vs average, plus GQuants dashboard trend signals where stored.",
        {"ticker": _T}, ["ticker"]),
    _fn("get_risk_metrics",
        "Risk and return over a period: total and annualised return, volatility, Sharpe and "
        "Sortino (no risk-free rate), max drawdown with dates, best/worst day, and versus a "
        "benchmark: beta, correlation, excess return, up/down capture. Benchmark can be SPY, QQQ, "
        "IWM, DIA or another watchlist ticker (use it for correlation between two holdings).",
        {"ticker": _T, "benchmark": {"type": "string", "default": "SPY"},
         "days": _DAYS(365, 3650), "start_date": _START, "end_date": _END}, ["ticker"]),
    _fn("get_price_history",
        "Price path over any window or date range with a summary: change, high, low, max "
        "drawdown, biggest up and down sessions. Use for 'what was X's price on/in ...'.",
        {"ticker": _T, "days": _DAYS(30, 3650), "start_date": _START, "end_date": _END}, ["ticker"]),
    _fn("get_events",
        "GQ FinXray events for a ticker: SEC filings, news, earnings, analyst ratings, insider "
        "trades, technical alerts. Filter by category and impact; up to 365 days back.",
        {"ticker": _T, "days": _DAYS(14, 365),
         "category": {"type": "string", "enum": list(EVENT_CATEGORIES),
                      "description": "Only this kind of event."},
         "min_impact": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
         "start_date": _START, "end_date": _END}, ["ticker"]),
    _fn("compare_tickers",
        "Side-by-side snapshot (price, all return windows, sector, market cap and, for stocks, "
        "key valuation multiples) for up to 5 watchlist tickers.",
        {"tickers": {"type": "array", "items": {"type": "string"}, "maxItems": 5}}, ["tickers"]),
    _fn("watchlist_overview",
        "The user's whole watchlist: top gainers and losers, breadth (up vs down, average, median), "
        "sector mix, uncovered tickers and HIGH-impact FinXray events from the last 7 days. Use for "
        "'how is my watchlist/portfolio doing', 'what moved today', 'anything important I should know'.",
        {"window": {"type": "string", "enum": ["1d", "1w", "1m", "3m", "6m", "ytd", "1y", "5y"],
                    "default": "1d"}}, []),
    _fn("rank_watchlist",
        "Rank the watchlist by a metric, e.g. best performers this year, cheapest by P/E, highest "
        "dividend yield, most oversold (RSI), furthest below 52-week high, fastest revenue growth. "
        "When the user names specific tickers (e.g. the sector ETFs), pass them in 'tickers': only "
        "those are ranked and ALL of them are returned. Call once per metric.",
        {"metric": {"type": "string", "enum": sorted(RANK_METRICS)},
         "order": {"type": "string", "enum": ["desc", "asc"], "default": "desc",
                   "description": "desc = highest first."},
         "limit": {"type": "integer", "default": 10},
         "type_filter": {"type": "string", "enum": ["stock", "etf"]},
         "tickers": {"type": "array", "items": {"type": "string"}, "maxItems": 30,
                     "description": "Rank only these watchlist tickers; omit to rank the whole watchlist."}},
        ["metric"]),
    _fn("price_chart",
        "Draw a price chart and send it to the user as an image: one ticker with its 50/200-day "
        "averages, or up to 5 tickers rebased to 100 for comparison. Use whenever the user asks "
        "to see, plot, chart or visualise prices, and it is a good companion to comparisons.",
        {"tickers": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
         "days": _DAYS(180, 3650), "start_date": _START, "end_date": _END}, ["tickers"]),
    _fn("get_etf_profile",
        "ETF facts: fund house, category, AUM, NAV, inception, expense ratio, sector and country "
        "weights where stored, plus its price snapshot. ETFs only.",
        {"ticker": _T}, ["ticker"]),
    _fn("get_etf_holdings",
        "Top holdings of an ETF by portfolio weight and their combined weight. ETFs only.",
        {"ticker": _T, "top_n": {"type": "integer", "description": "1-25.", "default": 10}},
        ["ticker"]),
    _fn("get_etf_overlap",
        "How much two watchlist ETFs overlap: shared holdings, weight overlap and the biggest "
        "common positions.",
        {"ticker_a": _T, "ticker_b": _T}, ["ticker_a", "ticker_b"]),
    _fn("get_etf_exposure",
        "Which ETFs on the user's watchlist hold a given stock, and at what weight. The stock does "
        "not need to be on the watchlist.",
        {"symbol": {"type": "string", "description": "Stock ticker to look for, e.g. NVDA."}}, ["symbol"]),
    _fn("get_etf_changes",
        "What changed in an ETF: holdings added or removed, SEC fund actions (liquidation, "
        "merger, fee, index change) and FinXray alerts. ETFs only.",
        {"ticker": _T, "days": _DAYS(30, 365)}, ["ticker"]),
]


def _compact(result: Any) -> str:
    text = json.dumps(result, default=str, separators=(",", ":"))
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + '..."[truncated]"'


class ToolRunner:
    """Runs tool calls for one user and records what was called."""

    def __init__(self, data: GQData, user_id, named: Optional[List[str]] = None):
        self.data, self.user_id = data, user_id
        # watchlist tickers the user named in the question
        self.named = list(named or [])
        self.calls: List[Dict[str, Any]] = []
        self.outputs: List[str] = []      # what the model was shown, for the grounding checks
        uid = user_id
        self.images: List[bytes] = []
        d = data
        self._table: Dict[str, Callable[..., Any]] = {
            "explain_move": lambda ticker, days=7, start_date=None, end_date=None:
                d.explain_move(uid, ticker, days, start_date, end_date),
            "get_snapshot": lambda ticker: d.snapshot(uid, ticker),
            "get_company_profile": lambda ticker: d.company_profile(uid, ticker),
            "get_financials": lambda ticker: d.financials(uid, ticker),
            "get_fundamentals": lambda ticker: d.financials(uid, ticker),
            "get_valuation": lambda ticker: d.valuation(uid, ticker),
            "get_peer_comparison": lambda ticker: d.peer_comparison(uid, ticker),
            "get_technicals": lambda ticker: d.technicals(uid, ticker),
            "get_risk_metrics": lambda ticker, benchmark="SPY", days=365, start_date=None, end_date=None:
                d.risk(uid, ticker, benchmark, days, start_date, end_date),
            "get_price_history": lambda ticker, days=30, start_date=None, end_date=None:
                d.price_history(uid, ticker, days, start_date, end_date),
            "get_events": lambda ticker, days=14, min_impact=None, category=None, start_date=None, end_date=None:
                d.events(uid, ticker, days, min_impact, category, start_date, end_date),
            "compare_tickers": lambda tickers: d.compare(uid, tickers),
            "watchlist_overview": lambda window="1d": d.watchlist_overview(uid, window),
            "rank_watchlist": lambda metric, order="desc", limit=10, type_filter=None, tickers=None:
                d.rank_watchlist(uid, metric, order, limit, type_filter, tickers),
            "price_chart": lambda tickers, days=180, start_date=None, end_date=None:
                d.chart(uid, tickers, days, start_date, end_date),
            "get_etf_profile": lambda ticker: d.etf_profile(uid, ticker),
            "get_etf_holdings": lambda ticker, top_n=10: d.etf_holdings(uid, ticker, top_n),
            "get_etf_overlap": lambda ticker_a, ticker_b: d.etf_overlap(uid, ticker_a, ticker_b),
            "get_etf_exposure": lambda symbol: d.etf_exposure(uid, symbol),
            "get_etf_changes": lambda ticker, days=30: d.etf_changes(uid, ticker, days),
        }

    @property
    def tickers(self) -> List[str]:
        out = []
        for c in self.calls:
            args = c.get("args") or {}
            singles = [args.get(k) for k in ("ticker", "ticker_a", "ticker_b", "symbol") if args.get(k)]
            for t in singles + list(args.get("tickers") or []):
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
            # A ranking of a named set must not fall back to the watchlist's top 10.
            if name == "rank_watchlist" and not args.get("tickers") and len(self.named) >= 2:
                args["tickers"] = self.named
            try:
                result = fn(**args)
            except TypeError as e:
                result = {"ok": False, "reason": "bad_arguments", "message": str(e)}
            except Exception as e:
                log.exception("[tools] %s crashed", name)
                result = {"ok": False, "reason": "tool_error", "message": f"{name} failed: {e}"}
        if isinstance(result, dict) and isinstance(result.get("_image"), (bytes, bytearray)):
            if len(self.images) < 3:
                self.images.append(bytes(result["_image"]))
            result = {k: v for k, v in result.items() if k != "_image"}
        self.calls.append({"name": name, "args": args,
                           "ok": bool(result.get("ok")) if isinstance(result, dict) else True})
        out = _compact(result)
        self.outputs.append(out)
        return out
