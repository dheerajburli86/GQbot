"""Pure financial maths over FMP statements. No I/O.

Two storage shapes are accepted and normalised to one:
  * heatmap `fundamentals` table: flat FMP rows, one list per statement
    (income_statement / balance_sheet / cash_flow annual, *_q quarterly),
    newest first, camelCase keys.
  * legacy `us_stock_fundamentals.data`: {"quarterly": [...], "annual": [...]}
    where each period nests income_statement / balance_sheet / cash_flow
    dicts with snake_case keys.

Every figure is computed here, never by the model: an LLM doing arithmetic
over raw statements is how a P/E of 31 gets reported as 13.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, List, Optional

# canonical name -> keys tried in order (FMP stable, FMP legacy, snake_case)
ALIASES: Dict[str, tuple] = {
    # income statement
    "revenue": ("revenue", "totalRevenue", "total_revenue"),
    "cost_of_revenue": ("costOfRevenue", "cost_of_revenue"),
    "gross_profit": ("grossProfit", "gross_profit"),
    "rnd": ("researchAndDevelopmentExpenses", "research_and_development"),
    "sga": ("sellingGeneralAndAdministrativeExpenses", "sga"),
    "operating_income": ("operatingIncome", "operating_income"),
    "ebit": ("ebit",),
    "ebitda": ("ebitda", "EBITDA"),
    "interest_expense": ("interestExpense", "interest_expense"),
    "pretax_income": ("incomeBeforeTax", "income_before_tax"),
    "income_tax": ("incomeTaxExpense", "income_tax_expense"),
    "net_income": ("netIncome", "net_income"),
    "eps": ("eps",),
    "eps_diluted": ("epsDiluted", "epsdiluted", "eps_diluted"),
    "shares_diluted": ("weightedAverageShsOutDil", "weighted_average_shares_diluted"),
    # balance sheet
    "cash": ("cashAndCashEquivalents", "cash_and_equivalents", "cash"),
    "cash_and_st": ("cashAndShortTermInvestments", "cash_and_short_term_investments"),
    "current_assets": ("totalCurrentAssets", "total_current_assets"),
    "current_liabilities": ("totalCurrentLiabilities", "total_current_liabilities"),
    "inventory": ("inventory",),
    "total_assets": ("totalAssets", "total_assets"),
    "total_liabilities": ("totalLiabilities", "total_liabilities"),
    "equity": ("totalStockholdersEquity", "totalEquity", "total_equity", "shareholders_equity"),
    "total_debt": ("totalDebt", "total_debt"),
    "net_debt": ("netDebt", "net_debt"),
    "goodwill": ("goodwill",),
    # cash flow
    "operating_cash_flow": ("operatingCashFlow", "netCashProvidedByOperatingActivities",
                            "operating_cash_flow"),
    "capex": ("capitalExpenditure", "capex", "capital_expenditure"),
    "free_cash_flow": ("freeCashFlow", "free_cash_flow"),
    "dividends_paid": ("commonDividendsPaid", "dividendsPaid", "netDividendsPaid", "dividends_paid"),
    "buybacks": ("commonStockRepurchased", "stock_repurchased", "buybacks"),
    "sbc": ("stockBasedCompensation", "stock_based_compensation"),
}
FLOWS = ("revenue", "cost_of_revenue", "gross_profit", "rnd", "sga", "operating_income", "ebit",
         "ebitda", "interest_expense", "pretax_income", "income_tax", "net_income", "eps",
         "eps_diluted", "operating_cash_flow", "capex", "free_cash_flow", "dividends_paid",
         "buybacks", "sbc")


def num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN -> None


def _pick(row: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        if k in row and row[k] is not None:
            v = num(row[k])
            if v is not None:
                return v
    return None


def _label(row: Dict[str, Any]) -> Dict[str, Any]:
    year = row.get("fiscalYear") or row.get("calendarYear") or row.get("fiscal_year")
    return {"date": str(row.get("date") or row.get("period_end") or "")[:10] or None,
            "period": row.get("period"), "fiscal_year": str(year) if year else None,
            "filing_date": str(row.get("filingDate") or row.get("fillingDate")
                               or row.get("filing_date") or "")[:10] or None}


def canon(row: Dict[str, Any]) -> Dict[str, Any]:
    """One period, any shape -> canonical keys."""
    flat = dict(row)
    for section in ("income_statement", "balance_sheet", "cash_flow"):
        if isinstance(row.get(section), dict):
            flat.update(row[section])
    out = _label(row)
    for name, keys in ALIASES.items():
        out[name] = _pick(flat, keys)
    if out["free_cash_flow"] is None and out["operating_cash_flow"] is not None and out["capex"] is not None:
        out["free_cash_flow"] = out["operating_cash_flow"] - abs(out["capex"])
    if out["ebit"] is None:
        out["ebit"] = out["operating_income"]
    if out["cash_and_st"] is None:
        out["cash_and_st"] = out["cash"]
    if out["total_debt"] is None and out["net_debt"] is not None and out["cash"] is not None:
        out["total_debt"] = out["net_debt"] + out["cash"]
    return out


def merge_statements(*lists: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Join income / balance / cash-flow rows on their period end date, newest first."""
    by_date: Dict[str, Dict[str, Any]] = {}
    for rows in lists:
        for r in rows or []:
            key = str(r.get("date") or "")[:10] or f"{r.get('fiscalYear')}{r.get('period')}"
            merged = by_date.setdefault(key, {})
            for k, v in r.items():
                if v is not None and k not in merged:
                    merged[k] = v
    return [canon(by_date[k]) for k in sorted(by_date, reverse=True)]


def periods_from_store(row: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """A `fundamentals` row or a legacy `data` blob -> {"annual": [...], "quarterly": [...]}."""
    if "quarterly" in row or "annual" in row:  # legacy blob
        q = sorted((canon(p) for p in row.get("quarterly") or []),
                   key=lambda p: p.get("date") or f"{p.get('fiscal_year')}{p.get('period')}", reverse=True)
        a = sorted((canon(p) for p in row.get("annual") or []),
                   key=lambda p: p.get("date") or p.get("fiscal_year") or "", reverse=True)
        return {"annual": a, "quarterly": q}
    return {"annual": merge_statements(row.get("income_statement"), row.get("balance_sheet"),
                                       row.get("cash_flow")),
            "quarterly": merge_statements(row.get("income_statement_q"), row.get("balance_sheet_q"),
                                          row.get("cash_flow_q"))}


# ── arithmetic helpers ───────────────────────────────────────────────────────
def ratio(a, b, pct: bool = False) -> Optional[float]:
    a, b = num(a), num(b)
    if a is None or b is None or b == 0:
        return None
    v = a / b * (100 if pct else 1)
    return round(v, 2)


def growth(new, old) -> Optional[float]:
    new, old = num(new), num(old)
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / abs(old) * 100, 2)


def cagr(new, old, years: float) -> Optional[float]:
    new, old = num(new), num(old)
    if new is None or old is None or old <= 0 or new <= 0 or years <= 0:
        return None
    return round(((new / old) ** (1 / years) - 1) * 100, 2)


def ttm(quarters: List[Dict[str, Any]], key: str, offset: int = 0) -> Optional[float]:
    window = quarters[offset:offset + 4]
    if len(window) < 4:
        return None
    vals = [q.get(key) for q in window]
    if any(v is None for v in vals):
        return None
    return sum(vals)


def _ttm_block(q: List[Dict[str, Any]], offset: int = 0) -> Dict[str, Optional[float]]:
    return {k: ttm(q, k, offset) for k in FLOWS}


def _avg(*vals) -> Optional[float]:
    v = [num(x) for x in vals if num(x) is not None]
    return sum(v) / len(v) if v else None


# ── the analysis ─────────────────────────────────────────────────────────────
def analyse(periods: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Profitability, growth, balance-sheet health, cash flow and returns on capital."""
    q, a = periods.get("quarterly") or [], periods.get("annual") or []
    if not q and not a:
        return {"ok": False, "reason": "no_statements"}
    t = _ttm_block(q) if len(q) >= 4 else {}
    t_prev = _ttm_block(q, 4) if len(q) >= 8 else {}
    basis = "TTM (last 4 quarters)" if t.get("revenue") is not None else "latest fiscal year"
    flow = t if t.get("revenue") is not None else (a[0] if a else {})
    flow_prev = t_prev if t_prev.get("revenue") is not None else (a[1] if len(a) > 1 else {})
    bal = q[0] if q else a[0]
    bal_prev = q[4] if len(q) > 4 else (a[1] if len(a) > 1 else {})

    rev, ni = flow.get("revenue"), flow.get("net_income")
    ebit = flow.get("ebit") if flow.get("ebit") is not None else flow.get("operating_income")
    tax_rate = ratio(flow.get("income_tax"), flow.get("pretax_income"))
    tax_rate = tax_rate if tax_rate is not None and 0 <= tax_rate <= 0.5 else 0.21
    invested = None
    if bal.get("equity") is not None:
        invested = (bal.get("equity") or 0) + (bal.get("total_debt") or 0) - (bal.get("cash_and_st") or 0)

    out: Dict[str, Any] = {
        "ok": True,
        "basis": basis,
        "latest_quarter": {k: bal.get(k) for k in ("date", "period", "fiscal_year", "filing_date")},
        "profitability_pct": {
            "gross_margin": ratio(flow.get("gross_profit"), rev, True),
            "operating_margin": ratio(flow.get("operating_income"), rev, True),
            "net_margin": ratio(ni, rev, True),
            "fcf_margin": ratio(flow.get("free_cash_flow"), rev, True),
            "rnd_pct_revenue": ratio(flow.get("rnd"), rev, True),
            "sbc_pct_revenue": ratio(flow.get("sbc"), rev, True),
        },
        "returns_on_capital_pct": {
            "roe": ratio(ni, _avg(bal.get("equity"), bal_prev.get("equity")), True),
            "roa": ratio(ni, _avg(bal.get("total_assets"), bal_prev.get("total_assets")), True),
            "roic": ratio(ebit * (1 - tax_rate) if ebit is not None else None, invested, True)
                    if invested and invested > 0 else None,
        },
        "growth_yoy_pct": {
            "revenue": growth(rev, flow_prev.get("revenue")),
            "gross_profit": growth(flow.get("gross_profit"), flow_prev.get("gross_profit")),
            "operating_income": growth(flow.get("operating_income"), flow_prev.get("operating_income")),
            "net_income": growth(ni, flow_prev.get("net_income")),
            "eps_diluted": growth(flow.get("eps_diluted"), flow_prev.get("eps_diluted")),
            "free_cash_flow": growth(flow.get("free_cash_flow"), flow_prev.get("free_cash_flow")),
        },
        "balance_sheet": {
            "as_of": bal.get("date"),
            "cash_and_short_term": bal.get("cash_and_st"),
            "total_debt": bal.get("total_debt"),
            "net_debt": (bal.get("total_debt") - bal.get("cash_and_st"))
                        if bal.get("total_debt") is not None and bal.get("cash_and_st") is not None
                        else bal.get("net_debt"),
            "equity": bal.get("equity"),
            "debt_to_equity": ratio(bal.get("total_debt"), bal.get("equity")),
            "net_debt_to_ebitda": ratio((bal.get("total_debt") or 0) - (bal.get("cash_and_st") or 0),
                                        flow.get("ebitda")) if flow.get("ebitda") else None,
            "current_ratio": ratio(bal.get("current_assets"), bal.get("current_liabilities")),
            "quick_ratio": ratio((bal.get("current_assets") or 0) - (bal.get("inventory") or 0),
                                 bal.get("current_liabilities")) if bal.get("current_assets") else None,
            "interest_coverage_x": ratio(ebit, abs(flow.get("interest_expense") or 0))
                                   if flow.get("interest_expense") else None,
            "goodwill_pct_assets": ratio(bal.get("goodwill"), bal.get("total_assets"), True),
        },
        "cash_flow": {
            "operating_cash_flow": flow.get("operating_cash_flow"),
            "capex": flow.get("capex"),
            "free_cash_flow": flow.get("free_cash_flow"),
            "fcf_to_net_income": ratio(flow.get("free_cash_flow"), ni) if ni and ni > 0 else None,
            "dividends_paid": abs(flow["dividends_paid"]) if flow.get("dividends_paid") else None,
            "buybacks": abs(flow["buybacks"]) if flow.get("buybacks") else None,
            "payout_ratio_pct": ratio(abs(flow.get("dividends_paid") or 0), ni, True)
                                if ni and ni > 0 and flow.get("dividends_paid") else None,
        },
        "ttm": {k: flow.get(k) for k in ("revenue", "gross_profit", "operating_income", "net_income",
                                          "ebitda", "eps_diluted", "free_cash_flow")},
    }
    if len(a) >= 2:
        yrs = min(len(a) - 1, 3)
        out["multi_year"] = {
            "years": yrs,
            "revenue_cagr_pct": cagr(a[0].get("revenue"), a[yrs].get("revenue"), yrs),
            "net_income_cagr_pct": cagr(a[0].get("net_income"), a[yrs].get("net_income"), yrs),
            "eps_cagr_pct": cagr(a[0].get("eps_diluted"), a[yrs].get("eps_diluted"), yrs),
            "fcf_cagr_pct": cagr(a[0].get("free_cash_flow"), a[yrs].get("free_cash_flow"), yrs),
        }
    out["quarterly_trend"] = quarterly_trend(q)
    out["annual_trend"] = [{"fiscal_year": p.get("fiscal_year"), "date": p.get("date"),
                            "revenue": p.get("revenue"), "net_income": p.get("net_income"),
                            "eps_diluted": p.get("eps_diluted"), "free_cash_flow": p.get("free_cash_flow"),
                            "operating_margin_pct": ratio(p.get("operating_income"), p.get("revenue"), True)}
                           for p in a[:4]]
    return out


def quarterly_trend(q: List[Dict[str, Any]], n: int = 8) -> List[Dict[str, Any]]:
    rows = []
    for i, p in enumerate(q[:n]):
        year_ago = q[i + 4] if i + 4 < len(q) else {}
        rows.append({"quarter": f"{p.get('period') or ''} {p.get('fiscal_year') or ''}".strip() or p.get("date"),
                     "date": p.get("date"),
                     "revenue": p.get("revenue"),
                     "revenue_yoy_pct": growth(p.get("revenue"), year_ago.get("revenue")),
                     "net_income": p.get("net_income"),
                     "eps_diluted": p.get("eps_diluted"),
                     "eps_yoy_pct": growth(p.get("eps_diluted"), year_ago.get("eps_diluted")),
                     "gross_margin_pct": ratio(p.get("gross_profit"), p.get("revenue"), True),
                     "operating_margin_pct": ratio(p.get("operating_income"), p.get("revenue"), True)})
    return rows


def valuation(periods: Dict[str, List[Dict[str, Any]]], market_cap: Optional[float],
              price: Optional[float]) -> Dict[str, Any]:
    """Price-based multiples. Inputs must be in the same currency."""
    q, a = periods.get("quarterly") or [], periods.get("annual") or []
    t = _ttm_block(q) if len(q) >= 4 else {}
    flow = t if t.get("revenue") is not None else (a[0] if a else {})
    bal = q[0] if q else (a[0] if a else {})
    mc = num(market_cap)
    if not mc and price and q and q[0].get("shares_diluted"):
        mc = price * q[0]["shares_diluted"]
    if not mc:
        return {"ok": False, "reason": "no_market_cap"}
    ev = mc + (bal.get("total_debt") or 0) - (bal.get("cash_and_st") or 0)
    ni, eps = flow.get("net_income"), flow.get("eps_diluted")
    pe = ratio(mc, ni) if ni and ni > 0 else None
    if pe is None and eps and eps > 0 and price:
        pe = ratio(price, eps)
    div = abs(flow.get("dividends_paid") or 0)
    bb = abs(flow.get("buybacks") or 0)
    growth_eps = growth(flow.get("eps_diluted"), (_ttm_block(q, 4) if len(q) >= 8 else {}).get("eps_diluted"))
    return {
        "ok": True,
        "market_cap": mc,
        "enterprise_value": ev,
        "pe_ttm": pe,
        "earnings": "negative (P/E not meaningful)" if ni is not None and ni <= 0 else None,
        "price_to_sales": ratio(mc, flow.get("revenue")),
        "price_to_book": ratio(mc, bal.get("equity")) if (bal.get("equity") or 0) > 0 else None,
        "ev_to_ebitda": ratio(ev, flow.get("ebitda")) if (flow.get("ebitda") or 0) > 0 else None,
        "ev_to_sales": ratio(ev, flow.get("revenue")),
        "price_to_fcf": ratio(mc, flow.get("free_cash_flow")) if (flow.get("free_cash_flow") or 0) > 0 else None,
        "earnings_yield_pct": ratio(ni, mc, True),
        "fcf_yield_pct": ratio(flow.get("free_cash_flow"), mc, True),
        "dividend_yield_pct": ratio(div, mc, True) if div else 0.0,
        "shareholder_yield_pct": ratio(div + bb, mc, True) if (div + bb) else 0.0,
        "peg": round(pe / growth_eps, 2) if pe and growth_eps and growth_eps > 0 else None,
    }


def summarise_peers(rows: List[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, Any]:
    out = {}
    for k in keys:
        vals = sorted(v for v in (num(r.get(k)) for r in rows) if v is not None)
        if vals:
            out[k] = {"median": round(median(vals), 2), "low": round(vals[0], 2),
                      "high": round(vals[-1], 2), "n": len(vals)}
    return out


def percentile_rank(value, population: Iterable) -> Optional[float]:
    v = num(value)
    pop = [x for x in (num(p) for p in population) if x is not None]
    if v is None or not pop:
        return None
    below = sum(1 for x in pop if x < v)
    return round(below / len(pop) * 100, 1)
