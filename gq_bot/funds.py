"""Mutual funds, fund analytics shared by ETFs and mutual funds, and US market context.

Mixed into GQData, so every method here has the same scope guarantees: a ticker
goes through `_scoped()` before anything is read.

Sources (all read-only)
  FinXray US   mf_master, mf_holdings (latest N-PORT snapshot), mf_holdings_state,
               mf_holdings_changes, mf_security_master (sector / cap class / country
               of what funds hold), mf_expense_breakdown, mf_managers,
               mf_manager_events, mf_fund_events, mf_scheme_events, fund_universe,
               fund_managers, fund_manager_events, etf_expense_breakdown,
               etf_listing_state, etf_company_data
  Heatmap      instruments, returns_latest, tech_indicators_52w, tech_indicators_rsi

What mutual-fund data is NOT here: there is no NAV or price history for mutual
funds anywhere in GQuants, so no returns, risk or technicals for them. Holdings
come from N-PORT, which is public once a quarter and up to 60 days late, so a
mutual fund's holdings are always "as of" a reporting date that must be stated.
"""

from __future__ import annotations

import re
import statistics
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

PAGE = 1000
CHUNK = 100
MAX_HOLDINGS = 6000
SKIP_EVENT_STATUS = {"NO_MATCH", "NO_DOC", "ERROR", "NO_SERIES", "NOT_WATCHED"}
NO_NAV = ("GQuants has no NAV or price history for mutual funds: no returns, risk, volatility or "
          "technicals. Holdings, fees, managers and SEC filings are available.")
WINDOWS = {"1d": "ret_1d", "1w": "ret_1w", "1m": "ret_1m", "3m": "ret_3m", "6m": "ret_6m",
           "ytd": "ret_ytd", "1y": "ret_1y"}
ASSET_CATEGORY = {"EC": "equity", "EP": "preferred equity", "DBT": "debt", "STIV": "short-term / cash vehicle",
                  "RF": "registered fund", "RE": "real estate", "DCO": "commodity derivative",
                  "DCR": "credit derivative", "DE": "equity derivative", "DFE": "FX derivative",
                  "DIR": "interest-rate derivative", "DO": "other derivative", "SN": "structured note",
                  "LON": "loan", "ABS-MBS": "mortgage-backed", "ABS-O": "asset-backed (other)",
                  "ABS-CBDO": "CDO/CBO", "ABS-APCP": "asset-backed CP", "COMM": "commodity", "OTHER": "other"}
COMPARE_METRICS = ("net_expense_ratio", "management_fee", "aum", "holdings_count", "top10_weight_pct",
                   "longest_manager_tenure_years", "ret_1y", "ret_5y")


def _num(v) -> Optional[float]:
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def _norm(t: str) -> str:
    return (t or "").strip().upper().lstrip("$")


def _refuse(t: str, reason: str, message: str, **extra) -> Dict[str, Any]:
    return {"ok": False, "ticker": t, "reason": reason, "message": message, **extra}


def _clean(d: Optional[Dict[str, Any]], drop=("id", "created_at", "updated_at", "fmp_checked_at")) -> Dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if k not in drop and v not in (None, "", [], {})}


def _hkey(symbol: Optional[str], name: Optional[str]) -> str:
    """One identity per security across funds: the ticker, else the normalised issuer name."""
    s = (symbol or "").strip().upper()
    if s and s not in ("N/A", "NONE", "-"):
        return s
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())[:40]


def _tenure_years(year, month, today: date) -> Optional[float]:
    y = _num(year)
    if not y:
        return None
    start = date(int(y), int(_num(month) or 1), 1)
    return round(max((today - start).days, 0) / 365.25, 1)


def _weighted(rows: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    tot: Dict[str, float] = {}
    for r in rows:
        tot[r.get(field) or "unclassified"] = tot.get(r.get(field) or "unclassified", 0.0) + (r["weight"] or 0)
    return [{"name": k, "weight_pct": round(v, 2)} for k, v in sorted(tot.items(), key=lambda x: -x[1]) if v]


class FundsMixin:
    # ── identity ─────────────────────────────────────────────────────────────
    def _mf_many(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        out, missing = {}, []
        for t in tickers:
            hit = self._mfc.get(("mf", t))
            if hit is None:
                missing.append(t)
            elif hit:
                out[t] = hit
        for k in range(0, len(missing), CHUNK):
            part = missing[k:k + CHUNK]
            rows = self._safe("mf_master", lambda p=part: self._rows(
                self.fx.table("mf_master").select("*").in_("ticker", p)), default=[])
            got = {_norm(r.get("ticker")): r for r in rows}
            for t in part:
                self._mfc.put(("mf", t), got.get(t) or {})
                if t in got:
                    out[t] = got[t]
        return out

    def _kinds(self, tickers: List[str]) -> Dict[str, Optional[str]]:
        """etf | mutual_fund | stock | None for each ticker."""
        insts = self._instruments_for(tickers)
        mfs = self._mf_many([t for t in tickers if t not in insts])
        kinds: Dict[str, Optional[str]] = {}
        for t in tickers:
            u = (insts.get(t) or {}).get("universe")
            kinds[t] = "etf" if u == "us_etfs" else "stock" if u == "us_stocks" else \
                "mutual_fund" if t in mfs else None
        unknown = [t for t, k in kinds.items() if k is None]
        if unknown:
            rows = self._safe("etf_master", lambda: self._rows(
                self.fx.table("etf_master").select("ticker").in_("ticker", unknown)), default=[])
            for r in rows:
                kinds[_norm(r.get("ticker"))] = "etf"
        return kinds

    def _series_for(self, t: str, kind: str) -> Optional[str]:
        if kind == "mutual_fund":
            return (self._mf_many([t]).get(t) or {}).get("series_id")
        hit = self._mfc.get(("series", t))
        if hit is None:
            rows = self._safe("fund_universe", lambda: self._rows(
                self.fx.table("fund_universe").select("series_id").eq("ticker", t).limit(1)), default=[])
            hit = self._mfc.put(("series", t), (rows[0].get("series_id") if rows else "") or "")
        return hit or None

    def _fund_scoped(self, user_id, ticker: str) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        t, no = self._scoped(user_id, ticker)
        if no:
            return None, None, no
        kind = self._kinds([t]).get(t)
        if kind not in ("etf", "mutual_fund"):
            return None, None, _refuse(t, "not_a_fund", f"{t} is not a fund. Fund tools cover ETFs and "
                                                       "mutual funds; use the stock tools for a company.")
        return t, kind, None

    def _watch_funds(self, user_id) -> Dict[str, str]:
        kinds = self._kinds(self.watchlist(user_id))
        return {t: k for t, k in kinds.items() if k in ("etf", "mutual_fund")}

    # ── holdings ─────────────────────────────────────────────────────────────
    def _fund_holdings(self, t: str, kind: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Every holding of a fund as {key, symbol, name, weight, ...}, biggest first, and its as-of date."""
        hit = self._mfc.get(("h", t))
        if hit is not None:
            return hit
        rows: List[Dict[str, Any]] = []
        if kind == "etf":
            table, col, val = "etf_holdings", "ticker", t
            cols = "security_symbol, security_name, portfolio_pct, as_of"
        else:
            sid = self._series_for(t, kind)
            if not sid:
                return self._mfc.put(("h", t), ([], None))
            table, col, val = "mf_holdings", "series_id", sid
            cols = "security_key, security_name, title, ticker, isin, asset_category, portfolio_pct, value_usd, rep_date"
        page = 0
        while len(rows) < MAX_HOLDINGS:
            chunk = self._rows(self.fx.table(table).select(cols).eq(col, val)
                               .order("portfolio_pct", desc=True).range(page * PAGE, page * PAGE + PAGE - 1))
            rows.extend(chunk)
            if len(chunk) < PAGE:
                break
            page += 1
        out = []
        for r in rows:
            sym = r.get("security_symbol") if kind == "etf" else r.get("ticker")
            out.append({"key": _hkey(sym, r.get("security_name")), "symbol": (sym or "").upper() or None,
                        "name": r.get("security_name") or r.get("title"), "weight": _num(r.get("portfolio_pct")),
                        "security_key": r.get("security_key"), "asset_category": r.get("asset_category"),
                        "value_usd": _num(r.get("value_usd"))})
        out.sort(key=lambda h: -(h["weight"] or 0))
        as_of = None
        if rows:
            dates = [str(r.get("as_of") or r.get("rep_date") or "")[:10] for r in rows]
            as_of = max(d for d in dates) or None
        return self._mfc.put(("h", t), (out, as_of))

    def _security_master(self, keys: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        keys = [k for k in dict.fromkeys(keys) if k]
        for k in range(0, len(keys), CHUNK):
            for r in self._rows(self.fx.table("mf_security_master")
                                .select("security_key, ticker, name, sector, industry, cap_class, country, "
                                        "asset_category, is_fund")
                                .in_("security_key", keys[k:k + CHUNK])):
                out[r["security_key"]] = r
        return out

    def _breakdown(self, t: str, kind: str, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        if kind == "etf":
            cd = self._rows(self.fx.table("etf_company_data").select("sector_weights, country_weights, top_country")
                            .eq("ticker", t).limit(1))
            cd = cd[0] if cd else {}
            pick = lambda xs, f: [{"name": x.get(f), "weight_pct": _num(x.get("weight"))}
                                  for x in (xs or [])[:12] if isinstance(x, dict)]
            return {"by_sector": pick(cd.get("sector_weights"), "sector"),
                    "by_country": pick(cd.get("country_weights"), "country"),
                    "source": "fund provider sector and country weights"}
        sm = self._security_master([h["security_key"] for h in holdings])
        rows = []
        for h in holdings:
            m = sm.get(h["security_key"]) or {}
            cat = h.get("asset_category") or m.get("asset_category")
            rows.append({"weight": h["weight"] or 0,
                         "asset": ASSET_CATEGORY.get(str(cat or "").upper(), cat or "unclassified"),
                         "sector": m.get("sector"), "cap": (m.get("cap_class") or "").replace("_", " ").lower() or None,
                         "country": m.get("country")})
        classified = sum(r["weight"] for r in rows if r["sector"])
        return {"by_asset_class": _weighted(rows, "asset")[:10],
                "by_sector": _weighted([r for r in rows if r["sector"]], "sector")[:12],
                "by_market_cap": _weighted([r for r in rows if r["cap"]], "cap")[:6],
                "by_country": _weighted([r for r in rows if r["country"]], "country")[:10],
                "weight_with_sector_data_pct": round(classified, 1),
                "source": "N-PORT holdings joined to the GQ security master"}

    # ── public: profile ──────────────────────────────────────────────────────
    def _managers(self, series_id: Optional[str], kind: str) -> List[Dict[str, Any]]:
        if not series_id:
            return []
        rows: List[Dict[str, Any]] = []
        for table in (("mf_managers", "fund_managers") if kind == "mutual_fund" else ("fund_managers", "mf_managers")):
            rows += self._safe(table, lambda tb=table: self._rows(
                self.fx.table(tb).select("manager_key, manager_name, title, role, since_year, since_month, "
                                         "since_raw, is_inception, is_current, filing_date")
                .eq("series_id", series_id).eq("is_current", True)), default=[])
        seen, out = set(), []
        today = self.today()
        for r in rows:
            k = r.get("manager_key") or r.get("manager_name")
            if k in seen:
                continue
            seen.add(k)
            out.append({"name": r.get("manager_name"), "title": r.get("title"), "role": r.get("role"),
                        "since": r.get("since_raw") or r.get("since_year"),
                        "tenure_years": _tenure_years(r.get("since_year"), r.get("since_month"), today),
                        "since_inception": bool(r.get("is_inception"))})
        out.sort(key=lambda m: -(m["tenure_years"] or 0))
        return out

    def _manager_events(self, series_ids: List[str], since: date) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        ids = [s for s in series_ids if s]
        for table in ("mf_manager_events", "fund_manager_events"):
            for k in range(0, len(ids), CHUNK):
                rows += self._safe(table, lambda tb=table, p=ids[k:k + CHUNK]: self._rows(
                    self.fx.table(tb).select("series_id, series_name, manager_name, event_type, effective_date, "
                                             "filing_date, form, filing_url, evidence")
                    .in_("series_id", p).gte("filing_date", since.isoformat())
                    .order("filing_date", desc=True).limit(200)), default=[])
        seen, out = set(), []
        for r in rows:
            k = (r.get("series_id"), r.get("manager_name"), r.get("event_type"), r.get("filing_date"))
            if k in seen:
                continue
            seen.add(k)
            out.append({"series_id": r.get("series_id"), "fund": r.get("series_name"), "manager": r.get("manager_name"),
                        "change": (r.get("event_type") or "").lower(), "effective": r.get("effective_date"),
                        "filed": r.get("filing_date"), "form": r.get("form"), "url": r.get("filing_url"),
                        "evidence": (r.get("evidence") or "")[:300] or None})
        out.sort(key=lambda e: str(e.get("filed") or ""), reverse=True)
        return out

    def _expense_breakdown(self, t: str, kind: str) -> Dict[str, Any]:
        table = "mf_expense_breakdown" if kind == "mutual_fund" else "etf_expense_breakdown"
        rows = self._safe(table, lambda: self._rows(
            self.fx.table(table).select("management_fee, other_expenses, acquired_fund_fees, gross_expense_ratio, "
                                        "fee_waiver, net_expense_ratio, form, filing_date, source_url")
            .eq("ticker", t).order("filing_date", desc=True).limit(1)), default=[])
        return _clean(rows[0]) if rows else {}

    def fund_profile(self, user_id, ticker: str) -> Dict[str, Any]:
        """What a fund is: ETF or mutual fund, manager, fees in detail, size, managers, status."""
        t, kind, no = self._fund_scoped(user_id, ticker)
        if no:
            return no

        def run():
            fees = self._safe("fee table", lambda: self._expense_breakdown(t, kind), default={})
            series = self._series_for(t, kind)
            managers = self._safe("managers", lambda: self._managers(series, kind), default=[])
            if kind == "etf":
                out = self.etf_profile(user_id, t)
                if not out.get("ok"):
                    return out
                listing = self._safe("listing state", lambda: self._rows(
                    self.fx.table("etf_listing_state").select("status, primary_exchange, closed_at")
                    .eq("ticker", t).limit(1)), default=[])
                out.update({"type": "etf", "fee_breakdown_pct": fees, "managers": managers})
                if listing:
                    out["listing_status"] = _clean(listing[0])
                    if (listing[0].get("status") or "").upper() in ("PENDING_CLOSE", "CLOSED"):
                        out["warning"] = f"{t} is marked {listing[0]['status']}: the fund is closing or closed."
                return out
            mf = self._mf_many([t]).get(t) or {}
            state = self._safe("N-PORT state", lambda: self._rows(
                self.fx.table("mf_holdings_state").select("rep_date, net_assets, holdings_count")
                .eq("series_id", series).limit(1)), default=[]) if series else []
            st = state[0] if state else {}
            scheme = self._safe("scheme events", lambda: self._rows(
                self.fx.table("mf_scheme_events").select("event_type, class_name, series_name, detail, created_at")
                .eq("ticker", t).order("created_at", desc=True).limit(5)), default=[])
            return {"ok": True, "ticker": t, "type": "mutual_fund",
                    "fund": _clean({k: mf.get(k) for k in ("series_name", "class_name", "entity_name", "fund_family",
                                                           "category", "fund_type", "inception_date", "city", "state")}),
                    "size": _clean({"net_assets_usd": _num(st.get("net_assets")) or _num(mf.get("aum")),
                                    "holdings_count": st.get("holdings_count"),
                                    "holdings_as_of": st.get("rep_date")}),
                    "expense_ratio_pct": _num(fees.get("net_expense_ratio")) or _num(mf.get("expense_ratio")),
                    "fee_breakdown_pct": fees, "managers": managers,
                    "recent_share_class_events": [_clean(s) for s in scheme],
                    "data_limits": NO_NAV}
        return self._safe("fund profile", run)

    def fund_holdings(self, user_id, ticker: str, top_n: int = 10) -> Dict[str, Any]:
        """Top holdings and the portfolio breakdown, for an ETF or a mutual fund."""
        t, kind, no = self._fund_scoped(user_id, ticker)
        if no:
            return no
        top_n = max(1, min(int(top_n or 10), 25))

        def run():
            hs, as_of = self._fund_holdings(t, kind)
            if not hs:
                return _refuse(t, "no_holdings", f"No holdings are stored for {t}.")
            top = [{"symbol": h["symbol"], "name": h["name"], "weight_pct": h["weight"],
                    **({"asset": ASSET_CATEGORY.get(str(h["asset_category"]).upper(), h["asset_category"])}
                       if h.get("asset_category") else {})} for h in hs[:top_n]]
            w = [h["weight"] or 0 for h in hs]
            return {"ok": True, "ticker": t, "type": kind, "as_of": as_of, "holdings_count": len(hs),
                    "top_holdings": top,
                    "concentration_pct": {"top_5": round(sum(w[:5]), 2), "top_10": round(sum(w[:10]), 2),
                                          "top_25": round(sum(w[:25]), 2)},
                    "breakdown": self._safe("portfolio breakdown", lambda: self._breakdown(t, kind, hs), default={}),
                    **({"note": "Mutual-fund holdings come from the latest quarterly N-PORT filing; "
                                "state the as_of date."} if kind == "mutual_fund" else {})}
        return self._safe("fund holdings", run)

    # ── public: changes and managers ─────────────────────────────────────────
    def _mf_changes(self, series_id: str, since: date) -> List[Dict[str, Any]]:
        rows, page = [], 0
        while page < 10:
            chunk = self._rows(self.fx.table("mf_holdings_changes")
                               .select("series_id, security_key, security_name, action, weight_prev, weight_new, "
                                       "rep_date_from, rep_date_to, detected_at")
                               .eq("series_id", series_id).gte("rep_date_to", since.isoformat())
                               .order("rep_date_to", desc=True).range(page * PAGE, page * PAGE + PAGE - 1))
            rows += chunk
            if len(chunk) < PAGE:
                break
            page += 1
        return rows

    @staticmethod
    def _summarise_mf_changes(rows: List[Dict[str, Any]], top: int = 10) -> Dict[str, Any]:
        def w(r, k):
            return _num(r.get(k)) or 0.0
        added = [r for r in rows if (r.get("action") or "").upper() == "ADDED"]
        removed = [r for r in rows if (r.get("action") or "").upper() == "REMOVED"]
        resized = [r for r in rows if (r.get("action") or "").upper() == "RESIZED"]
        show = lambda rs, f: [_clean({"name": r.get("security_name"), "weight_prev_pct": _num(r.get("weight_prev")),
                                      "weight_new_pct": _num(r.get("weight_new")),
                                      "change_pts": round(w(r, "weight_new") - w(r, "weight_prev"), 3),
                                      "period": f"{r.get('rep_date_from') or '?'} to {r.get('rep_date_to')}"})
                              for r in sorted(rs, key=f)[:top]]
        periods = sorted({(str(r.get("rep_date_from") or ""), str(r.get("rep_date_to") or "")) for r in rows},
                         reverse=True)
        return {"added_count": len(added), "removed_count": len(removed), "resized_count": len(resized),
                "biggest_new_positions": show(added, lambda r: -w(r, "weight_new")),
                "biggest_exits": show(removed, lambda r: -w(r, "weight_prev")),
                "biggest_increases": show([r for r in resized if w(r, "weight_new") > w(r, "weight_prev")],
                                          lambda r: -(w(r, "weight_new") - w(r, "weight_prev"))),
                "biggest_cuts": show([r for r in resized if w(r, "weight_new") < w(r, "weight_prev")],
                                     lambda r: w(r, "weight_new") - w(r, "weight_prev")),
                "filing_periods_compared": [f"{a or '?'} to {b}" for a, b in periods[:4]]}

    def fund_changes(self, user_id, ticker: str, days: int = 120) -> Dict[str, Any]:
        """What changed in a fund: holdings, managers, SEC fund actions, share-class events."""
        t, kind, no = self._fund_scoped(user_id, ticker)
        if no:
            return no
        days = max(1, min(int(days or 120), 730))
        since = self.today() - timedelta(days=days)

        def run():
            series = self._series_for(t, kind)
            mgr = self._safe("manager events", lambda: self._manager_events([series], since), default=[])
            if kind == "etf":
                out = self.etf_changes(user_id, t, min(days, 365))
                if isinstance(out, dict) and out.get("ok"):
                    out["manager_changes"] = mgr
                return out
            ch = self._safe("N-PORT changes", lambda: self._summarise_mf_changes(self._mf_changes(series, since)),
                            default={}) if series else {}
            ev = self._safe("fund events", lambda: [
                _clean({"filed": r.get("filing_date"), "form": r.get("form"), "event": r.get("event_type"),
                        "excerpt": (r.get("excerpt") or "")[:400], "url": r.get("filing_url")})
                for r in self._rows(self.fx.table("mf_fund_events")
                                    .select("filing_date, form, event_type, excerpt, filing_url, status")
                                    .contains("tickers", [t]).gte("filing_date", since.isoformat())
                                    .order("filing_date", desc=True).limit(10))
                if (r.get("status") or "") not in SKIP_EVENT_STATUS], default=[])
            return {"ok": True, "ticker": t, "type": "mutual_fund", "days": days, "holdings_changes": ch,
                    "manager_changes": mgr, "fund_actions": ev,
                    "note": "Mutual-fund holdings change only when a new quarterly N-PORT filing appears; "
                            "name the periods compared."}
        return self._safe("fund changes", run)

    def fund_managers(self, user_id, ticker: Optional[str] = None, days: int = 365) -> Dict[str, Any]:
        """One fund's management team and tenure, or recent manager changes across every watchlist fund."""
        days = max(1, min(int(days or 365), 1825))
        since = self.today() - timedelta(days=days)
        if ticker:
            t, kind, no = self._fund_scoped(user_id, ticker)
            if no:
                return no

            def one():
                series = self._series_for(t, kind)
                team = self._managers(series, kind)
                return {"ok": True, "ticker": t, "type": kind, "current_team": team,
                        "longest_tenure_years": team[0]["tenure_years"] if team else None,
                        "changes": self._manager_events([series], since), "days": days,
                        **({} if team else {"note": "No manager roster is stored for this fund."})}
            return self._safe("fund managers", one)

        def all_funds():
            funds = self._watch_funds(user_id)
            sids = {t: self._series_for(t, k) for t, k in funds.items()}
            by_series = {}
            for t, s in sids.items():
                if s:
                    by_series.setdefault(s, []).append(t)
            ev = self._manager_events(list(by_series), since)
            for e in ev:
                e["tickers"] = by_series.get(e.pop("series_id"), [])
            return {"ok": True, "days": days, "funds_checked": len(funds),
                    "funds_without_series_id": sorted(t for t, s in sids.items() if not s),
                    "changes": ev[:40], "funds_with_changes": sorted({t for e in ev for t in e["tickers"]})}
        return self._safe("watchlist manager changes", all_funds)

    def watchlist_mf_changes(self, user_id, days: int = 120) -> Dict[str, Any]:
        """N-PORT holdings changes for every mutual fund on the watchlist."""
        since = self.today() - timedelta(days=max(1, min(int(days or 120), 730)))
        funds = {t: k for t, k in self._watch_funds(user_id).items() if k == "mutual_fund"}

        def one(t):
            s = self._series_for(t, "mutual_fund")
            if not s:
                return t, None
            c = self._summarise_mf_changes(self._mf_changes(s, since), top=3)
            return t, c
        res = dict(self._map(lambda t: self._safe("mf changes", lambda: one(t), default=(t, None)), list(funds)))
        changed = {t: c for t, c in res.items() if c and (c["added_count"] or c["removed_count"] or c["resized_count"])}
        return {"funds_checked": len(funds),
                "changed": [{"fund": t, **{k: c[k] for k in ("added_count", "removed_count", "resized_count",
                                                             "biggest_new_positions", "biggest_exits",
                                                             "filing_periods_compared")}}
                            for t, c in sorted(changed.items(), key=lambda x: -(x[1]["added_count"] + x[1]["removed_count"]))],
                "no_new_filing_in_window": sorted(t for t in funds if t not in changed),
                "note": "Quarterly N-PORT comparisons; a fund with no new filing in the window shows no change."}

    # ── public: comparison, overlap, exposure ────────────────────────────────
    def compare_funds(self, user_id, tickers: Optional[List[str]] = None, sort_by: Optional[str] = None,
                      order: str = "asc") -> Dict[str, Any]:
        """Side by side: fees, size, holdings count, concentration, manager tenure, and ETF returns."""
        refused = []
        if tickers:
            want = []
            for raw in tickers[:40]:
                t, no = self._scoped(user_id, raw)
                (refused.append(_norm(raw)) if no else want.append(t))
            kinds = {t: k for t, k in self._kinds(want).items() if k in ("etf", "mutual_fund")}
            not_funds = [t for t in want if t not in kinds]
        else:
            kinds, not_funds = self._watch_funds(user_id), []
        if not kinds:
            return {"ok": False, "reason": "no_funds", "message": "No ETFs or mutual funds to compare.",
                    "not_on_watchlist": refused}
        sort_by = sort_by if sort_by in COMPARE_METRICS else None

        def one(item):
            t, kind = item
            row: Dict[str, Any] = {"ticker": t, "type": kind}
            series = self._series_for(t, kind)
            if kind == "mutual_fund":
                mf = self._mf_many([t]).get(t) or {}
                row["name"] = " ".join(x for x in (mf.get("series_name"), mf.get("class_name")) if x) or None
                row["category"] = mf.get("category")
                st = self._rows(self.fx.table("mf_holdings_state").select("net_assets, holdings_count, rep_date")
                                .eq("series_id", series).limit(1)) if series else []
                st = st[0] if st else {}
                row["aum"] = _num(st.get("net_assets")) or _num(mf.get("aum"))
                row["holdings_as_of"] = st.get("rep_date")
                exp = _num(mf.get("expense_ratio"))
            else:
                inst = self._instrument(t) or {}
                row["name"] = inst.get("name")
                row["category"] = inst.get("etf_category")
                m = self._rows(self.fx.table("etf_master").select("aum, name").eq("ticker", t).limit(1))
                row["aum"] = _num((m[0] if m else {}).get("aum"))
                exp = _num(inst.get("expense_ratio"))
                r = self._returns(inst["id"]) if inst.get("id") else None
                row["ret_1y"], row["ret_5y"] = (r or {}).get("ret_1y"), (r or {}).get("ret_5y")
            fees = self._expense_breakdown(t, kind)
            row["net_expense_ratio"] = _num(fees.get("net_expense_ratio")) or exp
            row["management_fee"] = _num(fees.get("management_fee"))
            if _num(fees.get("fee_waiver")):
                row["fee_waiver"] = _num(fees.get("fee_waiver"))
            hs, as_of = self._fund_holdings(t, kind)
            w = [h["weight"] or 0 for h in hs]
            row["holdings_count"] = len(hs) or None
            row["top10_weight_pct"] = round(sum(w[:10]), 2) if hs else None
            team = self._managers(series, kind)
            row["manager_count"] = len(team) or None
            row["longest_manager_tenure_years"] = team[0]["tenure_years"] if team else None
            return {k: v for k, v in row.items() if v is not None}

        rows = self._map(lambda it: self._safe("compare fund", lambda: one(it), default={"ticker": it[0]}),
                         sorted(kinds.items()))
        if sort_by:
            have = [r for r in rows if r.get(sort_by) is not None]
            have.sort(key=lambda r: r[sort_by], reverse=(order == "desc"))
            rows = have + [r for r in rows if r.get(sort_by) is None]
            for n, r in enumerate(have, 1):
                r["rank"] = n
        return {"ok": True, "sorted_by": sort_by, "order": order, "funds": rows,
                "not_funds": not_funds, "not_on_watchlist": refused,
                "note": "Expense ratios are percent of assets. Mutual funds have no return data in GQuants."}

    def fund_overlap(self, user_id, ticker_a: str, ticker_b: str) -> Dict[str, Any]:
        """Holdings two watchlist funds share (ETF or mutual fund, in any mix)."""
        a, ka, no = self._fund_scoped(user_id, ticker_a)
        if no:
            return no
        b, kb, no = self._fund_scoped(user_id, ticker_b)
        if no:
            return no
        if ka == kb == "etf":
            return self.etf_overlap(user_id, a, b)

        def run():
            (ha_l, da), (hb_l, db) = self._fund_holdings(a, ka), self._fund_holdings(b, kb)
            ha, hb = {}, {}
            for src, dst in ((ha_l, ha), (hb_l, hb)):
                for h in src:
                    if h["key"]:
                        dst[h["key"]] = dst.get(h["key"], 0.0) + (h["weight"] or 0)
            if not ha or not hb:
                return {"ok": False, "reason": "no_holdings", "message": f"Holdings are missing for {a if not ha else b}."}
            names = {h["key"]: h["name"] for h in ha_l + hb_l}
            common = set(ha) & set(hb)
            shared = sorted(({"holding": names.get(k) or k, f"weight_in_{a}": round(ha[k], 3),
                              f"weight_in_{b}": round(hb[k], 3), "overlap_weight": round(min(ha[k], hb[k]), 3)}
                             for k in common), key=lambda x: -x["overlap_weight"])
            return {"ok": True, "funds": [a, b], "types": {a: ka, b: kb}, "holdings_as_of": {a: da, b: db},
                    "holdings_count": {a: len(ha), b: len(hb)}, "common_holdings": len(common),
                    "weight_overlap_pct": round(sum(x["overlap_weight"] for x in shared), 2),
                    "top_shared": shared[:10],
                    "note": "Matched by ticker, else issuer name; a fund's holdings date can differ from the other's."}
        return self._safe("fund overlap", run)

    def fund_exposure(self, user_id, symbol: str) -> Dict[str, Any]:
        """Which ETFs AND mutual funds on the watchlist hold a stock, and at what weight."""
        s = _norm(symbol)

        def run():
            etf = self.etf_exposure(user_id, s)
            funds = {t: k for t, k in self._watch_funds(user_id).items() if k == "mutual_fund"}
            sids = {self._series_for(t, "mutual_fund"): t for t in funds}
            sids.pop(None, None)
            keys = [r["security_key"] for r in self._rows(self.fx.table("mf_security_master")
                                                          .select("security_key").eq("ticker", s).limit(20))]
            rows = []
            ids = list(sids)
            for k in range(0, len(ids), CHUNK):
                part = ids[k:k + CHUNK]
                rows += self._rows(self.fx.table("mf_holdings").select("series_id, security_name, portfolio_pct, rep_date")
                                   .in_("series_id", part).eq("ticker", s))
                if keys:
                    rows += self._rows(self.fx.table("mf_holdings").select("series_id, security_name, portfolio_pct, rep_date")
                                       .in_("series_id", part).in_("security_key", keys))
            best: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                t = sids.get(r.get("series_id"))
                w = _num(r.get("portfolio_pct")) or 0
                if t and w >= (best.get(t, {}).get("weight_pct") or -1):
                    best[t] = {"fund": t, "weight_pct": w, "as_of": r.get("rep_date")}
            mf_hits = sorted(best.values(), key=lambda x: -x["weight_pct"])
            held = [{"fund": h["etf"], "type": "etf", "weight_pct": h["weight_pct"], "as_of": h.get("as_of")}
                    for h in (etf.get("held_by") or [])] + [{**h, "type": "mutual_fund"} for h in mf_hits]
            held.sort(key=lambda x: -float(x["weight_pct"] or 0))
            return {"ok": True, "stock": s, "name": etf.get("name") or (rows[0].get("security_name") if rows else None),
                    "held_by": held, "etfs_checked": etf.get("etfs_checked", 0), "mutual_funds_checked": len(funds),
                    "average_weight_where_held_pct": round(statistics.mean(float(h["weight_pct"] or 0) for h in held), 3)
                    if held else None}
        return self._safe("fund exposure", run)

    # ── public: look-through ─────────────────────────────────────────────────
    def fund_lookthrough(self, user_id, ticker: str, window: str = "1m", top_n: int = 15) -> Dict[str, Any]:
        """A fund's top holdings with their own returns, contribution to the fund, RSI and 52-week position."""
        t, kind, no = self._fund_scoped(user_id, ticker)
        if no:
            return no
        col = WINDOWS.get((window or "1m").lower(), "ret_1m")
        top_n = max(3, min(int(top_n or 15), 30))

        def run():
            hs, as_of = self._fund_holdings(t, kind)
            if not hs:
                return _refuse(t, "no_holdings", f"No holdings are stored for {t}.")
            top = hs[:top_n]
            if kind == "mutual_fund":
                sm = self._security_master([h["security_key"] for h in top if not h["symbol"]])
                for h in top:
                    h["symbol"] = h["symbol"] or (sm.get(h["security_key"]) or {}).get("ticker")
            syms = [(h["symbol"] or "").upper().replace("/", ".") for h in top]
            insts = self._instruments_for([s for s in syms if s])
            ids = {i["id"]: s for s, i in insts.items()}
            rets = self._returns_many(list(ids), f"{col}, as_of")
            rsi, w52 = {}, {}
            for k in range(0, len(ids), CHUNK):
                part = list(ids)[k:k + CHUNK]
                for r in self._safe("RSI table", lambda p=part: self._rows(
                        self.hm.table("tech_indicators_rsi").select("instrument_id, rsi14, rsi_zone, rsi_divergence")
                        .in_("instrument_id", p).eq("timeframe", "D")), default=[]):
                    rsi[r["instrument_id"]] = r
                for r in self._safe("52-week table", lambda p=part: self._rows(
                        self.hm.table("tech_indicators_52w").select("instrument_id, dist_from_high_pct, is_new_52w_high, "
                                                                    "is_new_52w_low").in_("instrument_id", p)), default=[]):
                    w52[r["instrument_id"]] = r
            rows, covered, contrib, ret_asof = [], 0.0, 0.0, None
            for h, s in zip(top, syms):
                i = insts.get(s) or {}
                r = rets.get(i.get("id")) or {}
                v = _num(r.get(col))
                ret_asof = ret_asof or r.get("as_of")
                row = {"holding": h["name"], "symbol": s or None, "weight_pct": h["weight"]}
                if v is not None:
                    c = round((h["weight"] or 0) * v / 100, 3)
                    row.update({f"return_{window}_pct": round(v, 2), "contribution_pts": c})
                    covered += h["weight"] or 0
                    contrib += c
                x = rsi.get(i.get("id")) or {}
                if x:
                    row["rsi_14"] = _num(x.get("rsi14"))
                    row["rsi_zone"] = x.get("rsi_zone")
                    if x.get("rsi_divergence"):
                        row["rsi_divergence"] = x["rsi_divergence"]
                y = w52.get(i.get("id")) or {}
                if y:
                    row["pct_below_52w_high"] = _num(y.get("dist_from_high_pct"))
                    if y.get("is_new_52w_high"):
                        row["new_52w_high"] = True
                    if y.get("is_new_52w_low"):
                        row["new_52w_low"] = True
                rows.append(row)
            fund_ret = None
            if kind == "etf":
                inst = self._instrument(t) or {}
                fund_ret = _num((self._returns(inst["id"]) or {}).get(col)) if inst.get("id") else None
            priced = [r for r in rows if "contribution_pts" in r]
            return {"ok": True, "ticker": t, "type": kind, "window": window, "holdings_as_of": as_of,
                    "returns_as_of": ret_asof, "holdings": rows,
                    "top_holdings_weight_pct": round(sum(h["weight"] or 0 for h in top), 2),
                    "weight_with_price_data_pct": round(covered, 2),
                    "contribution_from_these_holdings_pts": round(contrib, 2),
                    **({"fund_return_pct": fund_ret} if fund_ret is not None else {}),
                    "top_contributors": sorted(priced, key=lambda r: -r["contribution_pts"])[:3],
                    "top_detractors": sorted(priced, key=lambda r: r["contribution_pts"])[:3],
                    "note": ("Contribution = weight x holding return, using today's weights, so it approximates "
                             "the fund's move." + (" Mutual-fund weights are from the quarterly N-PORT filing."
                                                   if kind == "mutual_fund" else ""))}
        return self._safe("fund look-through", run)

    # ── public: market context ───────────────────────────────────────────────
    def market_overview(self, window: str = "1d") -> Dict[str, Any]:
        """US market context: benchmarks, sector performance across all US stocks, breadth, highs/lows."""
        col = WINDOWS.get((window or "1d").lower(), "ret_1d")
        hit = self._sector.get(("overview", col))
        if hit is not None:
            return hit

        def run():
            bench = self._instruments_for(list(self.BENCH))
            br = self._returns_many([i["id"] for i in bench.values()], f"{col}, as_of")
            benchmarks = {t: _num((br.get(i["id"]) or {}).get(col)) for t, i in bench.items()}
            insts, page = [], 0
            while page < 10:
                chunk = self._rows(self.hm.table("instruments").select("id, sector, market_cap_usd")
                                   .eq("universe", "us_stocks").eq("is_active", True)
                                   .range(page * PAGE, page * PAGE + PAGE - 1))
                insts += chunk
                if len(chunk) < PAGE:
                    break
                page += 1
            ids = [i["id"] for i in insts]
            rets = self._returns_many(ids, f"{col}, as_of")
            secs: Dict[str, List[Tuple[float, float]]] = {}
            vals, as_of = [], None
            for i in insts:
                r = rets.get(i["id"]) or {}
                v = _num(r.get(col))
                if v is None:
                    continue
                as_of = as_of or r.get("as_of")
                vals.append(v)
                secs.setdefault(i.get("sector") or "Others", []).append((v, _num(i.get("market_cap_usd")) or 0))
            sectors = []
            for s, xs in secs.items():
                cap = sum(c for _, c in xs)
                sectors.append({"sector": s, "stocks": len(xs),
                                "median_pct": round(statistics.median(v for v, _ in xs), 2),
                                "cap_weighted_pct": round(sum(v * c for v, c in xs) / cap, 2) if cap else None,
                                "pct_up": round(sum(v > 0 for v, _ in xs) / len(xs) * 100, 1)})
            sectors.sort(key=lambda x: -(x["cap_weighted_pct"] if x["cap_weighted_pct"] is not None else x["median_pct"]))

            def count(table, **eqs):
                q = self.hm.table(table).select("instrument_id").eq("universe", "us_stocks")
                for k, v in eqs.items():
                    q = q.eq(k, v)
                n, page = 0, 0
                while page < 10:
                    chunk = self._rows(q.range(page * PAGE, page * PAGE + PAGE - 1))
                    n += len(chunk)
                    if len(chunk) < PAGE:
                        break
                    page += 1
                return n
            extremes = self._safe("market extremes", lambda: {
                "new_52w_highs": count("tech_indicators_52w", is_new_52w_high=True),
                "new_52w_lows": count("tech_indicators_52w", is_new_52w_low=True),
                "rsi_overbought": count("tech_indicators_rsi", timeframe="D", rsi_zone="overbought"),
                "rsi_oversold": count("tech_indicators_rsi", timeframe="D", rsi_zone="oversold")}, default={})
            return {"ok": bool(vals), "window": window, "as_of": as_of, "benchmarks_pct": benchmarks,
                    "breadth": {"stocks": len(vals), "up": sum(v > 0 for v in vals), "down": sum(v < 0 for v in vals),
                                "pct_up": round(sum(v > 0 for v in vals) / len(vals) * 100, 1) if vals else None,
                                "median_pct": round(statistics.median(vals), 2) if vals else None},
                    "sectors": sectors, "us_stock_extremes_today": extremes,
                    "note": "Market context across every active US stock on the GQuants heatmap (GICS sectors). "
                            "Sector rows are aggregates, not tickers to discuss on their own."}
        out = self._safe("market overview", run)
        if isinstance(out, dict) and out.get("ok"):
            self._sector.put(("overview", col), out)
        return out

    def watchlist_fund_changes(self, user_id, days: int = 90) -> Dict[str, Any]:
        """ETF and mutual-fund holdings changes across the whole watchlist, in one answer."""
        etf = self._safe("ETF changes", lambda: self.watchlist_etf_changes(user_id, days), default={})
        mf = self._safe("mutual-fund changes", lambda: self.watchlist_mf_changes(user_id, max(days, 120)), default={})
        return {"ok": bool((etf or {}).get("ok") or (mf or {}).get("funds_checked")), "days": days,
                "etfs": etf, "mutual_funds": mf}
