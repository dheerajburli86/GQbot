import json
from datetime import timedelta

import pytest

from fakes import TODAY, fmp, series, world

from gq_bot import financials as fin, technicals as tech
from gq_bot.data import GQData
from gq_bot.tools import ToolRunner


def make(**kw):
    fx, hm = world(**kw)
    return GQData(fx, hm, today=lambda: TODAY, workers=1)


# ── pure maths ────────────────────────────────────────────────────────────────
def test_ttm_and_growth_from_fmp_rows():
    p = fin.periods_from_store(fmp(100e9, 25e9))
    q = p["quarterly"]
    assert len(q) == 8 and q[0]["date"] > q[1]["date"]           # newest first
    ttm_rev = sum(100e9 * 1.02 ** -i for i in range(4))
    prev_rev = sum(100e9 * 1.02 ** -i for i in range(4, 8))
    a = fin.analyse(p)
    assert a["basis"].startswith("TTM")
    assert a["ttm"]["revenue"] == pytest.approx(ttm_rev)
    assert a["growth_yoy_pct"]["revenue"] == round((ttm_rev - prev_rev) / prev_rev * 100, 2)
    assert a["profitability_pct"]["gross_margin"] == 45.0 and a["profitability_pct"]["operating_margin"] == 30.0
    assert a["balance_sheet"]["debt_to_equity"] == round(100 / 60, 2)
    assert a["balance_sheet"]["current_ratio"] == 1.2
    assert a["multi_year"]["years"] == 3 and a["multi_year"]["revenue_cagr_pct"] == pytest.approx(8.0, abs=0.01)
    assert a["cash_flow"]["payout_ratio_pct"] == 15.0
    assert a["quarterly_trend"][0]["revenue_yoy_pct"] == round((1 - 1.02 ** -4) / 1.02 ** -4 * 100, 2)


def test_legacy_snake_case_blob_is_understood():
    legacy = {"quarterly": [{"period": f"Q{i}", "fiscal_year": "2026", "date": f"2026-0{9 - i}-30",
                             "income_statement": {"revenue": 10.0, "net_income": 2.0, "gross_profit": 5.0},
                             "balance_sheet": {"total_equity": 20.0, "total_debt": 10.0, "cash": 4.0},
                             "cash_flow": {"operating_cash_flow": 3.0, "capex": -1.0}} for i in range(4)],
              "annual": []}
    a = fin.analyse(fin.periods_from_store(legacy))
    assert a["ttm"]["revenue"] == 40.0 and a["profitability_pct"]["gross_margin"] == 50.0
    assert a["cash_flow"]["free_cash_flow"] == 8.0      # derived: OCF - |capex|
    assert a["balance_sheet"]["net_debt"] == 6.0


def test_valuation_multiples():
    p = fin.periods_from_store(fmp(100e9, 25e9))
    ttm_ni = sum(25e9 * 1.02 ** -i for i in range(4))
    v = fin.valuation(p, 3000e9, 180.0)
    assert v["pe_ttm"] == round(3000e9 / ttm_ni, 2)
    assert v["enterprise_value"] == 3000e9 + 100e9 - 50e9
    assert v["dividend_yield_pct"] == round(ttm_ni * 0.15 / 3000e9 * 100, 2)
    loss = fin.valuation(fin.periods_from_store(fmp(10e9, -1e9)), 50e9, 10.0)
    assert loss["pe_ttm"] is None and "negative" in loss["earnings"]


def test_rsi_and_trend_on_known_series():
    up = [{"date": (TODAY - timedelta(days=300 - i)).isoformat(), "close": 100 + i} for i in range(300)]
    t = tech.technicals(up)
    assert t["rsi_14"] == 100.0 and t["trend"].startswith("uptrend")
    assert t["range_52w"]["pct_below_high"] == 0.0 and t["above_sma_200"] is True
    assert t["last_50_200_cross"]["type"] == "none found" and t["sma_50_above_sma_200"] is True
    dn = [{"date": b["date"], "close": 500 - i} for i, b in enumerate(up)]
    assert tech.technicals(dn)["trend"].startswith("downtrend")
    assert tech.technicals(up[:10])["reason"] == "not_enough_history"


def test_risk_against_itself_is_beta_one():
    bars = series(100.0, 300)
    r = tech.risk(bars, bars, "SELF")
    assert r["vs_benchmark"]["beta"] == 1.0 and r["vs_benchmark"]["correlation"] == 1.0
    assert r["max_drawdown"]["pct"] <= 0 and r["volatility_ann_pct"] > 0


# ── data-layer tools ──────────────────────────────────────────────────────────
NEW_TOOLS = ("financials", "valuation", "peer_comparison", "technicals", "risk", "company_profile")


@pytest.mark.parametrize("name", NEW_TOOLS)
def test_new_tools_refuse_tickers_off_the_watchlist(name):
    r = getattr(make(), name)("u1", "TSLA")
    assert r["ok"] is False and r["reason"] == "not_on_watchlist"


def test_financials_for_a_stock_and_refusal_for_an_etf():
    d = make()
    f = d.financials("u1", "AAPL")
    assert f["ok"] and f["reported_currency"] == "USD" and len(f["quarterly_trend"]) == 8
    assert d.financials("u1", "QQQ")["reason"] == "is_etf"
    assert d.financials("u1", "MSFT")["reason"] == "no_fundamentals"
    assert d.financials("u1", "ABALX")["reason"] == "no_price_data"


def test_valuation_brings_peer_medians_and_skips_non_usd_peers():
    v = make().valuation("u1", "AAPL")
    assert v["ok"] and v["valuation"]["pe_ttm"] > 0
    pc = v["peer_comparison"]
    assert pc["peer_group"]["by"] == "industry" and set(pc["peers_used"]) == {"AMD", "INTC", "ORCL", "DELL"}
    assert pc["peer_medians"]["pe_ttm"]["n"] == 2          # INTC loses money, DELL reports in EUR


def test_peer_comparison_ranks_the_subject():
    p = make().peer_comparison("u1", "AAPL")
    assert p["ok"] and p["subject"]["ticker"] == "AAPL"
    ranks = p["subject_percentile_vs_peers"]
    assert ranks["ret_ytd"] is None                 # AAPL has no YTD figure: no rank, not a fake 0
    assert ranks["net_margin"] == 100.0             # 25% net margin beats every peer
    assert {x["ticker"] for x in p["peers"]} == {"AMD", "INTC", "ORCL", "DELL"}


def test_technicals_and_risk_from_price_history():
    d = make()
    t = d.technicals("u1", "MSFT")
    assert t["ok"] and t["moving_averages"]["sma_200"] and t["volume"]["last_vs_20d_avg_x"] == 3.0
    r = d.risk("u1", "MSFT", benchmark="TSLA")
    assert r["reason"] == "not_on_watchlist"                 # benchmark is scoped too
    r = d.risk("u1", "MSFT", benchmark="SPY")
    assert r["ok"] and r["sessions"] > 200


def test_company_profile_prefers_gq_notes():
    p = make().company_profile("u1", "AAPL")
    assert p["gq_company_notes"] == "Apple makes iPhones." and p["profile"]["ceo"] == "Tim Cook"
    assert "description" not in p


def test_watchlist_overview_and_ranking():
    d = make()
    o = d.watchlist_overview("u1", "1w")
    assert o["top_gainers"][0]["ticker"] == "QQQ" and o["top_losers"][0]["ticker"] == "AAPL"
    assert o["not_covered"] == ["ABALX"] and o["breadth"]["down"] == 3
    assert any(e["ticker"] == "AAPL" for e in o["high_impact_events_last_7d"])
    r = d.rank_watchlist("u1", "return_1w", order="asc")
    assert [x["ticker"] for x in r["ranked"]][:2] == ["AAPL", "MSFT"]
    pe = d.rank_watchlist("u1", "pe_ttm")
    assert [x["ticker"] for x in pe["ranked"]] == ["AAPL"]
    assert d.rank_watchlist("u1", "made_up")["reason"] == "unknown_metric"
    assert d.rank_watchlist("u1", "rsi_14")["ranked"][0]["ticker"] == "MSFT"


def test_etf_overlap_and_exposure():
    d = make()
    o = d.etf_overlap("u1", "QQQ", "VOO")
    assert o["common_holdings"] == 2 and o["weight_overlap_pct"] == 13.5
    assert d.etf_overlap("u1", "QQQ", "TSLA")["reason"] == "not_on_watchlist"
    e = d.etf_exposure("u1", "nvda")
    assert [h["etf"] for h in e["held_by"]] == ["QQQ"] and e["etfs_checked"] == 2


def test_chart_tool_returns_an_image_that_the_runner_lifts_out():
    fx, hm = world()
    r = ToolRunner(GQData(fx, hm, today=lambda: TODAY, workers=1), "u1")
    out = json.loads(r.run("price_chart", {"tickers": ["MSFT", "SPY", "TSLA"], "days": 200}))
    assert out["ok"] and out["charted"] == ["MSFT", "SPY"] and "_image" not in out
    assert out["refused"][0]["reason"] == "not_on_watchlist"
    assert len(r.images) == 1 and r.images[0][:8] == b"\x89PNG\r\n\x1a\n"


def test_a_real_cross_is_dated():
    xs = [200 - i for i in range(260)] + [-60 + 3 * i for i in range(1, 200)]
    bars = [{"date": (TODAY - timedelta(days=len(xs) - i)).isoformat(), "close": 100 + x}
            for i, x in enumerate(xs)]
    c = tech.technicals(bars)["last_50_200_cross"]
    assert c["type"].startswith("golden cross") and c["date"] > bars[260]["date"]


def test_rank_named_tickers_returns_all_of_them_and_only_them():
    r = make().rank_watchlist("u1", "return_1w", tickers=["qqq", "MSFT", "AAPL", "TSLA"])
    got = [x["ticker"] for x in r["ranked"]]
    assert set(got) == {"QQQ", "MSFT", "AAPL"} and r["ranked"][0]["rank"] == 1
    assert r["not_on_watchlist"] == ["TSLA"]
    assert make().rank_watchlist("u1", "return_1w", tickers=["TSLA"])["reason"] == "not_on_watchlist"


def test_rank_rsi_carries_the_zone_rule():
    assert "above 70" in make().rank_watchlist("u1", "rsi_14")["rsi_zones"]


def test_key_sessions_carry_the_market_move_the_same_day():
    ks = make().explain_move("u1", "MSFT", days=200)["key_sessions"]
    days = ks["biggest_down_days"] + ks["biggest_up_days"]
    hit = [m for m in days if "spy_same_day_pct" in m]
    assert hit and hit[0]["date"] == "2026-10-14" and hit[0]["spy_same_day_pct"] == -0.18
    # sessions with no SPY bar get no figure rather than an invented one
    assert all("spy_same_day_pct" not in m for m in days if m["date"] < "2026-10-01")


def test_markdown_is_stripped_for_telegram():
    from gq_bot.agent import plain_text
    out = plain_text("## Head\n**VTV** is cheaper.\n* one\n  * two\n- three")
    assert out == "Head\nVTV is cheaper.\n- one\n  - two\n- three"
