import json

import pytest

from fakes import TODAY, world

from gq_bot.data import GQData
from gq_bot.tools import ToolRunner


def make(**kw):
    fx, hm = world(**kw)
    return GQData(fx, hm, today=lambda: TODAY, workers=1)


def test_watchlist_knows_a_mutual_fund_by_name():
    w = {x["ticker"]: x for x in make().watchlist_details("u1")}
    assert w["ABALX"] == {"ticker": "ABALX", "type": "mutual fund", "name": "American Balanced Fund Class A"}
    assert w["QQQ"]["type"] == "etf" and w["AAPL"]["type"] == "stock"


@pytest.mark.parametrize("name,args", [("fund_profile", ("TSLA",)), ("fund_holdings", ("TSLA",)),
                                       ("fund_changes", ("TSLA",)), ("fund_lookthrough", ("TSLA",))])
def test_fund_tools_stay_on_the_watchlist(name, args):
    assert getattr(make(), name)("u1", *args)["reason"] == "not_on_watchlist"


def test_fund_tools_refuse_a_stock():
    assert make().fund_profile("u1", "AAPL")["reason"] == "not_a_fund"


def test_mutual_fund_profile_has_fees_managers_and_size_but_no_prices():
    p = make().fund_profile("u1", "ABALX")
    assert p["ok"] and p["type"] == "mutual_fund"
    assert p["fund"]["fund_family"] == "American Funds"
    assert p["size"] == {"net_assets_usd": 260e9, "holdings_count": 4, "holdings_as_of": "2026-06-30"}
    assert p["fee_breakdown_pct"]["management_fee"] == 0.21 and p["expense_ratio_pct"] == 0.56  # newest filing
    assert [m["name"] for m in p["managers"]] == ["Jane Doe", "Bob Roe"]          # current only, longest first
    assert p["managers"][0]["tenure_years"] > 16 and "no NAV" in p["data_limits"]


def test_etf_profile_gains_fee_table_and_managers():
    p = make().fund_profile("u1", "QQQ")
    assert p["ok"] and p["type"] == "etf" and p["fee_breakdown_pct"]["management_fee"] == 0.2
    assert p["managers"][0]["name"] == "Index Team" and "price" in p


def test_mutual_fund_holdings_with_asset_sector_and_cap_breakdown():
    h = make().fund_holdings("u1", "ABALX", top_n=2)
    assert h["as_of"] == "2026-06-30" and h["holdings_count"] == 4
    assert [x["name"] for x in h["top_holdings"]] == ["US Treasury Note 2031", "Microsoft Corp"]
    b = h["breakdown"]
    assert b["by_asset_class"][0] == {"name": "debt", "weight_pct": 10.0}
    assert {"name": "short-term / cash vehicle", "weight_pct": 3.0} in b["by_asset_class"]
    assert b["by_sector"] == [{"name": "Technology", "weight_pct": 9.0}]
    assert b["by_market_cap"] == [{"name": "mega cap", "weight_pct": 9.0}]
    assert h["concentration_pct"]["top_5"] == 22.0


def test_mutual_fund_changes_split_adds_exits_and_resizes():
    c = make().fund_changes("u1", "ABALX", days=200)
    ch = c["holdings_changes"]
    assert (ch["added_count"], ch["removed_count"], ch["resized_count"]) == (1, 1, 1)
    assert ch["biggest_exits"][0]["name"] == "Exxon Mobil"
    assert ch["biggest_increases"][0]["change_pts"] == 1.0
    assert ch["filing_periods_compared"] == ["2026-03-31 to 2026-06-30"]
    assert c["manager_changes"][0]["manager"] == "Old Hand" and c["fund_actions"][0]["event"] == "FEE_CHANGE"


def test_manager_changes_across_the_watchlist():
    m = make().fund_managers("u1")
    assert m["funds_with_changes"] == ["ABALX"] and m["changes"][0]["tickers"] == ["ABALX"]
    one = make().fund_managers("u1", "ABALX")
    assert one["longest_tenure_years"] > 16 and len(one["current_team"]) == 2


def test_compare_funds_ranks_by_fee_and_covers_mutual_funds():
    r = make().compare_funds("u1", sort_by="net_expense_ratio")
    got = {x["ticker"]: x for x in r["funds"]}
    assert set(got) == {"QQQ", "VOO", "ABALX"}
    assert got["QQQ"]["rank"] == 1 and got["ABALX"]["net_expense_ratio"] == 0.56
    assert got["ABALX"]["top10_weight_pct"] == 22.0 and got["ABALX"]["longest_manager_tenure_years"] > 16
    assert make().compare_funds("u1", ["AAPL", "TSLA"])["reason"] == "no_funds"


def test_overlap_between_a_mutual_fund_and_an_etf():
    o = make().fund_overlap("u1", "ABALX", "QQQ")
    assert o["ok"] and o["common_holdings"] == 1          # AAPL by ticker; MSFT has no ticker in N-PORT here
    assert o["top_shared"][0]["overlap_weight"] == 4.0
    assert make().fund_overlap("u1", "QQQ", "VOO")["common_holdings"] == 2   # ETF pair keeps the old path


def test_exposure_finds_etfs_and_mutual_funds():
    e = make().fund_exposure("u1", "MSFT")          # in the MF only via the security master's ticker
    assert [(h["fund"], h["type"]) for h in e["held_by"]] == [("QQQ", "etf"), ("VOO", "etf"), ("ABALX", "mutual_fund")]
    assert e["mutual_funds_checked"] == 1


def test_lookthrough_gives_contribution_rsi_and_52w():
    lt = make().fund_lookthrough("u1", "QQQ", window="1w")
    rows = {r["symbol"]: r for r in lt["holdings"]}
    assert rows["AAPL"]["contribution_pts"] == round(8.9 * -9.5 / 100, 3)
    assert rows["AAPL"]["rsi_zone"] == "oversold" and rows["AAPL"]["new_52w_low"] is True
    assert rows["NVDA"]["new_52w_high"] is True and lt["fund_return_pct"] == -0.8
    assert lt["top_detractors"][0]["symbol"] == "AAPL" and lt["top_contributors"][0]["symbol"] == "NVDA"
    mf = make().fund_lookthrough("u1", "ABALX", window="1w")
    assert {r["symbol"] for r in mf["holdings"] if r.get("symbol")} >= {"AAPL", "MSFT"}   # ticker from security master


def test_market_overview_breadth_sectors_and_extremes():
    m = make().market_overview("1w")
    assert m["ok"] and m["benchmarks_pct"]["SPY"] == -0.5
    tech = next(s for s in m["sectors"] if s["sector"] == "Technology")
    assert tech["stocks"] == 7 and tech["pct_up"] > 0
    assert m["us_stock_extremes_today"] == {"new_52w_highs": 1, "new_52w_lows": 1, "rsi_overbought": 1,
                                             "rsi_oversold": 1}


def test_technicals_carry_dashboard_indicator_tables():
    t = make().technicals("u1", "MSFT")
    g = t["gq_dashboard_indicators"]
    assert [p["pattern_name"] for p in g["candlestick_patterns"]] == ["Hammer"]   # latest day only
    assert g["all_time_high"]["dist_from_ath_pct"] == -1.5


def test_watchlist_fund_changes_covers_both_kinds():
    w = make().watchlist_fund_changes("u1", 200)
    assert w["etfs"]["changed"][0]["etf"] == "QQQ"
    assert w["mutual_funds"]["changed"][0]["fund"] == "ABALX"


def test_a_price_question_on_a_mutual_fund_explains_why_there_is_none():
    fx, hm = world()
    r = ToolRunner(GQData(fx, hm, today=lambda: TODAY, workers=1), "u1")
    out = json.loads(r.run("get_technicals", {"ticker": "ABALX"}))
    assert out["reason"] == "mutual_fund_no_prices" and "get_fund_holdings" in out["message"]


def test_old_etf_tool_names_still_work():
    fx, hm = world()
    r = ToolRunner(GQData(fx, hm, today=lambda: TODAY, workers=1), "u1")
    assert json.loads(r.run("get_etf_holdings", {"ticker": "ABALX"}))["type"] == "mutual_fund"
