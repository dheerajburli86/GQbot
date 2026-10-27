import json

from fakes import TODAY, series, world

from gq_bot.data import GQData
from gq_bot.portfolio import calendar_stats, correlation_matrix, drawdown_episodes
from gq_bot.tools import TOOLS, ToolRunner


def pair():
    fx, hm = world()
    hm.tables["price_history"] += [{**b, "instrument_id": "i-voo"} for b in series(500.0, 420, drift=0.0008, wiggle=0.006)]
    return fx, hm


def make():
    fx, hm = pair()
    return GQData(fx, hm, today=lambda: TODAY, workers=1)


def bars(pairs):
    return [{"date": d, "close": c} for d, c in pairs]


def test_drawdown_episodes_find_peak_trough_and_recovery():
    b = bars([("2026-01-01", 100), ("2026-01-02", 80), ("2026-01-03", 90), ("2026-01-04", 101),
              ("2026-01-05", 99), ("2026-01-06", 70), ("2026-01-07", 75)])
    r = drawdown_episodes(b)
    first, second = r["episodes"]
    assert (first["peak_date"], first["trough_date"], first["depth_pct"]) == ("2026-01-04", "2026-01-06", -30.69)
    assert first["recovered"] is False and first["recovery_date"] is None
    assert (second["depth_pct"], second["recovered"], second["recovery_date"]) == (-20.0, True, "2026-01-04")
    assert r["current"]["pct_below_window_high"] == -25.74


def test_small_dips_are_not_episodes():
    assert drawdown_episodes(bars([("2026-01-01", 100), ("2026-01-02", 98), ("2026-01-03", 101)]))["episodes"] == []


def test_calendar_stats_years_and_months():
    b = bars([("2024-12-31", 100), ("2025-06-30", 110), ("2025-12-31", 120), ("2026-01-31", 114), ("2026-02-27", 125)])
    s = calendar_stats(b)
    assert [(y["year"], y["return_pct"]) for y in s["years"]] == [(2024, 0.0), (2025, 20.0), (2026, 4.17)]
    assert s["years"][-1]["partial"] is True and s["years"][1]["partial"] is False
    assert s["months"]["worst"][0]["month"] == "Jan 2026"


def test_correlation_of_identical_and_opposite_series():
    up = [{"date": f"2026-01-{i:02d}", "close": 100 + i + (i % 3)} for i in range(1, 29)]
    up += [{"date": f"2026-02-{i:02d}", "close": 130 + i + (i % 4)} for i in range(1, 20)]
    inv = [{"date": b["date"], "close": 1000 - b["close"] * 3} for b in up]
    r = correlation_matrix({"A": up, "B": up, "C": inv})
    assert r["ok"] and r["matrix"]["A"]["B"] == 1.0 and r["matrix"]["A"]["C"] < -0.9
    assert r["most_correlated"][0]["a"] == "A"
    assert correlation_matrix({"A": up})["reason"] == "not_enough_history"


def test_performance_what_if_and_benchmark():
    d = make()
    r = d.performance("u1", "MSFT", start_date="2025-09-01", amount=10000, monthly_amount=500)
    assert r["ok"] and r["note"].startswith("Price return")
    assert r["lump_sum"]["value_now"] == round(10000 * r["end_close"] / r["start_close"], 2)
    assert r["monthly_investing"]["months"] >= 12 and r["monthly_investing"]["total_invested"] == 500 * r["monthly_investing"]["months"]
    assert r["max_drawdown"]["pct"] <= 0 and r["vs_benchmark"]["benchmark"] == "SPY"
    assert d.performance("u1", "TSLA")["reason"] == "not_on_watchlist"
    assert d.performance("u1", "ABALX")["reason"] == "no_price_data"


def test_calendar_returns_and_drawdowns_tools():
    d = make()
    c = d.calendar_returns("u1", "MSFT", years=3)
    assert c["ok"] and c["years"] and c["benchmark"] == "SPY"
    dd = d.drawdowns("u1", "MSFT", days=1000)
    assert dd["ok"] and dd["current"]["window_high"] > 0
    assert d.drawdowns("u1", "TSLA")["reason"] == "not_on_watchlist"


def test_watchlist_correlation_skips_mutual_funds():
    r = make().watchlist_correlation("u1", ["MSFT", "VOO", "ABALX"])
    assert r["ok"] and set(r["tickers"]) == {"MSFT", "VOO"} and r["excluded_mutual_funds_no_prices"] == ["ABALX"]
    assert make().watchlist_correlation("u1", ["TSLA"])["reason"] == "not_on_watchlist"


def test_portfolio_exposure_combines_funds_and_honours_weights():
    d = make()
    eq = d.portfolio_exposure("u1", ["QQQ", "VOO"])
    assert eq["ok"] and eq["weighting"].startswith("equal")
    aapl = next(h for h in eq["top_combined_holdings"] if h["symbol"] == "AAPL")
    assert aapl["combined_weight_pct"] == round((8.9 + 7.0) / 2, 3) and aapl["funds_holding"] == 2
    assert {x["symbol"] for x in eq["most_repeated"]} == {"MSFT", "AAPL"}
    w = d.portfolio_exposure("u1", ["QQQ", "VOO"], weights={"QQQ": 75, "VOO": 25})
    assert w["weights_pct"] == {"QQQ": 75.0, "VOO": 25.0} and w["weighting"] == "your weights"
    assert next(h for h in w["top_combined_holdings"] if h["symbol"] == "AAPL")["combined_weight_pct"] == round(0.75 * 8.9 + 0.25 * 7.0, 3)
    assert d.portfolio_exposure("u1", ["AAPL"])["reason"] == "no_funds"


def test_new_tools_are_declared_and_dispatch():
    names = {t["function"]["name"] for t in TOOLS}
    assert {"get_performance", "get_calendar_returns", "get_drawdowns", "watchlist_correlation",
            "portfolio_exposure"} <= names
    fx, hm = pair()
    r = ToolRunner(GQData(fx, hm, today=lambda: TODAY, workers=1), "u1", named=["QQQ", "VOO"])
    out = json.loads(r.run("portfolio_exposure", {}))
    assert out["ok"] and set(out["funds_used"]) == {"QQQ", "VOO"}      # named tickers fill the missing list


def test_weight_columns_accept_list_mapping_and_json_string():
    from gq_bot.funds import weight_rows
    want = [{"name": "Tech", "weight_pct": 60.0}, {"name": "Energy", "weight_pct": 40.0}]
    assert weight_rows({"Energy": 40, "Tech": 60}, "sector") == want
    assert weight_rows([{"sector": "Energy", "weight": 40}, {"sector": "Tech", "weight": "60"}], "sector") == want
    assert weight_rows('{"Tech": 60, "Energy": 40}', "sector") == want
    assert weight_rows(None, "sector") == [] and weight_rows("not json", "sector") == []


def test_etf_breakdown_survives_a_mapping_shaped_sector_column():
    b = make().fund_holdings("u1", "QQQ")["breakdown"]
    assert b["by_sector"] == [{"name": "Technology", "weight_pct": 50.0}]


def test_clean_args_coerces_model_output():
    from gq_bot.tools import clean_args
    assert clean_args({"days": "30", "ticker": " qqq ", "tickers": "AAPL", "weights": {"A": "5"}, "amount": "100"}) == \
        {"days": 30, "ticker": "qqq", "tickers": ["AAPL"], "weights": {"A": 5.0}, "amount": 100.0}
    assert clean_args({"days": "x", "weights": [1], "ticker": ["a"]}) == {}
