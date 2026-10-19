from fakes import TODAY, world

from gq_bot.data import GQData


def make(**kw):
    fx, hm = world(**kw)
    return GQData(fx, hm, today=lambda: TODAY), fx, hm


def test_user_lookup_by_chat_id_then_username():
    d, *_ = make()
    assert d.get_user(42)["id"] == "u1"
    assert d.get_user(999, "@LATE")["id"] == "u3"
    assert d.get_user(999) is None


def test_watchlist_is_normalised_and_cached():
    d, fx, _ = make()
    assert d.watchlist("u1") == ["AAPL", "MSFT", "QQQ"]
    n = fx.calls.count("watchlists")
    d.watchlist("u1")
    assert fx.calls.count("watchlists") == n


def test_watchlist_details_resolves_names_and_types():
    d, *_ = make()
    by = {i["ticker"]: i for i in d.watchlist_details("u1")}
    assert by["AAPL"] == {"ticker": "AAPL", "name": "Apple Inc", "type": "stock"}
    assert by["QQQ"]["type"] == "etf"


def test_every_ticker_tool_refuses_a_ticker_off_the_users_watchlist():
    d, *_ = make()
    calls = [d.snapshot, d.price_history, d.fundamentals, d.events, d.explain_move,
             d.etf_profile, d.etf_holdings, d.etf_changes]
    for fn in calls:
        r = fn("u1", "TSLA")
        assert r["ok"] is False and r["reason"] == "not_on_watchlist", fn.__name__
        assert "/add TSLA" in r["message"]
    assert d.compare("u1", ["TSLA"])["results"][0]["reason"] == "not_on_watchlist"


def test_snapshot_reads_the_dashboard_returns():
    d, *_ = make()
    s = d.snapshot("u1", "$aapl")
    assert s["ok"] and s["price_usd"] == 181.0 and s["as_of"] == "2026-10-16"
    assert s["returns_pct"]["ret_1w"] == -9.5 and s["type"] == "stock"


def test_events_are_scoped_windowed_and_filterable():
    d, *_ = make()
    ev = d.events("u1", "AAPL", days=14)["events"]
    assert [e["summary"] for e in ev] == ["Analyst lowered target.", "Apple cut iPhone guidance."]
    assert ev[1]["feature"] == "SEC EDGAR Filings"
    assert [e["impact"] for e in d.events("u1", "AAPL", 14, "HIGH")["events"]] == ["HIGH"]
    assert len(d.events("u1", "AAPL", days=500)["events"]) == 3   # capped at 90 days, still includes the 60-day one


def test_explain_move_joins_price_events_sector_and_market():
    d, *_ = make()
    r = d.explain_move("u1", "AAPL", days=7)
    assert r["ok"] and r["compared_on"] == "ret_1w"
    assert r["price_path"]["change_pct"] == -9.5 and r["price_path"]["sessions"] == 7
    assert any("guidance" in e["summary"] for e in r["events"])
    sec = r["sector_context"]
    assert sec["count"] == 3 and sec["median_pct"] == -1.0 and sec["stock_minus_sector_pct"] == -8.5
    assert r["market_context"] == {"benchmark": "SPY", "return_pct": -0.5, "stock_minus_market_pct": -9.0}


def test_explain_move_for_an_etf_skips_sector_context():
    d, *_ = make()
    r = d.explain_move("u1", "QQQ", days=7)
    assert "sector_context" not in r and r["market_context"]["return_pct"] == -0.5


def test_a_broken_source_degrades_one_piece_not_the_answer():
    d, *_ = make(fx_broken=("alerts",))
    r = d.explain_move("u1", "AAPL", days=7)
    assert r["ok"] and r["events"] == [] and r["price_path"]["enough_data"]
    assert d.events("u1", "AAPL")["reason"] == "source_error"


def test_fundamentals_are_trimmed_to_recent_periods_and_key_lines():
    d, *_ = make()
    f = d.fundamentals("u1", "AAPL")
    assert len(f["quarterly"]) == 4 and len(f["annual"]) == 2
    assert f["quarterly"][0]["income_statement"] == {"revenue": 90e9, "net_income": 20e9}
    assert d.fundamentals("u1", "MSFT")["reason"] == "no_fundamentals"


def test_price_history_caps_and_summarises():
    d, *_ = make()
    h = d.price_history("u1", "AAPL", days=10000)
    assert h["days"] == 400 and h["summary"]["change_pct"] == -9.5 and len(h["bars"]) == 7


def test_etf_tools():
    d, *_ = make()
    p = d.etf_profile("u1", "QQQ")
    assert p["ok"] and p["fund"]["name"] == "Invesco QQQ Trust" and "id" not in p["fund"]
    assert p["expenses"]["expense_ratio"] == 0.2
    assert d.etf_profile("u1", "AAPL")["reason"] == "not_an_etf"
    h = d.etf_holdings("u1", "QQQ", top_n=2)
    assert [x["symbol"] for x in h["top_holdings"]] == ["AAPL", "MSFT"]
    c = d.etf_changes("u1", "QQQ", days=30)
    assert c["holdings_changes"][0]["symbol"] == "PLTR"
    assert [e["event_type"] for e in c["fund_events"]] == ["INDEX_CHANGE"]


def test_compare_dedupes_and_caps():
    d, *_ = make()
    r = d.compare("u1", ["aapl", "AAPL", "msft", "QQQ"])
    assert [x["ticker"] for x in r["results"]] == ["AAPL", "MSFT", "QQQ"]
    many = d.compare("u1", list("ABCDEFG"))
    assert len(many["results"]) == 5 and "note" in many


def test_username_fallback_cannot_match_another_user_through_a_wildcard():
    d, *_ = make(fx={"users": [{"id": "victim", "telegram_chat_id": "7", "telegram_username": "dj12312",
                                "is_active": True},
                               {"id": "real", "telegram_chat_id": None, "telegram_username": "a_b",
                                "is_active": True}]})
    assert d.get_user(999, "d_12312") is None       # `_` must not act as a wildcard
    assert d.get_user(999, "dj%") is None
    assert d.get_user(999, "A_B")["id"] == "real"   # a real underscore still matches exactly
