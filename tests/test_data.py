from datetime import timedelta

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
    assert d.watchlist("u1") == ["AAPL", "ABALX", "MSFT", "QQQ", "VOO"]
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
    assert len(d.events("u1", "AAPL", days=500)["events"]) == 3   # capped at 365 days, includes the 60-day one
    assert [e["type"] for e in d.events("u1", "AAPL", 30, category="analyst")["events"]] == ["RATING"]
    assert [e["type"] for e in d.events("u1", "AAPL", 30, category="sec_filing")["events"]] == ["8-K"]
    old = d.events("u1", "AAPL", start_date=(TODAY - timedelta(days=70)).isoformat(),
                   end_date=(TODAY - timedelta(days=50)).isoformat())["events"]
    assert [e["summary"] for e in old] == ["Old news."]


def test_explain_move_joins_price_events_sector_and_market():
    d, *_ = make()
    r = d.explain_move("u1", "AAPL", days=7)
    assert r["ok"] and r["window"]["to"] == TODAY.isoformat()
    assert r["price_path"]["change_pct"] == -9.5 and r["price_path"]["sessions"] == 7
    assert any("guidance" in e["summary"] for e in r["events"])
    sec = r["sector_context"]
    assert sec["count"] == 7 and sec["median_pct"] == 0.5 and sec["stock_minus_sector_pct"] == -10.0
    assert r["market_context"] == {"benchmark": "SPY", "return_pct": -0.5, "stock_minus_market_pct": -9.0}
    worst = r["key_sessions"]["biggest_down_days"][0]
    assert worst["date"] == (TODAY - timedelta(days=4)).isoformat()


def test_explain_move_for_a_named_period_uses_those_dates():
    d, *_ = make()
    r = d.explain_move("u1", "AAPL", start_date=(TODAY - timedelta(days=4)).isoformat(),
                       end_date=(TODAY - timedelta(days=1)).isoformat())
    assert r["price_path"]["from"] == (TODAY - timedelta(days=4)).isoformat()
    assert r["price_path"]["to"] == (TODAY - timedelta(days=1)).isoformat()
    assert "sector_context" not in r   # dashboard columns are lookbacks, not arbitrary periods


def test_explain_move_for_an_etf_skips_sector_context():
    d, *_ = make()
    r = d.explain_move("u1", "QQQ", days=7)
    assert "sector_context" not in r and r["market_context"]["return_pct"] == -0.5


def test_a_broken_source_degrades_one_piece_not_the_answer():
    d, *_ = make(fx_broken=("alerts",))
    r = d.explain_move("u1", "AAPL", days=7)
    assert r["ok"] and r["events"] == [] and r["price_path"]["enough_data"]
    assert d.events("u1", "AAPL")["reason"] == "source_error"


def test_price_history_takes_a_date_range_and_samples_long_windows():
    d, *_ = make()
    h = d.price_history("u1", "AAPL", days=10)
    assert h["summary"]["change_pct"] == -9.5 and len(h["bars"]) == 7
    long = d.price_history("u1", "MSFT", days=3650)
    assert long["summary"]["sessions"] == 420 and len(long["sampled_closes"]) <= 25
    assert long["sampled_closes"][-1]["date"] == TODAY.isoformat()


def test_etf_tools():
    d, *_ = make()
    p = d.etf_profile("u1", "QQQ")
    assert p["ok"] and p["fund"]["name"] == "Invesco QQQ Trust" and "id" not in p["fund"]
    assert p["expenses"]["expense_ratio"] == 0.2
    assert d.etf_profile("u1", "AAPL")["reason"] == "not_an_etf"
    assert d.etf_holdings("u1", "QQQ")["top_n_combined_weight_pct"] == 25.5
    h = d.etf_holdings("u1", "QQQ", top_n=2)
    assert [x["symbol"] for x in h["top_holdings"]] == ["AAPL", "MSFT"]
    c = d.etf_changes("u1", "QQQ", days=30)
    assert c["holdings_changes"]["net_added"][0]["symbol"] == "PLTR"
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


def test_holding_changes_are_netted_and_cash_is_ignored():
    from gq_bot.data import net_holding_changes
    rows = [{"action": "ADDED", "security_symbol": "X", "security_name": "Xco", "detected_at": "2026-10-01"},
            {"action": "REMOVED", "security_symbol": "X", "security_name": "Xco", "detected_at": "2026-10-05"},
            {"action": "ADDED", "security_symbol": "Y", "security_name": "Yco", "weight_new": 0.5, "detected_at": "2026-10-02"},
            {"action": "REMOVED", "security_symbol": "Z", "security_name": "Zco", "detected_at": "2026-10-03"},
            {"action": "REMOVED", "security_symbol": "NLOK", "security_name": "Gen Digital Inc", "detected_at": "2026-10-06"},
            {"action": "ADDED", "security_symbol": "GEN", "security_name": "Gen Digital Inc", "detected_at": "2026-10-07"},
            {"action": "ADDED", "security_symbol": "", "security_name": "-USD CASH-", "detected_at": "2026-10-04"},
            {"action": "ADDED", "security_symbol": "", "security_name": "GOLDMAN FS TRSY OBLIG INST 468", "detected_at": "2026-10-04"},
            {"action": "REMOVED", "security_symbol": "", "security_name": "ICE Brent Crude Oil Future 10/30/2026", "detected_at": "2026-10-04"},
            {"action": "ADDED", "security_symbol": "", "security_name": "CME E-mini Russell 1000 Value Index Futures 12/18/2026", "detected_at": "2026-10-04"},
            {"action": "ADDED", "security_symbol": "", "security_name": "MKTLIQ 12/31/2049", "detected_at": "2026-10-04"},
            {"action": "REMOVED", "security_symbol": "", "security_name": "United States Treasury Bill 11/20/2026", "detected_at": "2026-10-04"},
            {"action": "ADDED", "security_symbol": "BILL", "security_name": "BILL Holdings Inc", "detected_at": "2026-10-04"}]
    n = net_holding_changes(rows)
    # BILL Holdings is a company, not a Treasury bill
    assert [e["symbol"] for e in n["net_added"]] == ["Y", "BILL"] and [e["symbol"] for e in n["net_removed"]] == ["Z"]
    assert n["churned_count"] == 2 and n["cash_and_derivative_lines_ignored"] == 6


def test_watchlist_etf_changes_covers_every_etf():
    d, *_ = make()
    r = d.watchlist_etf_changes("u1", 90)
    assert r["ok"] and r["etfs_checked"] >= 2
    assert r["changed"][0]["etf"] == "QQQ" and r["changed"][0]["net_added_count"] == 1
    assert "VOO" in r["no_net_change"]
