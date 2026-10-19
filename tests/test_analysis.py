from gq_bot import analysis as a


def test_window_column_matches_dashboard_periods():
    assert [a.window_column(d) for d in (1, 5, 7, 8, 30, 90, 180, 365, 999)] == \
        ["ret_1d", "ret_1w", "ret_1w", "ret_1m", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_1y"]


def test_summarise_bars_move_drawdown_and_extremes():
    bars = [{"date": d, "close": c} for d, c in
            [("2026-10-05", 100), ("2026-10-01", 90), ("2026-10-02", 110), ("2026-10-03", 99), ("2026-10-04", 104.5)]]
    s = a.summarise_bars(bars)
    assert s["from"] == "2026-10-01" and s["to"] == "2026-10-05"   # sorted, whatever the input order
    assert s["change_pct"] == 11.11
    assert s["high"] == {"date": "2026-10-02", "close": 110.0}
    assert s["max_drawdown_pct"] == -10.0 and s["drawdown_window"] == ["2026-10-02", "2026-10-03"]
    assert s["biggest_up_days"][0] == {"date": "2026-10-02", "move_pct": 22.22}
    assert s["biggest_down_days"][0]["date"] == "2026-10-03"
    assert s["up_sessions"] + s["down_sessions"] == 4


def test_summarise_bars_needs_two_closes():
    assert a.summarise_bars([{"date": "2026-10-01", "close": 1}])["enough_data"] is False
    assert a.summarise_bars([{"date": "2026-10-01", "close": None}, {"date": "2026-10-02", "close": 2}])["enough_data"] is False


def test_group_stats_ignores_missing():
    assert a.group_stats([1, None, -3, 5]) == {"count": 3, "median_pct": 1.0, "share_positive": 0.67}
    assert a.group_stats([None]) == {"count": 0}
    assert a.relative(-9.5, -1.0) == -8.5 and a.relative(None, 1) is None
    assert a.pct(0, 5) is None
