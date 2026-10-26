import json

from fakes import TODAY, world

from gq_bot.data import GQData
from gq_bot.tools import MAX_RESULT_CHARS, TOOLS, ToolRunner, _compact


def runner(uid="u1"):
    fx, hm = world()
    return ToolRunner(GQData(fx, hm, today=lambda: TODAY), uid)


def test_every_schema_has_a_handler():
    r = runner()
    # hidden aliases kept for old conversation memory
    aliases = {"get_fundamentals", "get_etf_profile", "get_etf_holdings", "get_etf_overlap",
               "get_etf_exposure", "get_etf_changes", "watchlist_etf_changes"}
    assert {t["function"]["name"] for t in TOOLS} == set(r._table) - aliases


def test_user_is_bound_by_the_runner_not_the_model():
    # A model that tries to pass user_id gets it ignored: u1 cannot see u9's TSLA.
    out = json.loads(runner().run("get_snapshot", json.dumps({"ticker": "TSLA", "user_id": "u9"})))
    assert out["reason"] == "not_on_watchlist"


def test_bad_and_unknown_calls_return_errors_not_exceptions():
    r = runner()
    assert json.loads(r.run("nope", "{}"))["reason"] == "unknown_tool"
    assert json.loads(r.run("get_snapshot", "not json"))["reason"] == "bad_arguments"
    assert json.loads(r.run("get_events", {"ticker": "AAPL", "days": None}))["ok"] is True
    assert [c["name"] for c in r.calls] == ["nope", "get_snapshot", "get_events"]


def test_tickers_are_collected_from_calls():
    r = runner()
    r.run("get_snapshot", {"ticker": "aapl"})
    r.run("compare_tickers", {"tickers": ["MSFT", "AAPL"]})
    assert r.tickers == ["AAPL", "MSFT"]


def test_large_results_are_truncated():
    assert len(_compact({"x": "y" * (MAX_RESULT_CHARS * 2)})) < MAX_RESULT_CHARS + 50
