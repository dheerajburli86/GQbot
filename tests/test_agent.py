from datetime import datetime

from fakes import TODAY, ScriptedLLM, call, say, world

from gq_bot import agent as A
from gq_bot.agent import GQBot
from gq_bot.config import Settings
from gq_bot.data import GQData
from gq_bot.llm import LLMError
from gq_bot.querylog import QueryLog


def bot(replies, steps=6, **kw):
    fx, hm = world(**kw)
    llm = ScriptedLLM(replies)
    s = Settings(max_tool_steps=steps)
    b = GQBot(s, GQData(fx, hm, today=lambda: TODAY), llm, querylog=QueryLog(fx, True),
              now=lambda: datetime(2026, 10, 16, 15, 0))
    return b, llm, fx


def test_tool_then_answer_round_trip_and_query_log():
    b, llm, fx = bot([call("explain_move", ticker="AAPL", days=7), say("AAPL fell 9.50% this week.")])
    ans = b.answer(42, "why is apple down this week?")
    assert ans.text == "AAPL fell 9.50% this week." and ans.error is None
    assert [c["name"] for c in ans.tools] == ["explain_move"] and ans.tickers == ["AAPL"]
    assert ans.prompt_tokens == 300 and ans.completion_tokens == 60
    second = llm.sent[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["function"]["name"] == "explain_move"
    assert second[-1]["role"] == "tool" and '"stock_minus_market_pct":-9.0' in second[-1]["content"]
    logged = fx.tables["gq_bot_queries"][-1]
    assert logged["user_id"] == "u1" and logged["tickers"] == ["AAPL"] and logged["error"] is None


def test_system_prompt_carries_today_and_the_watchlist_only():
    b, llm, _ = bot([say("ok")])
    b.answer(42, "hi")
    system = llm.sent[0]["messages"][0]["content"]
    assert "Friday 16 October 2026" in system
    assert "AAPL: Apple Inc (stock)" in system and "QQQ: Invesco QQQ Trust (etf)" in system
    assert "TSLA" not in system


def test_unknown_user_and_empty_watchlist_never_reach_the_llm():
    b, llm, _ = bot([])
    assert b.answer(999, "hi").text == A.MSG_UNKNOWN_USER
    b2, llm2, _ = bot([], fx={"watchlists": []})
    assert b2.answer(42, "hi").text == A.MSG_EMPTY_WATCHLIST
    assert llm.sent == [] and llm2.sent == []


def test_llm_failure_is_a_friendly_message():
    b, _, fx = bot([LLMError("HTTP 503")])
    ans = b.answer(42, "why?")
    assert ans.text == A.MSG_LLM_DOWN and "503" in ans.error


def test_tool_loop_is_bounded_and_the_last_step_has_no_tools():
    loop = [call("get_snapshot", ticker="AAPL")] * 10
    b, llm, _ = bot(loop + [say("final")], steps=2)
    ans = b.answer(42, "?")
    assert len(llm.sent) == 3 and llm.sent[-1]["tools"] is None
    # The scripted reply at step 3 was still a tool call; with tools withheld it has no text.
    assert ans.text == A.MSG_NO_ANSWER


def test_memory_carries_follow_ups_and_reset_clears_it():
    b, llm, _ = bot([say("AAPL is down."), say("Last month too.")])
    b.answer(42, "how is AAPL?")
    b.answer(42, "and last month?")
    msgs = llm.sent[1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "AAPL is down."
    b.reset(42)
    assert b.memory.get(42) == []


def test_question_is_length_capped():
    b, llm, _ = bot([say("ok")])
    b.answer(42, "x" * 5000)
    assert len(llm.sent[0]["messages"][-1]["content"]) == A.MAX_QUESTION_CHARS


def test_figures_without_a_tool_call_are_nudged_then_grounded():
    b, llm, _ = bot([say("XLE is up 11.23% over 3 months."),
                          call("get_snapshot", ticker="QQQ"), say("QQQ is up, per the data.")])
    a = b.answer(42, "rank the sectors")
    assert a.text == "QQQ is up, per the data." and a.error is None
    assert "called no tool" in llm.sent[1]["messages"][-1]["content"]


def test_figures_never_backed_by_a_tool_are_refused():
    b, llm, _ = bot([say("XLE is up 11.23%."), say("Still 11.23%, trust me.")])
    a = b.answer(42, "rank the sectors")
    assert a.error == "ungrounded_figures" and "11.23" not in a.text


def test_a_plain_reply_without_figures_needs_no_tool():
    b, llm, _ = bot([say("TSLA is not on your watchlist. Add it with /add TSLA.")])
    a = b.answer(42, "how is TSLA")
    assert a.error is None and len(llm.sent) == 1


def test_named_tickers_are_found_in_order_without_matching_inside_words():
    assert A.named_tickers("Rank QQQ, aapl and MSFT; a VOO-like fund", ["MSFT", "AAPL", "QQQ", "A"]) == \
        ["QQQ", "AAPL", "MSFT"]
    assert A.named_tickers("compare these", ["QQQ"]) == []


def test_a_ranking_of_named_tickers_is_scoped_even_if_the_model_forgets():
    b, llm, _ = bot([call("rank_watchlist", metric="return_1w"), say("Ranked.")])
    a = b.answer(42, "rank QQQ, MSFT and AAPL by 1 week return")
    assert a.tools[0]["args"]["tickers"] == ["QQQ", "MSFT", "AAPL"]
    assert '"requested":["QQQ","MSFT","AAPL"]' in llm.sent[1]["messages"][-1]["content"]


def test_dates_are_read_in_every_common_spelling():
    got = A.dates_in("on 2026-09-21, May 6, 2026, 6th May 2026 and Oct. 20 2026; not Feb 30, 2026")
    assert set(got) == {"2026-09-21", "2026-05-06", "2026-10-20"}


def test_an_invented_date_is_nudged_then_removed():
    b, llm, _ = bot([call("get_technicals", ticker="MSFT"),
                     say("- RSI is neutral.\n- Death cross on 2026-09-21."),
                     say("- RSI is neutral.\n- Death cross on September 21, 2026.")])
    a = b.answer(42, "technicals on MSFT")
    assert "2026-09-21" in llm.sent[2]["messages"][-1]["content"]
    assert a.text == "- RSI is neutral." and a.error == "ungrounded_dates_removed"


def test_dates_from_the_tool_or_the_question_pass():
    b, llm, _ = bot([call("get_technicals", ticker="MSFT"), say("MSFT closed on its as-of date. Since March 10, 2025 it rose.")])
    a = b.answer(42, "how has MSFT done since March 10, 2025?")
    assert a.error is None and len(llm.sent) == 2


def test_a_cross_date_the_model_left_out_is_added():
    out = A.add_cross_dates("last golden or death cross on IWM?", "- IWM: golden cross state.",
                            {"IWM": "the 50-day average is ABOVE the 200-day; the last cross was a "
                                    "golden cross on 2025-08-06 (300 sessions ago)."})
    assert out.endswith("- IWM 50/200-day: the 50-day average is ABOVE the 200-day; the last cross "
                        "was a golden cross on 2025-08-06 (300 sessions ago).")
    assert A.add_cross_dates("RSI on IWM?", "x", {"IWM": "s on 2025-08-06"}) == "x"
    assert A.add_cross_dates("cross?", "golden cross on Aug 6, 2025", {"IWM": "on 2025-08-06"}) == \
        "golden cross on Aug 6, 2025"


def test_explaining_a_figure_the_user_gave_is_not_refused():
    b, llm, _ = bot([say("A beta of 1.5 means it moves about 1.5x the market.")])
    a = b.answer(42, "what does a beta of 1.5 mean?")
    assert a.error is None and len(llm.sent) == 1


def test_a_three_year_window_start_is_not_treated_as_invented():
    b, llm, _ = bot([call("get_technicals", ticker="MSFT"),
                     say("- Over October 16, 2023 to October 16, 2026 MSFT rose.")])
    a = b.answer(42, "MSFT over the last 3 years?")
    assert a.error is None and "2023" in a.text


def test_a_macd_crossover_question_gets_no_50_200_line():
    assert A.add_cross_dates("when was the last MACD crossover?", "x", {"IWM": "on 2025-08-06"}) == "x"
    assert A.add_cross_dates("last golden or death cross?", "x", {"IWM": "on 2025-08-06"}) != "x"


def test_delivery_plumbing_never_reaches_the_user():
    assert A.plain_text("QQQ is up 44%.\n\nThe chart image is sent to the user automatically after your text.") == "QQQ is up 44%."


def test_exchange_suffixed_tickers_are_not_turned_into_links():
    out = A.plain_text("Added National Bank of Greece (ETE.AT) and Sun Pharma (SUNPHARMA.BO); US-listed BRK.B too.")
    assert "ETE.\u200bAT" in out and "SUNPHARMA.\u200bBO" in out and "BRK.B" in out
    assert out.replace("\u200b", "") == "Added National Bank of Greece (ETE.AT) and Sun Pharma (SUNPHARMA.BO); US-listed BRK.B too."
