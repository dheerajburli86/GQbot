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
