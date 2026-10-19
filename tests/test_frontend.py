from gq_bot.memory import ConversationMemory, RateLimiter
from gq_bot.prompts import system_prompt
from gq_bot.telegram_app import HELP, split_message


def test_split_message_respects_the_limit_and_keeps_text():
    text = ("para one " * 50 + "\n\n") * 30
    parts = split_message(text, limit=500)
    assert all(len(p) <= 500 for p in parts)
    assert " ".join(p.replace("\n", " ") for p in parts).split() == text.split()
    assert split_message("short") == ["short"]


def test_rate_limiter_per_minute_and_per_day():
    t = [0.0]
    rl = RateLimiter(per_minute=2, per_day=3, clock=lambda: t[0])
    assert rl.check(1)[0] and rl.check(1)[0] and not rl.check(1)[0]
    t[0] = 61
    assert rl.check(1)[0]
    t[0] = 200
    ok, why = rl.check(1)
    assert not ok and "today" in why
    assert rl.check(2)[0]


def test_memory_expires_and_is_bounded():
    t = [0.0]
    m = ConversationMemory(turns=2, ttl_s=100, clock=lambda: t[0])
    for i in range(5):
        m.add(1, f"q{i}", f"a{i}")
    assert [x["content"] for x in m.get(1)] == ["q3", "a3", "q4", "a4"]
    t[0] = 500
    assert m.get(1) == []


def test_prompt_and_help_state_the_product_rules():
    p = system_prompt("Friday", [{"ticker": "AAPL", "name": "Apple", "type": "stock"}])
    for rule in ("Only use numbers", "Only answer about tickers on the watchlist", "No investment advice",
                 "explain_move first", "/add TICKER"):
        assert rule in p
    assert system_prompt("x", []).count("(empty)") == 1
    assert "/watchlist" in HELP and "/add" in HELP


# ── Telegram handler, driven with stub update/context objects ─────────────────
import asyncio
from types import SimpleNamespace

from fakes import TODAY, ScriptedLLM, call, say, world

from gq_bot.agent import GQBot
from gq_bot.config import Settings
from gq_bot.data import GQData
from gq_bot.telegram_app import TelegramFrontEnd


class StubMessage:
    def __init__(self, text):
        self.text, self.replies = text, []

    async def reply_text(self, text, **_):
        self.replies.append(text)


def stub_update(text, chat_id=42, chat_type="private"):
    return SimpleNamespace(message=StubMessage(text),
                           effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
                           effective_user=SimpleNamespace(username="dj"))


class StubBot:
    def __init__(self):
        self.actions = 0

    async def send_chat_action(self, *_):
        self.actions += 1


def frontend(replies, per_minute=5):
    fx, hm = world()
    b = GQBot(Settings(), GQData(fx, hm, today=lambda: TODAY), ScriptedLLM(replies))
    return TelegramFrontEnd(b, RateLimiter(per_minute, 100))


def test_question_handler_answers_and_shows_typing():
    fe = frontend([call("get_snapshot", ticker="AAPL"), say("AAPL is at $181.00 as of 16 Oct 2026.")])
    up, ctx = stub_update("price of apple?"), SimpleNamespace(bot=StubBot())
    asyncio.run(fe.question(up, ctx))
    assert up.message.replies == ["AAPL is at $181.00 as of 16 Oct 2026."] and ctx.bot.actions >= 1


def test_group_chats_are_ignored_and_rate_limit_replies():
    fe = frontend([say("a"), say("b")], per_minute=1)
    grp = stub_update("hi", chat_type="group")
    asyncio.run(fe.question(grp, SimpleNamespace(bot=StubBot())))
    assert grp.message.replies == []
    first, second = stub_update("q1"), stub_update("q2")
    asyncio.run(fe.question(first, SimpleNamespace(bot=StubBot())))
    asyncio.run(fe.question(second, SimpleNamespace(bot=StubBot())))
    assert first.message.replies == ["a"] and "few seconds" in second.message.replies[0]


def test_watchlist_command_lists_names():
    fe = frontend([])
    up = stub_update("/watchlist")
    asyncio.run(fe.watchlist(up, SimpleNamespace(bot=StubBot())))
    assert "AAPL - Apple Inc (stock)" in up.message.replies[0]
