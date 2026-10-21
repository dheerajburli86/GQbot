"""The question -> tools -> grounded answer loop."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from .config import Settings
from .data import GQData
from .llm import LLMError
from .memory import ConversationMemory
from .prompts import system_prompt
from .querylog import QueryLog
from .tools import TOOLS, ToolRunner

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MAX_QUESTION_CHARS = 1000

MSG_UNKNOWN_USER = ("I couldn't find your GQ FinXray account. Open the GQ FinXray bot, send /start "
                    "to link your Telegram, then come back and ask me again.")
MSG_EMPTY_WATCHLIST = ("Your watchlist is empty, so there's nothing I can answer about yet. Add US "
                       "stocks or ETFs in the GQ FinXray bot with /add TICKER, then ask me "
                       "anything about them.")
MSG_LLM_DOWN = "I couldn't reach my language model just now. Please try again in a minute."
MSG_DATA_DOWN = "I couldn't reach the GQuants database just now. Please try again in a minute."
MSG_NO_ANSWER = "I couldn't put an answer together for that. Could you rephrase it?"
MSG_UNGROUNDED = ("I couldn't fetch the data to answer that properly just now, and I won't give you "
                  "numbers I haven't looked up. Please ask again.")
NUDGE_NO_TOOLS = ("You answered with figures but called no tool in this turn. Earlier answers and your "
                  "own knowledge are not a source. Call the tools that fetch these figures now, then "
                  "answer only from their results.")
_FIGURE = re.compile(r"\d+(?:\.\d+)?\s?%|\$\s?\d|\b\d+\.\d+\b|\b\d+(?:\.\d+)?x\b")


_MON = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MNAME = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_ISO = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)")
_MDY = re.compile(_MNAME + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d\d)\b", re.I)
_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MNAME + r",?\s+(20\d\d)\b", re.I)
NUDGE_DATES = ("These dates in your answer appear in no tool result from this turn: {dates}. Every "
               "date must come from a tool result. Remove or correct them and answer again.")


def dates_in(text: str) -> Dict[str, List[str]]:
    """Calendar dates written in the text, as ISO -> the spellings used."""
    found: Dict[str, List[str]] = {}

    def add(y, m, d, raw):
        try:
            iso = datetime(int(y), int(m), int(d)).date().isoformat()
        except ValueError:
            return
        found.setdefault(iso, []).append(raw)
    for m in _ISO.finditer(text or ""):
        add(m.group(1), m.group(2), m.group(3), m.group(0))
    for m in _MDY.finditer(text or ""):
        add(m.group(3), _MON[m.group(1)[:3].lower()], m.group(2), m.group(0))
    for m in _DMY.finditer(text or ""):
        add(m.group(3), _MON[m.group(2)[:3].lower()], m.group(1), m.group(0))
    return found


def ungrounded_dates(text: str, sources: List[str]) -> Dict[str, List[str]]:
    known = set()
    for src in sources:
        known |= set(dates_in(src))
    return {d: raw for d, raw in dates_in(text).items() if d not in known}


def drop_lines_with(text: str, spellings: List[str]) -> str:
    keep = [ln for ln in text.splitlines() if not any(s in ln for s in spellings)]
    return "\n".join(keep).strip()


_CROSS_Q = re.compile(r"golden|death|\bcross", re.I)


def add_cross_dates(question: str, text: str, summaries: Dict[str, str]) -> str:
    """When crosses were asked about, make sure each dated 50/200 cross reaches the user."""
    if not summaries or not _CROSS_Q.search(question or ""):
        return text
    have = set(dates_in(text))
    extra = [f"- {t} 50/200-day: {s}" for t, s in summaries.items()
             if set(dates_in(s)) - have or (not dates_in(s) and s not in text)]
    return text + ("\n\n" + "\n".join(extra) if extra else "")


def has_figures(text: str) -> bool:
    """Percentages, money, decimals or multiples: the things that must come from a tool."""
    return bool(_FIGURE.search(text or ""))


@dataclass
class Answer:
    text: str
    user_id: Optional[str] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    error: Optional[str] = None
    images: List[bytes] = field(default_factory=list)


_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)
_BULLET = re.compile(r"^(\s*)[*•]\s+", re.M)
_HEADING = re.compile(r"^\s*#{1,6}\s*", re.M)


def plain_text(text: str) -> str:
    """Telegram is sent without a parse mode, so markdown would show as literal symbols."""
    text = _BOLD.sub(lambda m: m.group(1) or m.group(2), text)
    text = _BULLET.sub(lambda m: m.group(1) + "- ", text)
    text = _HEADING.sub("", text)
    return text.strip()


def named_tickers(question: str, tickers: List[str]) -> List[str]:
    """Watchlist tickers written out in the question, in the order asked."""
    q = question or ""
    hits = []
    for t in tickers:
        # short tickers ("A", "IT", "ON") only count when typed in capitals
        flags = re.I if len(t) >= 3 else 0
        m = re.search(r"(?<![A-Za-z0-9.$])\$?" + re.escape(t.upper()) + r"(?![A-Za-z0-9])", q, flags)
        if m:
            hits.append((m.start(), t.upper()))
    return [t for _, t in sorted(hits)]


def _assistant_msg(reply) -> Dict[str, Any]:
    return {"role": "assistant", "content": reply.content or None,
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments}}
                           for c in reply.tool_calls]}


class GQBot:
    def __init__(self, settings: Settings, data: GQData, llm, memory: Optional[ConversationMemory] = None,
                 querylog: Optional[QueryLog] = None, now=lambda: datetime.now(ET)):
        self.s, self.data, self.llm = settings, data, llm
        self.memory = memory or ConversationMemory(settings.history_turns, settings.history_ttl_s)
        self.querylog = querylog or QueryLog(None, False)
        self.now = now

    def answer(self, chat_id, question: str, username: Optional[str] = None) -> Answer:
        started = time.monotonic()
        question = (question or "").strip()[:MAX_QUESTION_CHARS]
        ans = self._answer(chat_id, question, username)
        ans.latency_ms = int((time.monotonic() - started) * 1000)
        self.querylog.write({
            "user_id": ans.user_id, "chat_id": str(chat_id), "question": question,
            "answer": ans.text, "tools_used": ans.tools, "tickers": ans.tickers,
            "prompt_tokens": ans.prompt_tokens, "completion_tokens": ans.completion_tokens,
            "latency_ms": ans.latency_ms, "model": self.s.model, "error": ans.error})
        return ans

    def _answer(self, chat_id, question: str, username: Optional[str]) -> Answer:
        try:
            user = self.data.get_user(chat_id, username)
        except Exception as e:
            log.exception("[agent] user lookup failed")
            return Answer(MSG_DATA_DOWN, error=f"user_lookup: {e}")
        if not user:
            return Answer(MSG_UNKNOWN_USER, error="unknown_user")
        uid = user["id"]
        try:
            watchlist = self.data.watchlist_details(uid)
        except Exception as e:
            log.exception("[agent] watchlist lookup failed")
            return Answer(MSG_DATA_DOWN, user_id=uid, error=f"watchlist: {e}")
        if not watchlist:
            return Answer(MSG_EMPTY_WATCHLIST, user_id=uid, error="empty_watchlist")

        runner = ToolRunner(self.data, uid, named_tickers(question, [w["ticker"] for w in watchlist]))
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt(self.now().strftime("%A %d %B %Y, %H:%M"), watchlist)},
            *self.memory.get(chat_id),
            {"role": "user", "content": question},
        ]
        ptok = ctok = 0
        text, error = "", None
        nudged = dates_nudged = False
        today = self.now()
        allowed = [question, today.date().isoformat()]
        try:
            for step in range(self.s.max_tool_steps + 1):
                # The last step withholds tools so the model has to answer.
                tools = TOOLS if step < self.s.max_tool_steps else None
                reply = self.llm.chat(messages, tools)
                ptok += reply.prompt_tokens
                ctok += reply.completion_tokens
                if reply.tool_calls and tools:
                    messages.append(_assistant_msg(reply))
                    for call in reply.tool_calls:
                        messages.append({"role": "tool", "tool_call_id": call.id,
                                         "content": runner.run(call.name, call.arguments)})
                    continue
                text = reply.content or ""
                # A data answer with no tool call this turn is from memory or invented: push back once.
                if tools and not runner.calls and not nudged and has_figures(text):
                    nudged = True
                    log.warning("[agent] figures without a tool call; nudging")
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": NUDGE_NO_TOOLS})
                    continue
                bad = ungrounded_dates(text, runner.outputs + allowed) if runner.calls else {}
                if bad and tools and not dates_nudged:
                    dates_nudged = True
                    log.warning("[agent] ungrounded dates %s; nudging", sorted(bad))
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": NUDGE_DATES.format(dates=", ".join(sorted(bad)))})
                    continue
                break
        except LLMError as e:
            log.error("[agent] %s", e)
            return Answer(MSG_LLM_DOWN, user_id=uid, tools=runner.calls, tickers=runner.tickers,
                          prompt_tokens=ptok, completion_tokens=ctok, error=str(e))

        text = plain_text(text or "")
        bad = ungrounded_dates(text, runner.outputs + allowed) if runner.calls else {}
        if bad:
            log.error("[agent] removing lines with ungrounded dates %s", sorted(bad))
            text = drop_lines_with(text, [r for raws in bad.values() for r in raws])
            error = "ungrounded_dates_removed"
        if text and runner.ma_summaries:
            text = add_cross_dates(question, text, runner.ma_summaries)
        if text and not runner.calls and has_figures(text):
            log.error("[agent] refused an ungrounded answer")
            text, error = MSG_UNGROUNDED, "ungrounded_figures"
            return Answer(text, user_id=uid, tools=runner.calls, tickers=runner.tickers,
                          prompt_tokens=ptok, completion_tokens=ctok, error=error)
        if not text:
            text, error = MSG_NO_ANSWER, "empty_answer"
        else:
            self.memory.add(chat_id, question, text)
        return Answer(text, user_id=uid, tools=runner.calls, tickers=runner.tickers,
                      prompt_tokens=ptok, completion_tokens=ctok, error=error, images=runner.images)

    def reset(self, chat_id) -> None:
        self.memory.reset(chat_id)
