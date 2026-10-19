"""The question -> tools -> grounded answer loop."""

from __future__ import annotations

import logging
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

        runner = ToolRunner(self.data, uid)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt(self.now().strftime("%A %d %B %Y, %H:%M"), watchlist)},
            *self.memory.get(chat_id),
            {"role": "user", "content": question},
        ]
        ptok = ctok = 0
        text, error = "", None
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
                text = reply.content
                break
        except LLMError as e:
            log.error("[agent] %s", e)
            return Answer(MSG_LLM_DOWN, user_id=uid, tools=runner.calls, tickers=runner.tickers,
                          prompt_tokens=ptok, completion_tokens=ctok, error=str(e))

        if not text:
            text, error = MSG_NO_ANSWER, "empty_answer"
        else:
            self.memory.add(chat_id, question, text)
        return Answer(text, user_id=uid, tools=runner.calls, tickers=runner.tickers,
                      prompt_tokens=ptok, completion_tokens=ctok, error=error)

    def reset(self, chat_id) -> None:
        self.memory.reset(chat_id)
