"""Wire settings -> clients -> bot. The one place real dependencies are built."""

from __future__ import annotations

from .agent import GQBot
from .config import Settings, get_settings
from .data import GQData
from .db import make_clients
from .llm import make_llm
from .memory import ConversationMemory
from .querylog import QueryLog


def build_bot(settings: Settings | None = None) -> GQBot:
    s = settings or get_settings()
    fx, hm = make_clients(s)
    return GQBot(s, GQData(fx, hm), make_llm(s),
                 memory=ConversationMemory(s.history_turns, s.history_ttl_s),
                 querylog=QueryLog(fx, s.log_queries))
