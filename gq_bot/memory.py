"""Short per-chat memory and rate limits, in process.

Only the user's questions and the bot's final answers are remembered, not tool
results: follow-ups like "and last month?" need the topic, not 9 KB of JSON,
and every remembered byte is re-sent on every turn. The bot runs as a single
replica (Telegram long-polling allows only one), so in-process state is enough.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Tuple


class ConversationMemory:
    def __init__(self, turns: int = 6, ttl_s: int = 1800, clock: Callable[[], float] = time.monotonic):
        self.turns, self.ttl, self.clock = turns, ttl_s, clock
        self._store: Dict[str, Tuple[float, Deque[Dict[str, str]]]] = {}
        self._lock = threading.Lock()

    def get(self, chat_id) -> List[Dict[str, str]]:
        with self._lock:
            hit = self._store.get(str(chat_id))
            if not hit or self.clock() - hit[0] > self.ttl:
                self._store.pop(str(chat_id), None)
                return []
            return list(hit[1])

    def add(self, chat_id, question: str, answer: str) -> None:
        with self._lock:
            key = str(chat_id)
            hit = self._store.get(key)
            q = hit[1] if hit and self.clock() - hit[0] <= self.ttl else deque(maxlen=self.turns * 2)
            q.append({"role": "user", "content": question})
            q.append({"role": "assistant", "content": answer})
            self._store[key] = (self.clock(), q)

    def reset(self, chat_id) -> None:
        with self._lock:
            self._store.pop(str(chat_id), None)


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, clock: Callable[[], float] = time.time):
        self.per_minute, self.per_day, self.clock = per_minute, per_day, clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key) -> Tuple[bool, str]:
        now = self.clock()
        with self._lock:
            hits = self._hits[str(key)]
            while hits and now - hits[0] > 86400:
                hits.popleft()
            if len(hits) >= self.per_day:
                return False, "You've reached today's question limit. Please try again tomorrow."
            if sum(1 for t in hits if now - t < 60) >= self.per_minute:
                return False, "You're asking quickly. Give me a few seconds and try again."
            hits.append(now)
            return True, ""
