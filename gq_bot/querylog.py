"""Best-effort log of every question and answer to `gq_bot_queries`.

It exists to measure the bot (which questions fail, which tools run, latency,
token spend). A logging failure must never cost the user their answer, so every
write is swallowed after one warning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


class QueryLog:
    def __init__(self, client=None, enabled: bool = True):
        self.client, self.enabled = client, bool(enabled and client is not None)
        self._warned = False

    def write(self, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.client.table("gq_bot_queries").insert(row).execute()
        except Exception as e:
            if not self._warned:
                log.warning("[querylog] insert failed (%s); is migrations/001 applied?", e)
                self._warned = True
