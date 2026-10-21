"""All runtime settings, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # Telegram. A token of its own: two processes cannot long-poll one bot.
    telegram_token: str = field(default_factory=lambda: os.getenv("GQ_BOT_TELEGRAM_TOKEN", ""))

    # FinXray US database: users, watchlists, alerts, ETF tables.
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))
    # Heatmap dashboard database: instruments, returns_latest, price_history,
    # us_stock_fundamentals. Defaults to the FinXray project when unset.
    heatmap_url: str = field(default_factory=lambda: os.getenv("HEATMAP_SUPABASE_URL", ""))
    heatmap_key: str = field(default_factory=lambda: os.getenv("HEATMAP_SUPABASE_KEY", ""))

    # LLM. "openai" = any OpenAI-compatible chat-completions endpoint
    # (DeepInfra, OpenAI, OpenRouter, ...); "anthropic" = the Claude API.
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower())
    llm_base_url: str = field(default_factory=lambda: os.getenv(
        "LLM_BASE_URL", "https://api.deepinfra.com/v1/openai"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY")
                             or os.getenv("DEEPINFRA_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "")
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2")))
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 1600))
    llm_timeout_s: int = field(default_factory=lambda: _int("LLM_TIMEOUT_S", 90))
    max_tool_steps: int = field(default_factory=lambda: _int("GQ_BOT_MAX_TOOL_STEPS", 8))

    # Conversation + abuse limits.
    history_turns: int = field(default_factory=lambda: _int("GQ_BOT_HISTORY_TURNS", 6))
    history_ttl_s: int = field(default_factory=lambda: _int("GQ_BOT_HISTORY_TTL_S", 1800))
    per_minute_limit: int = field(default_factory=lambda: _int("GQ_BOT_PER_MINUTE", 6))
    per_day_limit: int = field(default_factory=lambda: _int("GQ_BOT_PER_DAY", 150))

    # The bot's OWN Supabase project: holds only what the bot writes (the
    # question log). FinXray and the heatmap are only ever read. Logging is off
    # until this is set, so nothing is ever written to those two projects.
    bot_url: str = field(default_factory=lambda: os.getenv("BOT_SUPABASE_URL", ""))
    bot_key: str = field(default_factory=lambda: os.getenv("BOT_SUPABASE_KEY", ""))
    log_queries: bool = field(default_factory=lambda: _bool("GQ_BOT_LOG_QUERIES", True))

    @property
    def model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return "claude-haiku-5-5" if self.llm_provider == "anthropic" else "google/gemini-2.5-flash"


def get_settings() -> Settings:
    return Settings()
