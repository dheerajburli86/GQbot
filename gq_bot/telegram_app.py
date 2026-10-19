"""Telegram front end.

Run as ONE replica: Telegram hands each update to whichever long-poller asks
first, so a second replica silently answers half the questions and loses the
conversation memory for the rest.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from telegram import LinkPreviewOptions, Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .agent import GQBot
from .config import get_settings
from .factory import build_bot
from .memory import RateLimiter

log = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4000  # Telegram's hard cap is 4096; leave headroom
NO_PREVIEW = LinkPreviewOptions(is_disabled=True)

HELP = (
    "I'm GQ bot. Ask me anything about the US stocks and ETFs on your GQ FinXray watchlist, "
    "and I'll combine FinXray's news, filings and alerts with the heatmap dashboard's price data.\n\n"
    "Try:\n"
    "- Why is NVDA down this week?\n"
    "- What happened to AAPL this month?\n"
    "- How has SPY done YTD vs QQQ?\n"
    "- Any recent filings or insider trades for MSFT?\n"
    "- Top holdings of QQQ, and did anything change?\n\n"
    "Commands:\n"
    "/watchlist - what I can answer about\n"
    "/reset - start a fresh conversation\n"
    "/help - this message\n\n"
    "Add or remove tickers in the GQ FinXray bot with /add and /remove. "
    "I describe what the data shows; I don't give buy or sell advice."
)


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> List[str]:
    """Split on paragraph, then line, then hard boundaries, never mid-word if avoidable."""
    text = (text or "").strip()
    parts: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts or [""]


class TelegramFrontEnd:
    def __init__(self, bot: GQBot, limiter: RateLimiter):
        self.bot, self.limiter = bot, limiter

    @staticmethod
    def _private(update: Update) -> bool:
        return bool(update.effective_chat and update.effective_chat.type == ChatType.PRIVATE)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._private(update):
            await update.message.reply_text(HELP)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._private(update):
            await update.message.reply_text(HELP)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self._private(update):
            self.bot.reset(update.effective_chat.id)
            await update.message.reply_text("Fresh start. What would you like to know?")

    async def watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._private(update):
            return
        chat = update.effective_chat.id
        username = update.effective_user.username if update.effective_user else None

        def lookup():
            user = self.bot.data.get_user(chat, username)
            return (user, self.bot.data.watchlist_details(user["id"])) if user else (None, [])
        try:
            user, items = await asyncio.to_thread(lookup)
        except Exception:
            log.exception("[tg] watchlist lookup failed")
            await update.message.reply_text("I couldn't reach the GQuants database just now.")
            return
        if not user:
            await update.message.reply_text("I couldn't find your GQ FinXray account. Send /start "
                                            "to the GQ FinXray bot first.")
            return
        if not items:
            await update.message.reply_text("Your watchlist is empty. Add tickers in the GQ FinXray "
                                            "bot with /add TICKER.")
            return
        lines = [f"{i['ticker']} - {i.get('name') or 'name unknown'} ({i.get('type')})" for i in items]
        for part in split_message("You can ask me about:\n" + "\n".join(lines)):
            await update.message.reply_text(part)

    async def question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._private(update) or not update.message or not update.message.text:
            return
        chat = update.effective_chat.id
        ok, why = self.limiter.check(chat)
        if not ok:
            await update.message.reply_text(why)
            return
        username = update.effective_user.username if update.effective_user else None

        # The answer can take several LLM round trips; keep "typing..." alive
        # (Telegram drops it after ~5s) and keep the event loop free.
        stop = asyncio.Event()

        async def send_typing():
            try:
                await context.bot.send_chat_action(chat, ChatAction.TYPING)
            except Exception:
                pass

        async def keep_typing():
            while True:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=4.5)
                    return
                except asyncio.TimeoutError:
                    await send_typing()

        # First indicator inline: a task's first step is not guaranteed to run
        # before a quick answer comes back.
        await send_typing()
        pulse = asyncio.create_task(keep_typing())
        try:
            ans = await asyncio.to_thread(self.bot.answer, chat, update.message.text, username)
        finally:
            stop.set()
            await pulse
        log.info("[tg] chat=%s tools=%s tickers=%s %dms err=%s", chat,
                 [c["name"] for c in ans.tools], ans.tickers, ans.latency_ms, ans.error)
        for part in split_message(ans.text):
            await update.message.reply_text(part, link_preview_options=NO_PREVIEW)

    async def on_error(self, update, context: ContextTypes.DEFAULT_TYPE):
        log.error("[tg] unhandled error: %s", context.error, exc_info=context.error)


def build_application(bot: GQBot, token: str, limiter: RateLimiter) -> Application:
    fe = TelegramFrontEnd(bot, limiter)
    app = Application.builder().token(token).concurrent_updates(True).build()
    app.add_handler(CommandHandler("start", fe.start))
    app.add_handler(CommandHandler("help", fe.help))
    app.add_handler(CommandHandler("reset", fe.reset))
    app.add_handler(CommandHandler("watchlist", fe.watchlist))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fe.question))
    app.add_error_handler(fe.on_error)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    for noisy in ("httpx", "httpcore", "telegram", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    s = get_settings()
    if not s.telegram_token:
        raise SystemExit("GQ_BOT_TELEGRAM_TOKEN is not set")
    bot = build_bot(s)
    app = build_application(bot, s.telegram_token, RateLimiter(s.per_minute_limit, s.per_day_limit))
    log.info("[tg] GQ bot starting (provider=%s model=%s)", s.llm_provider, s.model)
    # run_polling owns the event loop; do not wrap it in asyncio.run().
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
