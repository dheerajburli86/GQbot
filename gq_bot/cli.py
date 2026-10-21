"""Ask the bot from a terminal, against the real databases, without Telegram.

    python -m gq_bot.cli --chat-id 123456789 "Why is AAPL down this week?"
    python -m gq_bot.cli --chat-id 123456789            # interactive
    python -m gq_bot.cli --chat-id 123456789 --trace "..."  # show tool calls
"""

from __future__ import annotations

import argparse
import json
import logging

from .factory import build_bot


def main() -> None:
    p = argparse.ArgumentParser(description="Ask GQ bot a question from the terminal.")
    p.add_argument("--chat-id", required=True, help="Telegram chat id stored in users.telegram_chat_id")
    p.add_argument("--username", help="Telegram username, if the user row has no chat id")
    p.add_argument("--trace", action="store_true", help="print tool calls, tokens and latency")
    p.add_argument("question", nargs="*")
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    bot = build_bot()

    def ask(q: str) -> None:
        ans = bot.answer(a.chat_id, q, a.username)
        print("\n" + ans.text + "\n")
        for i, png in enumerate(ans.images, 1):
            name = f"gqbot_chart_{i}.png"
            with open(name, "wb") as f:
                f.write(png)
            print(f"[chart saved to {name}]")
        if a.trace:
            print(json.dumps({"tools": ans.tools, "tickers": ans.tickers, "error": ans.error,
                              "tokens": [ans.prompt_tokens, ans.completion_tokens],
                              "latency_ms": ans.latency_ms}, indent=2, default=str))

    if a.question:
        ask(" ".join(a.question))
        return
    print("GQ bot (Ctrl-D to quit, /reset to clear memory)")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if q == "/reset":
            bot.reset(a.chat_id)
        elif q:
            ask(q)


if __name__ == "__main__":
    main()
