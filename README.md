# GQ bot

A Telegram equity assistant for GQuants users. Ask it about the US stocks and
ETFs on your GQ FinXray watchlist. It combines two GQuants sources and explains
how they connect:

- **GQ FinXray US** says *what happened*: SEC filings, news, earnings, analyst
  ratings, insider trades and ETF fund actions, each with a summary, impact and
  date.
- **The heatmap dashboard (US tile)** says *what the price did*: returns from
  1D to ALL, daily closes, sector and fundamentals.

> "Why is AAPL down this week?" → AAPL fell 9.50% (as of 16 Oct) while its
> sector median was -1.00% and SPY -0.50%, so the move is stock-specific. It
> coincided with the 13 Oct 8-K cutting iPhone guidance (HIGH impact).

## Scope (v1)

| Rule | Where it is enforced |
|---|---|
| Only tickers on the asking user's own FinXray watchlist | `data.py` `_scoped()`, called by every ticker lookup. Not just a prompt rule |
| US stocks and ETFs only | `instruments.universe in (us_stocks, us_etfs)` + prompt |
| Answers use only data returned by tools | prompt rule 1; the model gets no other price or news source |
| No buy/sell advice or price predictions | prompt rule 6 |
| Private chats only; per-user rate limits | `telegram_app.py`, `memory.py` |

One deliberate exception: `explain_move` adds the stock's **sector median** and
**SPY** over the same window, so the bot can say "the whole market fell too".
Those figures are aggregate context. No tool answers a standalone question
about a ticker that isn't on the user's watchlist.

Watchlists are managed in the existing GQ FinXray bot (`/add`, `/remove`). GQ
bot reads the same `watchlists` table.

## How it works

```
Telegram ──> telegram_app.py ──> agent.py ──(tool calls)──> tools.py ──> data.py ──> Supabase
              typing, limits      LLM loop                  user-bound    scope check   FinXray + heatmap
```

1. The chat id is matched to a FinXray `users` row. A private chat's id equals
   the Telegram user id, which is the same across bots, so the chat id the
   FinXray bot stored works here. Username is the fallback.
2. The system prompt carries today's date (ET) and the user's watchlist with
   names, so "Apple" resolves to AAPL.
3. The LLM calls tools (up to `GQ_BOT_MAX_TOOL_STEPS`, default 6). On the last
   step tools are withheld, which forces it to answer.
4. Only questions and final answers go into short-term memory (6 turns, 30
   minutes), so follow-ups like "and last month?" work.
5. Every question is logged to `gq_bot_queries` in the bot's own Supabase project. This is best-effort and never
   blocks an answer.

### Tools

| Tool | Data |
|---|---|
| `explain_move` | price path + drawdown, FinXray events in/before the window, sector median, SPY |
| `get_snapshot` | price and 1D to ALL returns, as on the dashboard |
| `get_price_history` | daily closes + summary (high, low, max drawdown, biggest days) |
| `get_events` | FinXray alerts, filterable by impact |
| `get_fundamentals` | last 4 quarters + 2 fiscal years (stocks) |
| `compare_tickers` | up to 5 watchlisted tickers side by side |
| `get_etf_profile` | fund facts, expense ratio, company data (ETFs) |
| `get_etf_holdings` | top holdings by weight (ETFs) |
| `get_etf_changes` | holdings added/removed, SEC fund actions, alerts (ETFs) |

### Tables read

| Database | Tables |
|---|---|
| FinXray US | `users`, `watchlists`, `alerts`, `etf_master`, `etf_company_data`, `etf_expenses`, `etf_holdings`, `etf_holdings_changes`, `etf_fund_events` |
| Heatmap | `instruments`, `returns_latest`, `price_history`, `us_stock_fundamentals` |

These column names were taken from the two repos' code and migrations and have
**not yet been checked against the live databases**. Each read is wrapped, so a
mismatch shows up as a `source_error` for that piece (and a warning in the logs)
rather than a crash. Run the CLI with `--trace` after deploying to confirm.

## Setup

1. **Create a new Telegram bot** with @BotFather. Don't reuse the FinXray alert
   bot's token, because two processes can't long-poll one bot.
2. **Configure**: `cp .env.example .env` and fill in the token, both Supabase
   projects (leave the `HEATMAP_*` pair blank if it's the same project) and the
   LLM. The default is DeepInfra `google/gemini-2.5-flash` via the
   OpenAI-compatible API, the same account FinXray uses. Set
   `LLM_PROVIDER=anthropic` for Claude (default model `claude-haiku-5-5`).
3. **Create the bot's own Supabase project** (new and empty) and run
   `migrations/001_gq_bot_query_log.sql` in it. Put its URL and service_role key
   in `BOT_SUPABASE_URL` / `BOT_SUPABASE_KEY`. FinXray and the heatmap projects
   are only read, never written. Without these two variables the bot works but
   does not log.
4. **Install and test**:
   ```bash
   pip install -r requirements.txt pytest
   python -m pytest -q
   ```
5. **Try it without Telegram** against the real data:
   ```bash
   python -m gq_bot.cli --chat-id <your telegram chat id> --trace "Why is AAPL down this week?"
   ```
6. **Run the bot**: `python -m gq_bot`

### Deploy (Railway)

`railway.toml` and `Procfile` start `python -m gq_bot`. Keep it at **one
replica**: Telegram gives each update to whichever poller asks first, so a
second replica would answer half the messages and split conversation memory.

## Tests

`tests/` runs fully offline with in-memory fakes for Supabase, the LLM and
Telegram. It covers the watchlist guarantee on every tool, user binding (a model
can't pass another user's id), the tool loop bound, provider message
conversion, rate limits, memory expiry and message splitting.

## Layout

```
gq_bot/
  config.py        env settings
  db.py            Supabase clients (FinXray + heatmap)
  data.py          read-only, watchlist-scoped lookups
  analysis.py      pure price-path maths
  tools.py         LLM tool schemas + user-bound dispatcher
  llm.py           OpenAI-compatible and Anthropic clients, one interface
  prompts.py       system prompt
  agent.py         question -> tools -> answer loop
  memory.py        per-chat memory + rate limiter
  querylog.py      best-effort question log
  telegram_app.py  Telegram front end
  cli.py           terminal client
migrations/        gq_bot_queries table
tests/             offline test suite
```
