-- GQ bot: one row per question. Apply in the FinXray US Supabase project.
-- Optional: with GQ_BOT_LOG_QUERIES=false the bot never touches this table.

CREATE TABLE IF NOT EXISTS public.gq_bot_queries (
    id                 BIGSERIAL PRIMARY KEY,
    user_id            UUID,
    chat_id            TEXT NOT NULL,
    question           TEXT NOT NULL,
    answer             TEXT,
    tools_used         JSONB DEFAULT '[]'::jsonb,
    tickers            TEXT[] DEFAULT '{}',
    prompt_tokens      INTEGER DEFAULT 0,
    completion_tokens  INTEGER DEFAULT 0,
    latency_ms         INTEGER,
    model              TEXT,
    error              TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS gq_bot_queries_created_idx ON public.gq_bot_queries (created_at DESC);
CREATE INDEX IF NOT EXISTS gq_bot_queries_user_idx    ON public.gq_bot_queries (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS gq_bot_queries_error_idx   ON public.gq_bot_queries (error) WHERE error IS NOT NULL;

-- If users.id is not a UUID in your project, change user_id's type to match.

-- Read-only helpers the bot relies on already exist in FinXray (alerts_ticker_filing_type_idx,
-- watchlists_ticker_idx). This one makes the per-ticker, newest-first alert read cheap:
CREATE INDEX IF NOT EXISTS alerts_ticker_created_idx ON public.alerts (ticker, created_at DESC);
