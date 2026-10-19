-- GQ bot: question log. Run this in the BOT's own Supabase project (the new
-- one), NOT in FinXray or the heatmap project: the bot only reads those.
-- Optional: without BOT_SUPABASE_URL/KEY the bot simply does not log.

CREATE TABLE IF NOT EXISTS public.gq_bot_queries (
    id                 BIGSERIAL PRIMARY KEY,
    user_id            TEXT,                 -- FinXray users.id, kept as text
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

-- The bot connects with the service_role key, which bypasses row level
-- security. Keep RLS ON so the public (anon) key can never read these
-- questions.
ALTER TABLE public.gq_bot_queries ENABLE ROW LEVEL SECURITY;
