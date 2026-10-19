"""Supabase clients for the two source databases."""

from __future__ import annotations

from .config import Settings


def make_clients(s: Settings):
    """(finxray, heatmap). Returns one client twice when both point at one project."""
    from supabase import create_client

    if not (s.supabase_url and s.supabase_key):
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
    fx = create_client(s.supabase_url, s.supabase_key)
    hm_url = s.heatmap_url or s.supabase_url
    hm_key = s.heatmap_key or s.supabase_key
    if (hm_url, hm_key) == (s.supabase_url, s.supabase_key):
        return fx, fx
    return fx, create_client(hm_url, hm_key)


def make_bot_client(s: Settings):
    """Client for the bot's own project, or None when it is not configured."""
    from supabase import create_client

    if not (s.bot_url and s.bot_key):
        return None
    return create_client(s.bot_url, s.bot_key)
