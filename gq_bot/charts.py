"""PNG price charts for Telegram. matplotlib with the Agg backend (no display)."""

from __future__ import annotations

import io
from datetime import date
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

GREEN = "#16a34a"
PALETTE = [GREEN, "#2563eb", "#d97706", "#7c3aed", "#db2777"]


def _d(s: str) -> date:
    return date.fromisoformat(s[:10])


def price_chart(series: Dict[str, List[Dict]], title: str, normalise: bool,
                show_ma: bool = True, plot_from: str = "") -> bytes:
    """One ticker: price with 50/200-day averages. Several: rebased to 100."""
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=130)
    for i, (name, bars) in enumerate(series.items()):
        full_bars = [b for b in bars if b.get("close") is not None]
        bars = [b for b in full_bars if b["date"] >= plot_from]
        if not bars:
            continue
        skip = len(full_bars) - len(bars)   # earlier bars only warm up the averages
        xs = [_d(b["date"]) for b in bars]
        ys = [float(b["close"]) for b in bars]
        if normalise:
            ys = [y / ys[0] * 100 for y in ys]
        ax.plot(xs, ys, color=PALETTE[i % len(PALETTE)], lw=1.8, label=name)
        if show_ma and not normalise and len(series) == 1:
            full = [float(b["close"]) for b in full_bars]
            for n, style in ((50, "--"), (200, ":")):
                if len(full) >= n:
                    ma = [None] * (n - 1) + [sum(full[j - n + 1:j + 1]) / n for j in range(n - 1, len(full))]
                    ax.plot(xs, ma[skip:], color="#6b7280", lw=1, ls=style, label=f"{n}-day avg")
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    ax.set_ylabel("Rebased to 100" if normalise else "Price (USD)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.99, 0.985, "GQuants", ha="right", va="top", fontsize=8, color=GREEN)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
