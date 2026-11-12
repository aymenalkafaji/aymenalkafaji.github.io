import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch

# ---- palette (validated against the site's Depth/#12131F navy block) ----
COL_NVDA = "#BD6F50"   # terracotta/signal — the subject, the site's brand accent
COL_AMD  = "#3F6FB0"   # steel blue
COL_AVGO = "#B08D1F"   # bronze
COL_MKT  = "#9A9AA5"   # neutral muted grey — Nasdaq benchmark
TEXT      = "#F5F4F1"
TEXT_MUTED = "#B7B6C2"
GRID      = "#2A2B3A"
GOOD      = "#4C9A6A"
BAD       = "#C24B3F"

# ---- card frame (padded rounded-border card, matching the site's chosen look) ----
CANVAS_BG = "#0A0D18"
CARD_BORDER = "#3A4560"

plt.rcParams.update({
    "figure.facecolor": CANVAS_BG,
    "axes.facecolor": "none",
    "savefig.facecolor": CANVAS_BG,
    "text.color": TEXT,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "font.family": "DejaVu Sans",
    "font.size": 13,
})

def add_card_frame(fig, margin=0.018):
    """Draws a padded, rounded-border card frame behind the chart, inset
    from the canvas edge, matching the site's chosen chart style. Uses a
    dedicated background Axes (not fig.patches, which draws above Axes
    content regardless of zorder) so the card sits behind the plot."""
    from matplotlib.patches import FancyBboxPatch
    bg_ax = fig.add_axes((0, 0, 1, 1), zorder=-10)
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.add_patch(FancyBboxPatch(
        (margin, margin), 1 - 2 * margin, 1 - 2 * margin,
        boxstyle="round,pad=0,rounding_size=0.018",
        transform=bg_ax.transAxes, linewidth=1.3,
        edgecolor=CARD_BORDER, facecolor=CANVAS_BG,
    ))
    bg_ax.axis("off")
    fig.subplots_adjust(left=margin + 0.075, right=1 - margin - 0.02,
                         top=1 - margin - 0.09, bottom=margin + 0.11)

prices = pd.read_csv("daily_prices.csv", index_col=0, parse_dates=True)
rets = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)
summary = pd.read_csv("summary.csv", index_col=0)

def style_axes(ax):
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

# =================================================================
# Chart 1 — Indexed price performance, Oct 1 - Nov 10 2026
# =================================================================
window = prices.loc["2026-10-01":"2026-11-10"]
idx = window / window.iloc[0] * 100

fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)

event_start = pd.Timestamp("2026-11-09")
event_end = pd.Timestamp("2026-11-10")
ax.axvspan(event_start, event_end + pd.Timedelta(hours=18), color=BAD, alpha=0.10, zorder=0)
ax.text(event_start, idx.values.max() * 1.015, "Scandal-specific\ncrash (Nov 9–10)",
        color=BAD, fontsize=10.5, fontweight="bold", va="bottom")

series = [("NVDA", COL_NVDA, "NVDA"), ("AMD", COL_AMD, "AMD"), ("AVGO", COL_AVGO, "AVGO"), ("IXIC", COL_MKT, "Nasdaq Composite")]
for col, color, label in series:
    lw = 2.6 if col == "NVDA" else 1.8
    plot_kwargs = dict(color=color, linewidth=lw, label=label, zorder=3)
    if col == "IXIC":
        plot_kwargs["dashes"] = (5, 2)
    line, = ax.plot(idx.index, idx[col], **plot_kwargs)
    # highlight the two event-window points
    ax.scatter(idx.loc[[event_start, event_end]].index, idx.loc[[event_start, event_end], col],
                color=color, s=28, zorder=4, edgecolor="#12131F", linewidth=0.8)
    ax.annotate(label, xy=(idx.index[-1], idx[col].iloc[-1]), xytext=(8, 0),
                textcoords="offset points", color=color, fontsize=11.5, fontweight="bold",
                va="center")

style_axes(ax)
ax.set_ylabel("Price, indexed to 100 on Oct 1, 2026")
ax.set_xlim(idx.index.min(), idx.index.max() + pd.Timedelta(days=4))
ax.set_title("NVIDIA vs. chip peers vs. the market, Oct–Nov 2026", color=TEXT, fontsize=15.5,
             fontweight="bold", loc="left", pad=14)
ax.legend(loc="lower left", frameon=False, labelcolor=TEXT_MUTED, fontsize=10.5, ncol=2)
add_card_frame(fig)
fig.savefig("chart_indexed_prices.png", dpi=200)
plt.close(fig)

# =================================================================
# Chart 2 — Two-session dollar market-value change, Nov 9-10
# =================================================================
fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
syms = ["NVDA", "AMD", "AVGO"]
colors = [COL_NVDA, COL_AMD, COL_AVGO]
vals = [summary.loc[s, "total_dollar_B"] for s in syms]
bars = ax.bar(syms, vals, color=colors, width=0.56, zorder=3)
for b, v in zip(bars, vals):
    va = "bottom" if v >= 0 else "top"
    offset = 6 if v >= 0 else -6
    ax.annotate(f"{'+' if v>=0 else ''}${v:,.0f}B", xy=(b.get_x()+b.get_width()/2, v),
                xytext=(0, offset), textcoords="offset points", ha="center", va=va,
                color=TEXT, fontsize=13, fontweight="bold")
ax.axhline(0, color=GRID, linewidth=1.2, zorder=2)
style_axes(ax)
ax.spines["bottom"].set_visible(False)
ax.set_ylabel("Market value change, Nov 9–10 2026 ($B)")
ax.set_title("NVIDIA fell while its chip peers rose", color=TEXT, fontsize=15.5,
             fontweight="bold", loc="left", pad=14)
add_card_frame(fig)
fig.savefig("chart_two_day_divergence.png", dpi=200)
plt.close(fig)

# =================================================================
# Chart 3 — NVDA waterfall: before -> market/sector effect -> scandal-specific -> after
# =================================================================
before = summary.loc["NVDA", "mcap_before_B"]
market_fx = summary.loc["NVDA", "market_explained_dollar_B"]
abnormal_fx = summary.loc["NVDA", "abnormal_dollar_B"]
after = summary.loc["NVDA", "mcap_after_B"]

labels = ["Market cap\nNov 6 close", "Market/sector\neffect", "Scandal-specific\n(abnormal) effect", "Market cap\nNov 10 close"]
starts = [0, before, before + market_fx]
heights = [before, market_fx, abnormal_fx]
bar_colors = [TEXT_MUTED, GOOD if market_fx >= 0 else BAD, BAD if abnormal_fx < 0 else GOOD]

fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)

# base + step bars
ax.bar(0, before, width=0.6, color="#6B6C7C", zorder=3)
cum = before
step_x = [1, 2]
step_vals = [market_fx, abnormal_fx]
step_colors = [GOOD if market_fx >= 0 else BAD, BAD if abnormal_fx < 0 else GOOD]
for x, v, c in zip(step_x, step_vals, step_colors):
    bottom = cum if v >= 0 else cum + v
    ax.bar(x, abs(v), bottom=bottom, width=0.6, color=c, zorder=3)
    va = "bottom" if v >= 0 else "top"
    offset = 6 if v >= 0 else -6
    ax.annotate(f"{'+' if v>=0 else '−'}${abs(v):,.0f}B", xy=(x, bottom + abs(v) if v>=0 else bottom),
                xytext=(0, offset), textcoords="offset points", ha="center", va=va,
                color=TEXT, fontsize=12.5, fontweight="bold")
    cum += v
ax.bar(3, cum, width=0.6, color="#6B6C7C", zorder=3)

for x, v in [(0, before), (3, cum)]:
    ax.annotate(f"${v:,.0f}B", xy=(x, v), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", color=TEXT, fontsize=12.5, fontweight="bold")

# connector lines
ax.plot([0.3, 0.7], [before, before], color=GRID, linewidth=1, zorder=2)
ax.plot([1.3, 1.7], [before + market_fx, before + market_fx], color=GRID, linewidth=1, zorder=2)

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylim(0, before * 1.08)
style_axes(ax)
ax.spines["bottom"].set_visible(False)
ax.set_ylabel("NVIDIA market capitalization ($B)")
ax.set_title("Where NVIDIA's two-day loss actually came from", color=TEXT, fontsize=15.5,
             fontweight="bold", loc="left", pad=14)
add_card_frame(fig)
fig.savefig("chart_nvda_waterfall.png", dpi=200)
plt.close(fig)

print("Saved chart_indexed_prices.png, chart_two_day_divergence.png, chart_nvda_waterfall.png")
