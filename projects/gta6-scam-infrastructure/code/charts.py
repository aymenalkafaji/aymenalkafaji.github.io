"""
Charts for the GTA VI web-infrastructure analysis.

Usage: python charts.py <data_dir> <assets_dir>
Reads domains_labeled.csv, group_metrics.csv, sample_flow.csv and
first_seen_daily.csv from data_dir. The case-study figure is a redrawn layout
diagram (no screenshot pixels are reproduced).
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import pandas as pd

# ---- palette: reference categorical slots 1-3 (dark steps), validated
# all-pairs against the #0A0D18 card surface; gray = de-emphasised 'uncertain'.
COL_FAN = "#3987e5"
COL_DEC = "#d95926"
COL_TOKEN = "#199e70"
COL_UNC = "#8a8a96"
TEXT = "#F5F4F1"
TEXT_MUTED = "#B7B6C2"
GRID = "#2A2B3A"
CANVAS_BG = "#0A0D18"
CARD_BORDER = "#3A4560"
HUB = "#C9C8D3"

plt.rcParams.update({
    "figure.facecolor": CANVAS_BG, "axes.facecolor": "none", "savefig.facecolor": CANVAS_BG,
    "text.color": TEXT, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED,
    "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED,
    "font.family": "DejaVu Sans", "font.size": 13,
})

LABEL_COLOR = {"suspicious_offer": COL_DEC, "corroborated_malicious": COL_DEC,
               "fan_news": COL_FAN, "token": COL_TOKEN, "uncertain": COL_UNC}
PLATFORM_NAME = {"AS13335": "Cloudflare", "AS16509": "AWS edge*", "AS14618": "AWS",
                 "AS54113": "Fastly", "AS139341": "EdgeOne",
                 "AS13238": "Yandex Cloud", "AS203273": "AS203273"}


def card(fig, margin=0.018, left=0.075, bottom=0.11, top=0.09, right=0.02):
    bg = fig.add_axes((0, 0, 1, 1), zorder=-10)
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.add_patch(FancyBboxPatch((margin, margin), 1 - 2 * margin, 1 - 2 * margin,
                                boxstyle="round,pad=0,rounding_size=0.018", transform=bg.transAxes,
                                linewidth=1.3, edgecolor=CARD_BORDER, facecolor=CANVAS_BG))
    bg.axis("off")
    fig.subplots_adjust(left=margin + left, right=1 - margin - right, top=1 - margin - top, bottom=margin + bottom)


def title(fig, main, sub):
    fig.text(0.045, 0.935, main, fontsize=19, fontweight="bold", color=TEXT, va="top")
    fig.text(0.045, 0.885, sub, fontsize=12.5, color=TEXT_MUTED, va="top")


def network(d, out):
    """Small multiples: each shared IP (left) and each shared visual template
    (right) drawn as a hub with its member domains arranged around it."""
    import math
    keep = d[d.label.isin(LABEL_COLOR.keys())].copy()
    ip_groups = [(ip, g) for ip, g in keep.groupby("page_ip") if g.reg_domain.nunique() >= 3]
    ip_groups.sort(key=lambda t: -len(t[1]))
    tpl_groups = [(f, g) for f, g in keep.groupby("template_family") if g.reg_domain.nunique() >= 2]
    order = ["retailer_clone", "telegram_giveaway_yandex", "cyberleek_terminal", "apk_pink_download",
             "gta6hub_download", "edgeone_logo_splash", "apk_wordpress_empty", "gamespot_mirror",
             "leonida100_map", "vice_city_countdown", "leonida_token_map", "audacity_token"]
    tpl_groups.sort(key=lambda t: order.index(t[0]) if t[0] in order else 99)
    nice = {"retailer_clone": "Retailer clone", "telegram_giveaway_yandex": "TG giveaway",
            "gta6hub_download": "Hub + download", "apk_wordpress_empty": "Empty APK blog",
            "apk_pink_download": "APK download", "gamespot_mirror": "News-site mirror",
            "edgeone_logo_splash": "Logo splash", "cyberleek_terminal": "Cyberleek page",
            "leonida100_map": "Fan map mirror", "vice_city_countdown": "Countdown",
            "leonida_token_map": "Token map", "audacity_token": "Token site"}

    fig = plt.figure(figsize=(14, 9.6), dpi=150)
    card(fig, left=0, right=0, bottom=0, top=0)
    title(fig, "Who shares what: shared IPs vs shared page templates",
          "Each hub is one IP address (left) or one visually identical page template (right); dots are domains")

    def flower(ax, cx, cy, labels, hub_marker, caption, sub):
        n = len(labels)
        r = 0.30 if n <= 8 else 0.36
        for i, lab in enumerate(sorted(labels, key=lambda l: list(LABEL_COLOR).index(l))):
            ang = math.pi / 2 + 2 * math.pi * i / n
            x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
            ax.plot([cx, x], [cy, y], color="#3A4060", lw=0.9, zorder=1)
            ax.scatter([x], [y], s=95 if n <= 12 else 60, color=LABEL_COLOR[lab], edgecolors=CANVAS_BG,
                       linewidths=1.2, zorder=3)
        ax.scatter([cx], [cy], s=150, marker=hub_marker, color=CANVAS_BG, edgecolors=TEXT, linewidths=1.6, zorder=4)
        ax.text(cx, cy - 0.47, caption, ha="center", va="top", fontsize=10.5, color=TEXT, fontweight="bold")
        ax.text(cx, cy - 0.60, sub, ha="center", va="top", fontsize=9.5, color=TEXT_MUTED)

    # left panel: IPs
    axl = fig.add_axes((0.04, 0.12, 0.40, 0.70)); axl.axis("off")
    cols = 3
    rows = math.ceil(len(ip_groups) / cols)
    for i, (ip, g) in enumerate(ip_groups):
        cx, cy = (i % cols) * 1.3, -(i // cols) * 1.35
        asn = g.page_asn.iloc[0]
        flower(axl, cx, cy, list(g.label), "o", PLATFORM_NAME.get(asn, asn), f"{len(g)} domains")
    axl.set_xlim(-0.6, (cols - 1) * 1.3 + 0.6); axl.set_ylim(-(rows - 1) * 1.35 - 0.8, 0.5)
    axl.set_aspect("equal")
    fig.text(0.24, 0.835, "Shared IPs (3+ domains)", ha="center", fontsize=13.5, color=TEXT, fontweight="bold")

    # right panel: templates
    axr = fig.add_axes((0.52, 0.12, 0.45, 0.70)); axr.axis("off")
    cols = 4
    rows = math.ceil(len(tpl_groups) / cols)
    for i, (f, g) in enumerate(tpl_groups):
        cx, cy = (i % cols) * 1.15, -(i // cols) * 1.35
        flower(axr, cx, cy, list(g.label), "D", nice.get(f, f), f"{len(g)} domains")
    axr.set_xlim(-0.6, (cols - 1) * 1.15 + 0.6); axr.set_ylim(-(rows - 1) * 1.35 - 0.8, 0.5)
    axr.set_aspect("equal")
    fig.text(0.745, 0.835, "Shared visual templates", ha="center", fontsize=13.5, color=TEXT, fontweight="bold")

    handles = [Line2D([], [], marker="o", ls="", color=COL_DEC, markersize=9, label="Deceptive offer"),
               Line2D([], [], marker="o", ls="", color=COL_FAN, markersize=9, label="Fan / news"),
               Line2D([], [], marker="o", ls="", color=COL_TOKEN, markersize=9, label="Meme token"),
               Line2D([], [], marker="o", ls="", color=COL_UNC, markersize=9, label="Uncertain")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11.5,
               bbox_to_anchor=(0.5, 0.035), labelcolor=TEXT_MUTED)
    fig.text(0.045, 0.105, "*AWS edge IPs here front Vercel and Netlify deployments. TG = Telegram.",
             fontsize=10, color=TEXT_MUTED)
    fig.savefig(out / "chart_infra_network.png", dpi=150)
    plt.close(fig)


def offers(d, out):
    dec = d[d.label.isin(["suspicious_offer", "corroborated_malicious"])]
    g = dec.groupby("offer_group").agg(n=("reg_domain", "size"),
                                       corr=("label", lambda s: int((s == "corroborated_malicious").sum())))
    g = g.sort_values("n")
    fig = plt.figure(figsize=(12, 6.8), dpi=150)
    card(fig, left=0.29, bottom=0.12, top=0.17, right=0.06)
    ax = fig.add_subplot(111)
    title(fig, f"What the {len(dec)} deceptive-offer domains were offering",
          "Primary offer seen in the archived screenshot, Oct 18 - Nov 15, 2026")
    y = range(len(g))
    ax.barh(y, g.n, color=COL_DEC, height=0.62, zorder=3)
    for i, (n, c) in enumerate(zip(g.n, g["corr"])):
        note = f"{n}" + (f"   ({c} also tagged by third parties)" if c else "")
        ax.text(n + 0.3, i, note, va="center", fontsize=11.5, color=TEXT)
    ax.set_yticks(list(y)); ax.set_yticklabels(g.index, fontsize=12, color=TEXT)
    ax.set_xlim(0, max(g.n) * 1.5)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("Domains", fontsize=11.5)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.045, 0.045, "Each domain is counted once, under its primary offer.", fontsize=10.5, color=TEXT_MUTED)
    fig.savefig(out / "chart_offer_types.png", dpi=150)
    plt.close(fig)


def metrics(m, out):
    order = ["template_reuse", "shared_ip", "shared_ip_noncdn", "cdn_paas", "platform_subdomain", "new_domain"]
    names = {"template_reuse": "Shares a visual page template",
             "shared_ip": "Shares an IP (any network)",
             "shared_ip_noncdn": "Shares an IP off CDN / PaaS",
             "cdn_paas": "Hosted on a major CDN / PaaS",
             "platform_subdomain": "Free-platform subdomain",
             "new_domain": "Registered < 30 days before scan*"}
    m = m.set_index("metric").loc[order]
    fig = plt.figure(figsize=(12, 7.2), dpi=150)
    card(fig, left=0.30, bottom=0.17, top=0.17, right=0.05)
    ax = fig.add_subplot(111)
    title(fig, "Deceptive-offer sites vs fan / news sites",
          "Share of domains with each signal, with 95% Wilson confidence intervals")
    for i, (k, r) in enumerate(m.iterrows()):
        yy = len(order) - 1 - i
        for pct, ci, col, dy in [(r.deceptive_pct, r.deceptive_ci, COL_DEC, 0.14), (r.fan_pct, r.fan_ci, COL_FAN, -0.14)]:
            lo, hi = [float(x) for x in ci.split("-")]
            ax.plot([lo, hi], [yy + dy] * 2, color=col, lw=2, solid_capstyle="round", zorder=3)
            ax.scatter([pct], [yy + dy], s=70, color=col, edgecolors=CANVAS_BG, linewidths=1.5, zorder=4)
        ax.text(101, yy + 0.14, f"{r.deceptive_pct:.0f}%", va="center", fontsize=11, color=TEXT)
        ax.text(101, yy - 0.14, f"{r.fan_pct:.0f}%", va="center", fontsize=11, color=TEXT_MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in reversed(order)], fontsize=12, color=TEXT)
    ax.set_xlim(0, 108); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    nd = int(m.loc["template_reuse", "deceptive_n"]); nf = int(m.loc["template_reuse", "fan_n"])
    handles = [Line2D([], [], marker="o", color=COL_DEC, lw=2, markersize=8, label=f"Deceptive offer (n={nd})"),
               Line2D([], [], marker="o", color=COL_FAN, lw=2, markersize=8, label=f"Fan / news (n={nf})")]
    fig.legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=11.5,
               bbox_to_anchor=(0.04, 0.035), labelcolor=TEXT_MUTED)
    fig.text(0.96, 0.05, "*independently registered domains only (n=40 vs 112)", fontsize=10,
             color=TEXT_MUTED, ha="right")
    fig.savefig(out / "chart_group_metrics.png", dpi=150)
    plt.close(fig)


def social(m, out):
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    card(fig)
    fig.text(0.07, 0.80, "Mapping GTA VI-themed web infrastructure", fontsize=30, fontweight="bold", color=TEXT)
    fig.text(0.07, 0.71, "Deceptive offers, fan sites and shared hosting, Oct 18 - Nov 15, 2026",
             fontsize=17, color=TEXT_MUTED)
    r = m.set_index("metric").loc["template_reuse"]
    fig.text(0.07, 0.42, f"{r.deceptive_pct:.0f}%", fontsize=88, fontweight="bold", color=COL_DEC)
    fig.text(0.07, 0.30, "of deceptive-offer domains\nreused a visual template", fontsize=17, color=TEXT, va="top")
    fig.text(0.58, 0.42, f"{r.fan_pct:.0f}%", fontsize=88, fontweight="bold", color=COL_FAN)
    fig.text(0.58, 0.30, "of fan / news domains\ndid the same", fontsize=17, color=TEXT, va="top")
    fig.text(0.07, 0.08, "335 domains from 444 urlscan.io scans, labelled from archived screenshots",
             fontsize=14, color=TEXT_MUTED)
    fig.savefig(out / "social-card.png", dpi=100)
    plt.close(fig)


def flow(f, out):
    """Sample-selection flow: which domains feed which analysis."""
    n = dict(zip(f.step, f.n))
    fig = plt.figure(figsize=(14, 6.4), dpi=150)
    card(fig, left=0, right=0, bottom=0, top=0)
    title(fig, "From search results to the comparison groups",
          "Every number on this page comes from one of these boxes")
    ax = fig.add_axes((0.03, 0.05, 0.94, 0.76)); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.4)

    def box(x, y, w, h, big, small, edge=CARD_BORDER, fill="#121729"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    linewidth=1.4, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h * 0.62, big, ha="center", va="center", fontsize=17, fontweight="bold", color=TEXT)
        ax.text(x + w / 2, y + h * 0.27, small, ha="center", va="center", fontsize=10.5, color=TEXT_MUTED)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=TEXT_MUTED, lw=1.3, shrinkA=0, shrinkB=0))

    w, h, y = 2.6, 1.25, 4.4
    xs = [0.3, 3.5, 6.7, 11.0]
    box(xs[0], y, w, h, f"{n['rows returned by 11 queries']}", "search results, 11 queries")
    box(xs[1], y, w, h, f"{n['unique scans']}", "unique scans")
    box(xs[2], y, w, h, f"{n['registrable domains']}", "registrable domains")
    box(xs[3], y, w, h, f"{n['relevant domains']}", "relevant to GTA VI")
    for a, b in zip(xs, xs[1:]):
        arrow(a + w, y + h / 2, b - 0.05, y + h / 2)
    gx = (xs[2] + w + xs[3]) / 2
    ax.text(gx, y + h + 0.25, f"minus {n['out of scope (parked, error, unrelated)']} out of scope\n(parked, error,\nunrelated)",
            ha="center", va="bottom", fontsize=10.5, color=TEXT_MUTED)

    get = lambda k: n.get(f"relevant: {k}", 0)
    dec = get("suspicious_offer") + get("corroborated_malicious")
    other = n["relevant domains"] - dec - get("fan_news")
    y2, h2 = 0.7, 1.75
    box(1.2, y2, 3.6, h2, f"{dec}", f"deceptive offer\n({get('suspicious_offer')} offer only, "
        f"{get('corroborated_malicious')} offer + threat tag)", edge=COL_DEC)
    box(5.4, y2, 3.2, h2, f"{get('fan_news')}", "fan / news", edge=COL_FAN)
    box(9.2, y2, 4.4, h2, f"{other}", f"not in the main comparison\n{get('token')} meme token, {get('uncertain')} uncertain,\n"
        f"{get('commerce')} commerce, {get('official')} official", edge=GRID, fill=CANVAS_BG)
    for cx in [3.0, 7.0, 11.4]:
        arrow(xs[3] + w / 2, y - 0.05, cx, y2 + h2 + 0.05)
    ax.text(5.0, 0.25, "Main comparison", ha="center", fontsize=11, color=TEXT_MUTED)
    ax.plot([1.2, 8.6], [0.5, 0.5], color=TEXT_MUTED, lw=1)
    fig.savefig(out / "chart_sample_flow.png", dpi=150)
    plt.close(fig)


def timeline(t, out):
    """First observed scan per domain, by day: deceptive vs fan/news."""
    t = t.set_index("day")
    days = pd.to_datetime(t.index)
    fig = plt.figure(figsize=(12, 7.0), dpi=150)
    card(fig, left=0.06, bottom=0.14, top=0.17, right=0.04)
    title(fig, "When each domain was first scanned",
          "First urlscan.io sighting per domain, by UTC day. Scan dates are not deployment dates")
    gs = fig.add_gridspec(2, 1, hspace=0.35)
    ymax = max(t.deceptive.max(), t.fan_news.max()) + 3
    for i, (col, c, lab) in enumerate([("deceptive", COL_DEC, "Deceptive offer"), ("fan_news", COL_FAN, "Fan / news")]):
        ax = fig.add_subplot(gs[i])
        ax.bar(days, t[col], width=0.75, color=c, zorder=3)
        ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.text(0.0, 1.04, f"{lab} ({int(t[col].sum())} domains)", transform=ax.transAxes, fontsize=12.5,
                color=TEXT, fontweight="bold")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=0))
        if col == "deceptive":
            pk = t.deceptive.idxmax()
            ax.annotate(f"{int(t.deceptive.max())} domains, all from one\nsubmitter feed on the same day",
                        xy=(pd.Timestamp(pk), t.deceptive.max()), xytext=(pd.Timestamp(pk) + pd.Timedelta(days=2.5), ymax * 0.82),
                        fontsize=10.5, color=TEXT_MUTED, va="top",
                        arrowprops=dict(arrowstyle="-", color=TEXT_MUTED, lw=0.9))
    fig.text(0.045, 0.045, "Domains with an uncertain, token, commerce or official label are not shown.",
             fontsize=10, color=TEXT_MUTED)
    fig.savefig(out / "chart_first_seen.png", dpi=150)
    plt.close(fig)


def case_studies(out):
    """Redrawn layout diagrams of two template families, drawn from the
    archived screenshots. Text is paraphrased; no pixels are reproduced."""
    fig = plt.figure(figsize=(14, 10.2), dpi=150)
    card(fig, left=0, right=0, bottom=0, top=0)
    title(fig, "Two reused templates, redrawn",
          "Layout diagrams drawn from urlscan.io screenshots. Orange outlines mark the offer")
    ax = fig.add_axes((0.03, 0.03, 0.94, 0.82)); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 10.6)

    def blk(x, y, w, h, text="", fill="#1A2033", edge=None, fs=8.5, col=TEXT_MUTED, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    linewidth=1.8 if edge else 0, edgecolor=edge or fill, facecolor=fill))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=col,
                    fontweight="bold" if bold else "normal")

    def page(x, y, w, h):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    linewidth=1.2, edgecolor=CARD_BORDER, facecolor="#0F1322"))

    # row 1: retailer clone, 3 identical copies
    ax.text(0.2, 10.35, "A. Cloned retailer product page: identical on 3 domains", fontsize=13.5, color=TEXT,
            fontweight="bold", va="top")
    ax.text(0.2, 9.95, "Copy of a major Brazilian retailer's store, offering a GTA VI PS5 'pre-order' with a Pix payment discount",
            fontsize=10.5, color=TEXT_MUTED, va="top")
    doms = ["magaoferta[.]netlify[.]app", "magazineluizaofertas[.]com[.]br", "ofertasdamadrugada[.]com"]
    for i, dname in enumerate(doms):
        x, y, w, h = 0.2 + i * 4.6, 5.75, 4.3, 3.9
        page(x, y, w, h)
        blk(x + 0.1, y + 3.45, w - 0.2, 0.32, "Sale banner (11.11 Black Friday)", fill="#2A2F45")
        blk(x + 0.1, y + 3.0, w - 0.2, 0.38, "Retailer header, logo, search", fill="#1E4E9A", col=TEXT)
        blk(x + 0.2, y + 1.55, 1.3, 1.3, "Game\ncover art", fill="#2A2F45")
        blk(x + 1.65, y + 2.3, 1.25, 0.55, "Product title:\nGTA VI PS5\npre-order", fs=7.5)
        blk(x + 3.0, y + 2.3, 1.15, 0.55, "Price R$ 249\nPix discount", edge=COL_DEC, fs=7.5, col=TEXT)
        blk(x + 3.0, y + 1.6, 1.15, 0.45, "BUY", fill="#1F7A3A", edge=COL_DEC, col=TEXT, bold=True)
        blk(x + 1.65, y + 1.6, 1.25, 0.55, "Delivery and\nseller info", fs=7.5)
        blk(x + 0.2, y + 0.45, w - 0.4, 0.9, "'Customers also viewed' row: consoles, controllers", fs=7.5)
        ax.text(x + w / 2, y - 0.18, dname, ha="center", va="top", fontsize=10.5, color=TEXT)

    # row 2: Telegram giveaway, same page in three languages
    ax.text(0.2, 4.75, "B. Telegram giveaway page: one template, three languages", fontsize=13.5, color=TEXT,
            fontweight="bold", va="top")
    ax.text(0.2, 4.35, "Unofficial 'fan' page promising a free copy at launch to channel subscribers, plus a pre-order button",
            fontsize=10.5, color=TEXT_MUTED, va="top")
    langs = [("German", "gta6-de[.]website[.]yandexcloud[.]net"), ("Dutch", "gta6-nl[.]website[.]yandexcloud[.]net"),
             ("Swedish", "gta6-sv[.]website[.]yandexcloud[.]net")]
    for i, (lang, dname) in enumerate(langs):
        x, y, w, h = 0.2 + i * 4.6, 0.35, 4.3, 3.45
        page(x, y, w, h)
        blk(x + 0.1, y + 3.05, w - 0.2, 0.3, "Nav bar + release countdown", fill="#2A2F45")
        blk(x + 0.25, y + 2.45, 1.6, 0.3, "Badge: unofficial, Telegram", fs=7.5)
        blk(x + 0.25, y + 1.75, 2.4, 0.6, f"Headline ({lang}):\n'GTA 6 release date'", fs=8, col=TEXT)
        blk(x + 0.25, y + 1.2, 2.4, 0.45, "Text: subscribers get\na free copy at launch", edge=COL_DEC, fs=7.5, col=TEXT)
        blk(x + 0.25, y + 0.55, 1.15, 0.4, "Join giveaway", edge=COL_DEC, fs=7.5, col=TEXT)
        blk(x + 1.5, y + 0.55, 1.15, 0.4, "Pre-order", fill="#B03A6A", edge=COL_DEC, fs=7.5, col=TEXT)
        blk(x + 2.85, y + 0.55, 1.25, 1.9, "Game\ncover art", fill="#2A2F45")
        blk(x + 0.1, y + 0.1, w - 0.2, 0.3, "Fact strip: platforms, price, date", fill="#1A2033", fs=7.5)
        ax.text(x + w / 2, y - 0.12, dname, ha="center", va="top", fontsize=10.5, color=TEXT)
    fig.savefig(out / "chart_case_studies.png", dpi=150)
    plt.close(fig)


def main(data, assets):
    data, assets = Path(data), Path(assets)
    assets.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(data / "domains_labeled.csv")
    m = pd.read_csv(data / "group_metrics.csv")
    network(d, assets)
    offers(d, assets)
    metrics(m, assets)
    social(m, assets)
    flow(pd.read_csv(data / "sample_flow.csv"), assets)
    timeline(pd.read_csv(data / "first_seen_daily.csv"), assets)
    case_studies(assets)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
