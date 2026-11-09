"""Build out/results.json and the two 2560x1440 charts from the finished runs.

Run 1 is the headline (chosen before any run started). Runs 2 and 3 show the spread,
because the API no longer accepts temperature, so reruns can differ.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

import core
from config import (COST_BPS, CUTOFF, MODEL, OUT_DIR)

# Optional fonts: put InstrumentSerif-Regular.ttf and InstrumentSans-Variable.ttf
# in fonts/ to match the video. Without them, matplotlib's default font is used.
FONT_DIR = Path(__file__).resolve().parent / "fonts"
SERIF, SANS = "Instrument Serif", "Instrument Sans"
for f in ("InstrumentSerif-Regular.ttf", "InstrumentSans-Variable.ttf"):
    if (FONT_DIR / f).exists():
        font_manager.fontManager.addfont(str(FONT_DIR / f))
    else:
        SERIF = SANS = "DejaVu Sans"
matplotlib.rcParams["text.parse_math"] = False  # "$1" is money, not math

# Black, white and blue only, on flat paper (reel DECISIONS.md + brand scheme).
PAPER, INK, DECK = "#EEF0E6", "#0B0F14", "#46546A"
BLUE, BLUE_LIGHT, GRID = "#3367A3", "#A9C3D6", "#A9C3D6"

BANNED = ["return", "profit", "backtest", "alpha", "sharpe", "drawdown", "guarantee"]


def load_runs():
    runs = {}
    for p in sorted(OUT_DIR.glob("run*_curve.csv")):
        n = int(p.stem[3:].split("_")[0])
        df = pd.read_csv(p, parse_dates=["date"])
        runs[n] = df.set_index("date")["value"]
    return runs


def base_axes(fig):
    ax = fig.add_axes([0.07, 0.13, 0.72, 0.60])
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(DECK)
        ax.spines[s].set_linewidth(2)
    ax.grid(axis="y", color=GRID, alpha=0.42, linewidth=1.5)
    ax.tick_params(colors=DECK, labelsize=34, length=0, pad=14)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(SANS)
    return ax


def new_fig(title, sub):
    fig = plt.figure(figsize=(25.6, 14.4), dpi=100, facecolor=PAPER)
    fig.text(0.07, 0.89, title, fontfamily=SERIF, fontsize=96, color=INK, va="baseline")
    fig.text(0.07, 0.845, sub, linespacing=1.4, fontfamily=SANS, fontsize=36, color=DECK, va="top")
    return fig


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(18, dy), textcoords="offset points",
                fontfamily=SANS, fontsize=38, color=color, va="center",
                annotation_clip=False)


def fmt_dates(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(SANS)


def chart_value(runs, spy, start, end):
    fig = new_fig("What $1 became: Claude bot vs SPY",
                  f"{start:%-d %b} to {end:%-d %b %Y}: every day is after the model's training data ends.\n"
                  f"It saw prices only, no news. Each trade paid a small fee ({COST_BPS / 100:g}%).")
    ax = base_axes(fig)
    d0 = start - pd.Timedelta(days=1)
    for n, s in runs.items():
        if n == 1:
            continue
        ax.plot([d0] + list(s.index), [1.0] + list(s.values), color=BLUE_LIGHT, lw=3)
    s1 = runs[1]
    ax.plot([d0] + list(spy.index), [1.0] + list(spy.values), color=INK, lw=5)
    ax.plot([d0] + list(s1.index), [1.0] + list(s1.values), color=BLUE, lw=6)
    ax.axhline(1.0, color=DECK, lw=1.5, ls=(0, (6, 6)))
    b, m = s1.iloc[-1], spy.iloc[-1]
    gap = 0 if abs(b - m) > 0.012 else 22
    end_label(ax, s1.index[-1], b, f"Claude bot {(b - 1) * 100:+.1f}%", BLUE, dy=gap if b > m else -gap)
    end_label(ax, spy.index[-1], m, f"SPY {(m - 1) * 100:+.1f}%", INK, dy=-gap if b > m else gap)
    if len(runs) > 1:
        lo, hi = min(r.iloc[-1] for r in runs.values()), max(r.iloc[-1] for r in runs.values())
        fig.text(0.07, 0.04, f"Thin light lines: the same test run again ({len(runs)} runs, ended "
                 f"{(lo - 1) * 100:+.1f}% to {(hi - 1) * 100:+.1f}%). The model gives different answers each run.",
                 fontfamily=SANS, fontsize=30, color=DECK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    fmt_dates(ax)
    return fig


def chart_fall(bot, spy, start, end):
    fig = new_fig("How far each fell from its last high",
                  "The loss line: 0% means at a new high. Lower means a deeper fall.")
    ax = base_axes(fig)
    d0 = start - pd.Timedelta(days=1)
    fb, fs = core.fall_series(bot.values) * 100, core.fall_series(spy.values) * 100
    xb = [d0] + list(bot.index)
    ax.fill_between(xb, [0] + list(fb), 0, color=BLUE_LIGHT, alpha=0.55, lw=0)
    ax.plot(xb, [0] + list(fb), color=BLUE, lw=5)
    ax.plot([d0] + list(spy.index), [0] + list(fs), color=INK, lw=4)
    ax.axhline(0, color=DECK, lw=2)
    wb, ws = fb.min(), fs.min()
    ib, is_ = fb.argmin(), fs.argmin()
    ax.annotate(f"Claude bot worst: {wb:.1f}%", (bot.index[ib], wb), xytext=(-24, -6),
                textcoords="offset points", ha="right", va="center", fontfamily=SANS, fontsize=36,
                color=BLUE, annotation_clip=False)
    ax.annotate(f"SPY worst: {ws:.1f}%", (spy.index[is_], ws), xytext=(-24, -6),
                textcoords="offset points", ha="right", va="center", fontfamily=SANS, fontsize=36,
                color=INK, annotation_clip=False)
    ax.set_ylim(min(wb, ws) * 1.45, 0.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fmt_dates(ax)
    return fig


def figure_text(fig):
    out = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        out += [t.get_text() for t in ax.texts]
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        fig.canvas.draw()
        out += [l.get_text() for l in ax.get_xticklabels() + ax.get_yticklabels()]
    return out


def banned_hits(fig):
    return [(w, t) for t in figure_text(fig) for w in BANNED if w in t.lower()]


def main():
    opens, closes = core.load_prices()
    days = core.window_days(closes)
    start, end = days[0], days[-1]
    spy = pd.Series(dict(core.hold_spy_curve(days, opens, closes)))
    eq = pd.Series(dict(core.hold_equal_curve(days, opens, closes)))
    runs = load_runs()
    if 1 not in runs:
        raise SystemExit("run 1 is missing")
    bot = runs[1]
    assert list(bot.index) == list(spy.index), "run 1 did not finish the window"
    s1 = json.loads((OUT_DIR / "run1_summary.json").read_text())
    summaries = {n: json.loads((OUT_DIR / f"run{n}_summary.json").read_text()) for n in runs}

    res = {
        "bot_end_pct": round((bot.iloc[-1] - 1) * 100, 2),
        "spy_end_pct": round((spy.iloc[-1] - 1) * 100, 2),
        "bot_worst_fall_pct": round(core.worst_fall(bot.values) * 100, 2),
        "spy_worst_fall_pct": round(core.worst_fall(spy.values) * 100, 2),
        "equal_weight_hold_end_pct": round((eq.iloc[-1] - 1) * 100, 2),
        "equal_weight_hold_worst_fall_pct": round(core.worst_fall(eq.values) * 100, 2),
        "trading_days": len(days),
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "model": MODEL,
        "cutoff": CUTOFF,
        "total_cost_pct": s1["total_cost_pct"],
        "trade_count": s1["trade_count"],
        "notes": {
            "headline_run": 1,
            "end_pct": "change in account value from start to the last close, after trading costs",
            "worst_fall_pct": "largest fall from a previous high, as a positive number (6.2 = fell 6.2%)",
            "total_cost_pct": "all trading costs paid, as a % of the starting account",
            "trade_count": "one trade = one stock changed on one day (changes under 0.5% skipped)",
            "spy": "bought at the first open, paying the same 5 bps once, then held",
            "equal_weight_hold": "$1 split equally over the same 10 stocks at the first open, 5 bps once, never traded again",
            "refusal_days_run1": s1["refusal_days"],
        },
        "all_runs": {n: {k: s[k] for k in ("end_pct", "worst_fall_pct", "total_cost_pct",
                                           "trade_count", "refusal_days", "cost_usd")}
                     for n, s in summaries.items()},
        "api_tokens_in": sum(s["input_tokens"] for s in summaries.values()),
        "api_tokens_out": sum(s["output_tokens"] for s in summaries.values()),
        "api_cost_usd": round(sum(s["cost_usd"] for s in summaries.values()), 2),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    for name, fig in (("chart_a_value.png", chart_value(runs, spy, start, end)),
                      ("chart_b_loss_line.png", chart_fall(bot, spy, start, end))):
        hits = banned_hits(fig)
        if hits:
            raise SystemExit(f"banned words on {name}: {hits}")
        fig.savefig(OUT_DIR / name, dpi=100, facecolor=PAPER)
        plt.close(fig)
        print("wrote", OUT_DIR / name)


if __name__ == "__main__":
    main()
