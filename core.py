"""Pure logic: load prices, build the daily prompt, parse replies, simulate the account.

No network here, so tests can drive all of it.

Timing rule (the honest part):
  On trading day t the model sees closes up to day t-1 only.
  Its weights are traded at the OPEN of day t, then held.
  So the decision never sees day t, and the trade price is one it could get.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from config import (BENCH, COST_BPS, LOOKBACK_DAYS, RAW_DIR, STOCKS,
                    TRADE_BAND, WINDOW_START)

SYSTEM_PROMPT = (
    "You manage a long-only US stock account. Each trading day you choose target "
    "weights for 10 stocks and cash. You see only daily closing prices up to "
    "yesterday's close. You get no news. Your trades fill at today's open. "
    "Every trade costs 5 basis points of the traded value. Weights are fractions "
    "of the account between 0 and 1. The 10 stock weights plus cash must add up to 1. "
    "Reply only in the JSON format you are given. Keep 'reason' under 40 words."
)


def load_prices(raw_dir=RAW_DIR):
    """Return (opens, closes): DataFrames indexed by date, one column per ticker."""
    opens, closes = {}, {}
    for t in STOCKS + [BENCH]:
        df = pd.read_csv(raw_dir / f"{t}.csv", index_col="Date", parse_dates=True)
        opens[t] = df["Open"]
        closes[t] = df["Close"]
    o = pd.DataFrame(opens).sort_index()
    c = pd.DataFrame(closes).sort_index()
    if o.isna().any().any() or c.isna().any().any():
        raise ValueError("missing prices")
    return o, c


def window_days(closes, start=WINDOW_START, end=None):
    idx = closes.index[closes.index >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return list(idx)


def build_prompt(closes, day, holdings):
    """User message for trading day `day`. Uses only closes dated before `day`."""
    day = pd.Timestamp(day)
    hist = closes.loc[closes.index < day].tail(LOOKBACK_DAYS)
    # Hard guard: nothing dated `day` or later may reach the model.
    assert hist.index.max() < day, "future data in prompt"
    assert len(hist) == LOOKBACK_DAYS, "not enough history"
    cols = STOCKS + [BENCH]
    lines = ["date," + ",".join(cols)]
    for d, row in hist[cols].iterrows():
        lines.append(d.strftime("%Y-%m-%d") + "," + ",".join(f"{row[c]:.2f}" for c in cols))
    hold = ", ".join(f"{t} {holdings.get(t, 0.0):.3f}" for t in STOCKS)
    return (
        f"Today is {day.strftime('%Y-%m-%d')} (before the open).\n"
        f"Daily closing prices (adjusted), last {LOOKBACK_DAYS} trading days, "
        f"up to yesterday. {BENCH} is shown for context only; you cannot trade it.\n"
        + "\n".join(lines)
        + f"\n\nYour weights now (after price moves): {hold}, cash {holdings.get('CASH', 1.0):.3f}.\n"
        "Give your target weights for today."
    )


def prompt_hash(system, user):
    return hashlib.sha256((system + "\n---\n" + user).encode()).hexdigest()[:16]


def output_schema():
    props = {t: {"type": "number"} for t in STOCKS}
    props["CASH"] = {"type": "number"}
    props["reason"] = {"type": "string"}
    return {
        "type": "object",
        "properties": props,
        "required": STOCKS + ["CASH", "reason"],
        "additionalProperties": False,
    }


def parse_weights(reply):
    """Clean the model's weights: no shorts, no leverage. Cash takes the rest."""
    w = {t: max(0.0, float(reply.get(t, 0.0))) for t in STOCKS}
    s = sum(w.values())
    if s > 1.0:
        w = {t: v / s for t, v in w.items()}
    w["CASH"] = 1.0 - sum(w[t] for t in STOCKS)
    return w


def drift(weights, growth):
    """Weights after each holding grows by `growth[t]` (cash grows by 1)."""
    vals = {t: weights[t] * growth.get(t, 1.0) for t in weights}
    tot = sum(vals.values())
    return {t: v / tot for t, v in vals.items()}, tot


class Account:
    """Tracks one account through the window. Start value 1.0, all cash."""

    def __init__(self):
        self.value = 1.0
        self.w = {**{t: 0.0 for t in STOCKS}, "CASH": 1.0}
        self.last_close = None
        self.cost_total = 0.0      # in account units (start = 1.0)
        self.trades = 0
        self.curve = []            # (date, value at close)

    def step(self, day, target, opens, closes):
        """Trade at the open of `day` to `target`, then mark at the close."""
        day = pd.Timestamp(day)
        if self.last_close is not None:
            g = {t: opens.at[day, t] / closes.at[self.last_close, t] for t in STOCKS}
            self.w, f = drift(self.w, g)
            self.value *= f
        traded = 0.0
        new_w = dict(self.w)
        for t in STOCKS:
            dw = target[t] - self.w[t]
            if abs(dw) >= TRADE_BAND:
                new_w[t] = target[t]
                traded += abs(dw)
                self.trades += 1
        new_w["CASH"] = 1.0 - sum(new_w[t] for t in STOCKS)
        cost = traded * self.value * COST_BPS / 1e4
        self.value -= cost
        self.cost_total += cost
        self.w = new_w
        g = {t: closes.at[day, t] / opens.at[day, t] for t in STOCKS}
        self.w, f = drift(self.w, g)
        self.value *= f
        self.last_close = day
        self.curve.append((day, self.value))


def hold_spy_curve(days, opens, closes):
    """Buy SPY at the first open (paying the same 5 bps once), hold to the end."""
    start = pd.Timestamp(days[0])
    units = (1.0 - COST_BPS / 1e4) / opens.at[start, BENCH]
    return [(pd.Timestamp(d), units * closes.at[pd.Timestamp(d), BENCH]) for d in days]


def hold_equal_curve(days, opens, closes):
    """Split $1 equally over the 10 stocks at the first open (5 bps once), never trade again."""
    start = pd.Timestamp(days[0])
    each = (1.0 - COST_BPS / 1e4) / len(STOCKS)
    units = {t: each / opens.at[start, t] for t in STOCKS}
    return [(pd.Timestamp(d), sum(u * closes.at[pd.Timestamp(d), t] for t, u in units.items()))
            for d in days]


def worst_fall(values):
    """Largest fall from a previous peak, as a positive fraction (0.062 = fell 6.2%).
    The peak includes the start value 1.0."""
    v = np.asarray([1.0] + list(values), dtype=float)
    peak = np.maximum.accumulate(v)
    return float(np.max(1.0 - v / peak))


def fall_series(values):
    v = np.asarray([1.0] + list(values), dtype=float)
    peak = np.maximum.accumulate(v)
    return (v / peak - 1.0)[1:]
