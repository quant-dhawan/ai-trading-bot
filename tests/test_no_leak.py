"""No data from day t or later may reach the decision for day t.

These tests call the same build_prompt the live run uses.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402
import run_bot  # noqa: E402
from config import RAW_DIR, STOCKS  # noqa: E402

if not (RAW_DIR / "SPY.csv").exists():
    pytest.skip("no price data: run fetch_data.py first", allow_module_level=True)

OPENS, CLOSES = core.load_prices()
DAYS = core.window_days(CLOSES)
HOLD = {**{t: 0.1 for t in STOCKS}, "CASH": 0.0}


@pytest.mark.parametrize("i", [0, 1, len(DAYS) // 2, len(DAYS) - 1])
def test_shuffling_the_future_changes_nothing(i):
    day = DAYS[i]
    before = core.build_prompt(CLOSES, day, HOLD)
    rng = np.random.default_rng(i)
    poisoned = CLOSES.copy()
    future = poisoned.index >= day
    # Shuffle the future rows and also scale them: any leak would change the prompt.
    vals = poisoned.loc[future].to_numpy()
    poisoned.loc[future] = rng.permutation(vals) * rng.uniform(0.5, 2.0, vals.shape)
    after = core.build_prompt(poisoned, day, HOLD)
    assert before == after


def test_latest_date_in_prompt_is_the_day_before():
    for i, day in enumerate(DAYS):
        p = core.build_prompt(CLOSES, day, HOLD)
        dates = [l.split(",")[0] for l in p.splitlines() if l[:4].isdigit() and l[4] == "-"]
        assert max(dates) < str(day.date())
        prev = CLOSES.index[CLOSES.index < day][-1]
        assert max(dates) == str(prev.date())


def test_dropping_the_future_changes_nothing():
    day = DAYS[10]
    assert core.build_prompt(CLOSES, day, HOLD) == core.build_prompt(
        CLOSES.loc[CLOSES.index < day], day, HOLD)


def test_model_request_holds_only_fixed_system_and_one_prompt():
    day = DAYS[3]
    user = core.build_prompt(CLOSES, day, HOLD)
    args = run_bot.request_args(user)
    assert args["system"] == core.SYSTEM_PROMPT
    assert args["messages"] == [{"role": "user", "content": user}]
    assert "tools" not in args            # no web search, no news
    blob = json.dumps(args)
    # Day t's own date appears only in the "Today is" line, never as a price row.
    assert f"\\n{day.date()}," not in blob
    later = [str(d.date()) for d in CLOSES.index if d > day]
    assert not any(d in blob for d in later)


def test_trade_uses_open_of_day_not_a_later_price():
    # Lag check on the account: with a flat open-to-close day, the day's value
    # must depend only on prices up to the open of that day plus that day's close.
    idx = pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-06"])
    o = pd.DataFrame({t: [100.0, 110.0, 120.0] for t in STOCKS + ["SPY"]}, index=idx)
    c = pd.DataFrame({t: [100.0, 110.0, 120.0] for t in STOCKS + ["SPY"]}, index=idx)
    a = core.Account()
    target = {**{t: 0.1 for t in STOCKS}, "CASH": 0.0}
    a.step(idx[0], target, o, c)
    v0 = a.value
    # Changing day 3 prices must not change the value at day 1 or day 2.
    a2 = core.Account()
    o2, c2 = o.copy(), c.copy()
    o2.iloc[2] = 999.0
    c2.iloc[2] = 999.0
    a2.step(idx[0], target, o2, c2)
    a.step(idx[1], target, o, c)
    a2.step(idx[1], target, o2, c2)
    assert a.value == a2.value
    assert abs(v0 - (1 - 5e-4)) < 1e-12    # day 1: buy-in cost only
    assert abs(a.value - v0 * 1.1) < 1e-12  # 10% overnight gap, no trade (inside band)
