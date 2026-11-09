"""Charts carry no banned word, and render at 2560x1440. Uses toy curves, no API."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_results as mr  # noqa: E402


def toy():
    idx = pd.bdate_range("2026-07-01", periods=40)
    rng = np.random.default_rng(0)
    bot = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 40)), index=idx)
    spy = pd.Series(np.cumprod(1 + rng.normal(0, 0.008, 40)), index=idx)
    return bot, spy, idx


def test_no_banned_words_and_size():
    bot, spy, idx = toy()
    for fig in (mr.chart_value({1: bot, 2: bot * 1.01}, spy, idx[0], idx[-1]),
                mr.chart_fall(bot, spy, idx[0], idx[-1])):
        assert mr.banned_hits(fig) == []
        w, h = fig.get_size_inches() * fig.dpi
        assert (round(w), round(h)) == (2560, 1440)


def test_banned_check_catches_a_word():
    bot, spy, idx = toy()
    fig = mr.chart_fall(bot, spy, idx[0], idx[-1])
    fig.text(0.5, 0.5, "Max Drawdown")
    assert mr.banned_hits(fig)
