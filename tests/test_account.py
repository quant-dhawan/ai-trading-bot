import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402
from config import STOCKS  # noqa: E402


def test_parse_weights_no_shorts_no_leverage():
    w = core.parse_weights({**{t: 0.3 for t in STOCKS}, "NVDA": -0.5, "CASH": 0.0})
    assert all(w[t] >= 0 for t in STOCKS)
    assert abs(sum(w.values()) - 1) < 1e-12
    assert w["NVDA"] == 0


def test_worst_fall():
    assert abs(core.worst_fall([1.1, 0.99, 1.2]) - 0.1) < 1e-12
    assert core.worst_fall([1.0, 1.1, 1.2]) == 0.0
    assert abs(core.worst_fall([0.9]) - 0.1) < 1e-12   # start value counts as a peak


def test_cost_is_5bps_per_side():
    idx = pd.to_datetime(["2026-07-01", "2026-07-02"])
    px = pd.DataFrame({t: [50.0, 50.0] for t in STOCKS + ["SPY"]}, index=idx)
    a = core.Account()
    a.step(idx[0], {**{t: 0.1 for t in STOCKS}, "CASH": 0.0}, px, px)
    assert abs(a.cost_total - 5e-4) < 1e-12 and a.trades == 10
    a.step(idx[1], {**{t: 0.0 for t in STOCKS}, "CASH": 1.0}, px, px)
    assert abs(a.cost_total - (5e-4 + (1 - 5e-4) * 5e-4)) < 1e-12 and a.trades == 20
