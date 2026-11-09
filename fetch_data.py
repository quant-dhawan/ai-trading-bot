"""Download free daily prices (Yahoo Finance via yfinance) and save the raw CSVs.

Prices are split- and dividend-adjusted (auto_adjust=True), so SPY's line
includes its dividends, the same as the stocks.
The end date is pinned so a new download covers the same days as the test.
Run: python fetch_data.py
"""
import datetime as dt
import json
from pathlib import Path

import yfinance as yf

from config import ALL_TICKERS, HISTORY_START, RAW_DIR

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    end = "2026-10-31"            # yfinance end is exclusive: last day is 30 Oct 2026
    for t in ALL_TICKERS:
        df = yf.download(t, start=HISTORY_START, end=end, auto_adjust=True,
                         progress=False, multi_level_index=False)
        if df.empty:
            raise SystemExit(f"no data for {t}")
        df.index.name = "Date"
        df[["Open", "High", "Low", "Close", "Volume"]].to_csv(RAW_DIR / f"{t}.csv")
        print(t, len(df), df.index.min().date(), df.index.max().date())
    meta = {
        "source": "Yahoo Finance via yfinance " + yf.__version__,
        "adjusted": "auto_adjust=True (splits and dividends)",
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tickers": ALL_TICKERS,
    }
    (RAW_DIR / "download_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))

if __name__ == "__main__":
    main()
