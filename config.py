"""Fixed settings for the test. Change nothing here after the live run starts."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
OUT_DIR = HERE / "out"
LOG_DIR = HERE / "logs"
CACHE_DIR = Path.home() / ".cache" / "yt-ai-bot-build" / "replies"

# Ten large US stocks, picked by size. The pick uses today's knowledge (see README limits).
STOCKS = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM", "BRK-B"]
BENCH = "SPY"
ALL_TICKERS = STOCKS + [BENCH]

HISTORY_START = "2026-01-01"   # lookback data before the window
LOOKBACK_DAYS = 60             # closes shown to the model each day

# Model and cutoff. Source: platform.claude.com/docs/en/about-claude/models/overview
# (read 2026-11-01): Claude Opus 5.5, training data cutoff "Jun 2026".
MODEL = "claude-opus-5-5"
CUTOFF = "2026-06"
WINDOW_START = "2026-07-01"    # first trading day after the cutoff month
EFFORT = "medium"              # set explicitly; temperature is not accepted by the API

# Costs and rules (assumptions)
COST_BPS = 5.0                 # per side, on traded value
TRADE_BAND = 0.005             # skip a trade smaller than 0.5% of the account
PRICE_IN = 4.00 / 1e6          # USD per input token  (Opus 5.5, docs 2026-11-01)
PRICE_OUT = 20.00 / 1e6        # USD per output token (thinking tokens bill as output)
SPEND_CAP_USD = 25.0
