"""Live run: one Claude call per trading day, sequential (the prompt shows current holdings).

Usage:
  python run_bot.py --run 1            # live run, writes logs/run1.jsonl and out/run1_curve.csv
  python run_bot.py --run 1 --estimate # count tokens for day 1 and project the cost, no model call
  python run_bot.py --run 1 --replay   # free: reuse the model replies saved in logs/run1.jsonl
Replies are cached by (run, date, prompt hash), so a crash never pays twice.
A replay makes no API call and needs no API key. It writes to out/replay/.
"""
import argparse
import json
import time

import anthropic
import pandas as pd

import core
from config import (CACHE_DIR, EFFORT, LOG_DIR, MODEL, OUT_DIR, PRICE_IN,
                    PRICE_OUT, SPEND_CAP_USD, STOCKS)

MAX_TOKENS = 16000


def request_args(user):
    return dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=core.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": EFFORT,
                       "format": {"type": "json_schema", "schema": core.output_schema()}},
    )


def call_model(client, user):
    """The model gets the fixed system prompt and this one user string. Nothing else."""
    for attempt in range(5):
        try:
            return client.messages.create(**request_args(user))
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.InternalServerError) as e:
            time.sleep(10 * (attempt + 1))
            last = e
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--replay", action="store_true",
                    help="reuse the replies in logs/run{n}.jsonl; no API call, no key")
    ap.add_argument("--cap", type=float, default=SPEND_CAP_USD,
                    help="stop this run if its spend passes this many USD")
    a = ap.parse_args()

    opens, closes = core.load_prices()
    days = core.window_days(closes)
    acct = core.Account()

    if a.replay:
        replay(a.run, days, opens, closes, acct)
        return

    client = anthropic.Anthropic()

    if a.estimate:
        user = core.build_prompt(closes, days[0], acct.w)
        n = client.messages.count_tokens(
            model=MODEL, system=core.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": core.output_schema()}},
        ).input_tokens
        print(f"days={len(days)} {days[0].date()}..{days[-1].date()} input_tokens/day={n}")
        for out in (1000, 3000, 6000):
            usd = len(days) * (n * PRICE_IN + out * PRICE_OUT)
            print(f"  if {out} output tokens/day: ${usd:.2f} per run")
        return

    cache = CACHE_DIR / f"run{a.run}"
    cache.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run{a.run}.jsonl"
    log = open(log_path, "w")
    tok_in = tok_out = 0
    refusals = 0
    for day in days:
        user = core.build_prompt(closes, day, acct.w)
        h = core.prompt_hash(core.SYSTEM_PROMPT, user)
        f = cache / f"{day.date()}_{h}.json"
        if f.exists():
            rec = json.loads(f.read_text())
        else:
            r = call_model(client, user)
            assert r.model == MODEL, f"served by {r.model}, expected {MODEL}"
            text = next((b.text for b in r.content if b.type == "text"), None)
            notes = " ".join(b.thinking for b in r.content if b.type == "thinking")
            rec = {
                "date": str(day.date()), "prompt_hash": h, "model": r.model,
                "stop_reason": r.stop_reason, "reply_text": text, "thinking_summary": notes,
                "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
            }
            f.write_text(json.dumps(rec))
        tok_in += rec["input_tokens"]
        tok_out += rec["output_tokens"]
        if rec["stop_reason"] == "end_turn" and rec["reply_text"]:
            target = core.parse_weights(json.loads(rec["reply_text"]))
        else:
            refusals += 1          # keep the current weights for this day
            target = dict(acct.w)
        acct.step(day, target, opens, closes)
        log.write(json.dumps({**rec, "system": core.SYSTEM_PROMPT, "user": user,
                              "target": target, "value_close": acct.value}) + "\n")
        log.flush()
        spend = tok_in * PRICE_IN + tok_out * PRICE_OUT
        print(f"{day.date()} value={acct.value:.4f} spend=${spend:.2f}", flush=True)
        if spend > a.cap:
            raise SystemExit(f"spend cap ${a.cap} passed on {day.date()}; stopping")
    log.close()
    write_outputs(OUT_DIR, a.run, days, acct, refusals, tok_in, tok_out)


def replay(run, days, opens, closes, acct):
    """Same day loop as the live run, but each reply comes from logs/run{n}.jsonl.
    The prompt is rebuilt each day and its hash is checked against the logged one."""
    logged = {}
    with open(LOG_DIR / f"run{run}.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            logged[rec["date"]] = rec
    days = [d for d in days if d <= pd.Timestamp(max(logged))]
    if [str(d.date()) for d in days] != sorted(logged):
        raise SystemExit("trading days in data/raw do not match the dates in the log")
    tok_in = tok_out = 0
    refusals = mismatches = 0
    for day in days:
        user = core.build_prompt(closes, day, acct.w)
        h = core.prompt_hash(core.SYSTEM_PROMPT, user)
        rec = logged[str(day.date())]
        if h != rec["prompt_hash"]:
            mismatches += 1        # prices differ from the original download
        tok_in += rec["input_tokens"]
        tok_out += rec["output_tokens"]
        if rec["stop_reason"] == "end_turn" and rec["reply_text"]:
            target = core.parse_weights(json.loads(rec["reply_text"]))
        else:
            refusals += 1
            target = dict(acct.w)
        acct.step(day, target, opens, closes)
        print(f"{day.date()} value={acct.value:.4f}", flush=True)
    if mismatches:
        print(f"WARNING: {mismatches} of {len(days)} prompts differ from the log. "
              "Your price data is not the original download, so the result can differ a little.")
    else:
        print(f"All {len(days)} prompts match the log byte for byte.")
    print("Replay: no API call made, $0 spent.")
    write_outputs(OUT_DIR / "replay", run, days, acct, refusals, tok_in, tok_out)


def write_outputs(out_dir, run, days, acct, refusals, tok_in, tok_out):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(acct.curve, columns=["date", "value"]).to_csv(
        out_dir / f"run{run}_curve.csv", index=False)
    summary = {
        "run": run, "model": MODEL, "days": len(days),
        "end_pct": round((acct.value - 1) * 100, 2),
        "worst_fall_pct": round(core.worst_fall([v for _, v in acct.curve]) * 100, 2),
        "total_cost_pct": round(acct.cost_total * 100, 3),
        "trade_count": acct.trades, "refusal_days": refusals,
        "input_tokens": tok_in, "output_tokens": tok_out,
        "cost_usd": round(tok_in * PRICE_IN + tok_out * PRICE_OUT, 2),
    }
    (out_dir / f"run{run}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
