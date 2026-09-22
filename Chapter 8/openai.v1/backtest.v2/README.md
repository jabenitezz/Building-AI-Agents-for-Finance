# Chapter 8 / openai.v1 / backtest.v2

This version extends the point-in-time backtest with an adversarial validation
layer while preserving the Portfolio Manager sizing semantics already validated
in v1.

Architecture:

    point-in-time data
           |
    4 specialist analysts
           |
    Portfolio Manager
    ACTION + CONFIDENCE + SIZE_PCT
           |
    deterministic hard-risk precheck
           |
       BULL     BEAR
        \       /
     Devil's Advocate
           |
         Judge
      BUY / HOLD / SELL
           |
   derived validation
  APPROVED / REJECTED
           |
    qualitative macro-risk gate
           |
        execution

The adversarial layer is a validator, not a second Portfolio Manager. It never
changes SIZE_PCT. The Judge keeps the directional vocabulary from the book:

    VERDICT=BUY|HOLD|SELL
    CONFIDENCE=0-100

BUY validates the PM proposal and preserves the exact PM SIZE_PCT. HOLD and SELL
both reject the long entry and set the effective target to 0%; SELL never opens
a short. A derived validation_status is stored as APPROVED for Judge BUY and
REJECTED for Judge HOLD/SELL.

Bull and Bear use the same model tier, reasoning effort and token budget:
make_opus_equivalent(max_tokens=1500). Their only intended difference is the
role prompt. Both receive the four original reports plus the PM narrative with
the ACTION/CONFIDENCE/SIZE_PCT control line removed.

Their structured outputs include:

    BULL_CONVICTION=0-100
    BEAR_CONVICTION=0-100

These values are stored for audit and later analysis only. They do not resize
the PM position. The backtest also records conviction_gap =
bull_conviction - bear_conviction as a diagnostic. Conviction/confidence values
are self-reported model scores, not calibrated probabilities.

The debate only runs for BUY proposals that pass hard risk. HOLD and SELL
already imply 0% long exposure, so debating them cannot change the long-only
portfolio and would only add cost.

## Point-in-time fundamentals

v2 reconstructs fresh fundamental snapshots from SEC Company Facts without a
paid market-data API. The absolute rule remains:

    filed <= as_of_date

Flow metrics (revenue and net income) are reconstructed TTM. At fiscal year-end
the 10-K annual value is TTM; after Q1/Q2/Q3 the calculation is:

    TTM = latest annual 10-K + current YTD 10-Q - prior-year comparable YTD 10-Q

Balance-sheet metrics use the latest 10-Q/10-K period available at the decision
date. P/E is reconstructed as decision-date market cap / TTM net income rather
than combining the historical price with stale annual EPS.

The signal audit stores the SEC forms, filing dates and periods used in the TTM
components. To inspect fundamentals without spending any LLM calls:

    python test_fundamentals.py --ticker NVDA --as-of 2026-01-30

Hard risk is deterministic:
- annualized 30d volatility > 60% -> reject
- SIZE_PCT > 3.0% -> reject
- CONFIDENCE < 50 -> reject

The final qualitative risk gate preserves the original macro-headwind check.
It does not see the debate, so the committee-only baseline and adversarial path
remain directly comparable.

signals.csv stores both paths:
- committee_action / committee_size_pct = committee-only baseline
- action / size_pct = final adversarial strategy

run_backtest.py compares:
- MAS Committee Only
- MAS + Adversarial
- SPY Buy & Hold
- Universe Buy & Hold
- Equal Weight Rebalanced

judge_effect.csv records the next-period underlying return for each debated BUY
from its execution open to the next rebalance open, or to evaluation-end close
for the final signal. It includes Bull/Bear conviction, conviction_gap,
Judge BUY/HOLD/SELL, Judge confidence and validation_status. It is a diagnostic
of the debate layer, not portfolio attribution.

Install from Chapter 8/openai.v1:

    uv pip install -r "backtest.v2/requirements.txt"

Required environment:

    OPENAI_API_KEY=...
    SEC_EDGAR_EMAIL=...
    ALPHAVANTAGE_API_KEY=...

Recommended execution order from backtest.v2:

    python test_news.py
    python generate_signals.py --plan-only

    python generate_signals.py \
      --tickers TSLA MSFT NVDA \
      --start 2026-01-01 \
      --end 2026-04-01 \
      --freq monthly \
      --output output/signals.csv \
      --overwrite

    python run_backtest.py

LLM cost:
- base: 5 calls per ticker/date (4 analysts + PM)
- hard-risk-passing BUY: +5 calls (Bull, Bear, Devil, Judge, qualitative risk)
- maximum: 10 calls per ticker/date

Outputs:
    output/signals.csv
    output/backtest/summary.csv
    output/backtest/equity_curve.csv
    output/backtest/committee_orders.csv
    output/backtest/committee_trades.csv
    output/backtest/adversarial_orders.csv
    output/backtest/adversarial_trades.csv
    output/backtest/mas_orders.csv
    output/backtest/mas_trades.csv
    output/backtest/signals_used.csv
    output/backtest/judge_effect.csv

Portfolio semantics remain unchanged:

    SIZE_PCT = target position as % of TOTAL PORTFOLIO NAV

    BUY  -> rebalance to SIZE_PCT
    HOLD -> target 0% long
    SELL -> target 0% long, never short

Therefore BUY 1.5% followed by BUY 1.0% means rebalance from 1.5% to 1.0%, not
accumulate to 2.5%.
