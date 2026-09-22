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
    APPROVED / REJECTED
           |
    qualitative macro-risk gate
           |
        execution

The adversarial layer is a validator, not a second Portfolio Manager. It never
changes SIZE_PCT. Judge APPROVED keeps the exact PM target; Judge REJECTED turns
the effective target into 0%.

Bull and Bear use the same model tier, reasoning effort and token budget:
make_opus_equivalent(max_tokens=1500). Their only intended difference is the
role prompt. Both receive the four original reports plus the PM narrative with
the ACTION/CONFIDENCE/SIZE_PCT control line removed.

The debate only runs for BUY proposals that pass hard risk. HOLD and SELL
already imply 0% long exposure, so debating them cannot change the long-only
portfolio and would only add cost.

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
for the final signal. It is a Judge-filter diagnostic, not portfolio attribution.

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
