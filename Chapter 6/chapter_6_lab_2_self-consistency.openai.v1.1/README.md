# Self-Consistency v1.1 — OpenAI + Financial Datasets

Enhanced version of Chapter 6 Lab 2.

This version keeps the **Self-Consistency** pattern but enriches the financial input so the independent paths have enough evidence to disagree meaningfully instead of defaulting to `HOLD` because growth and cash-flow information are missing.

```text
Financial Datasets
        |
        +-- Income statement
        +-- Price snapshot
        +-- Financial metrics snapshot
               |
               +-- revenue growth
               +-- earnings growth
               +-- EPS growth
               +-- FCF growth
               +-- operating income growth
               +-- EBITDA growth
               +-- debt/equity
               +-- current ratio
               +-- FCF/share
               +-- market cap
        |
        v
  same enriched snapshot
        |
  +-----+-----+-----+
  |     |     |     |
Path1 Path2 Path3 ... PathN
        |
        v
  plurality vote
```

The model is `gpt-4.1-mini` by default and uses the OpenAI Responses API.

The lab prints short user-facing rationales only. It does not request or expose hidden chain-of-thought.

> Educational use only. The aggregate label is a model output, not personalized investment advice.

## What changed from v1.0

The first OpenAI version requested Financial Datasets' metrics snapshot but used almost only `market_cap`. That made growth unavailable whenever the API returned only one comparable annual income statement.

v1.1 now uses the growth and financial-strength fields already returned by `/financial-metrics/snapshot`:

- `revenue_growth`
- `earnings_growth`
- `earnings_per_share_growth`
- `free_cash_flow_growth`
- `operating_income_growth`
- `ebitda_growth`
- `debt_to_equity`
- `current_ratio`
- `free_cash_flow_per_share`

When two annual income statements are available, revenue growth is calculated directly from them. Otherwise, the metrics-snapshot value is used as a fallback and its source is shown explicitly.

v1.1 also computes FCF yield from FCF/share and the current price when both values are available.

The recommendation parser is more robust for Spanish output and accepts both `RECOMMENDATION:` and `RECOMENDACIÓN:`, while the prompt still asks the model to keep the final machine-readable line in English.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your own keys. Do not commit the file.

## Run

```bash
python self-consistency.py --ticker AAPL --language es
python self-consistency.py --ticker MSFT --language es
python self-consistency.py --ticker NVDA --language es
```

Seven paths:

```bash
python self-consistency.py --ticker AMD --n-paths 7 --language es
```

## Self-Consistency behavior

Each path receives the exact same enriched financial snapshot but is sampled independently.

Missing information should reduce confidence but should not automatically force `HOLD`. Each path must weigh the evidence that is actually available.

After all paths finish, the code extracts `BUY`, `HOLD`, or `SELL` and performs a plurality vote. Exact top-vote ties return `NO_CONSENSUS`.

## Output

Reports are saved under:

```text
output/<TICKER>_self_consistency_<timestamp>.md
```
