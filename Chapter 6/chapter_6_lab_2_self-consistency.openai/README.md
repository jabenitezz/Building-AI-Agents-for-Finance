# Self-Consistency — OpenAI + Financial Datasets

OpenAI version of Chapter 6 Lab 2.

This lab implements the **Self-Consistency** pattern by sampling several independent analyses of the same stock, extracting a final `BUY` / `HOLD` / `SELL` label from each one, and aggregating the labels with a plurality vote.

```text
Financial Datasets
        ↓
   same stock snapshot
        ↓
 ┌──────┼──────┐
 ↓      ↓      ↓
Path 1 Path 2 ... Path N
 ↓      ↓          ↓
BUY   HOLD        HOLD
 └──────┬───────────┘
        ↓
 plurality vote
        ↓
 aggregate label
```

Unlike the original lab, the stock summary is **not hard-coded**. The script fetches real company data from Financial Datasets and uses the OpenAI Responses API with `gpt-4.1-mini` by default.

The sample paths contain short user-facing rationales only; the lab does not request or expose hidden chain-of-thought.

> Educational use only. The aggregate label is a model output, not personalized investment advice.

## Requirements

- Python 3.10+
- OpenAI API key
- Financial Datasets API key

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
FINANCIAL_DATASETS_API_KEY=your_financial_datasets_key_here
```

## Run

Default run:

```bash
python self-consistency.py --ticker AAPL
```

Five independent paths in Spanish:

```bash
python self-consistency.py --ticker AAPL --language es
```

Seven paths:

```bash
python self-consistency.py --ticker NVDA --n-paths 7
```

Lower sampling diversity:

```bash
python self-consistency.py --ticker AMD --temperature 0.4
```

## How it works

Each path receives exactly the same Financial Datasets snapshot but is sampled independently. The prompt prohibits outside company knowledge and asks the model to treat missing information as uncertainty rather than inventing facts.

After all paths finish, the code parses the terminal line:

```text
RECOMMENDATION: BUY
```

and counts the labels. If two or more labels tie for the highest vote count, the final result is `NO_CONSENSUS` rather than selecting an arbitrary winner.

## Data used

`financial_data.py` requests:

1. Latest annual income statements, up to 2 periods.
2. Current price snapshot.
3. Current financial-metrics snapshot.

It supplies report period, revenue, YoY revenue growth when available, net income, diluted EPS, gross margin, operating margin, current price, P/E and market capitalization.

## Cost

With `N` paths, this lab makes `N` OpenAI calls. The default is 5.

## Output

The run is printed to the terminal and saved under:

```text
output/<TICKER>_self_consistency_<timestamp>.md
```

## Files

- `self-consistency.py` — OpenAI sampling and plurality-vote logic.
- `financial_data.py` — Financial Datasets integration.
- `.env.example` — environment template.
- `.gitignore` — excludes secrets, venvs, caches and generated reports.
- `requirements.txt` — dependencies.
