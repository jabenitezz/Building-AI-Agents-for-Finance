# Self-Refine — OpenAI + Financial Datasets

OpenAI version of Chapter 6 Lab 1.

This lab implements the **Self-Refine / evaluator-optimizer** pattern for equity research:

```text
Financial Datasets
        ↓
  initial thesis
        ↓
     evaluator
        ↓
   refine thesis
        ↓
 evaluator again
        ↓
 stop or repeat
```

Unlike the original lab, the financial metrics are **not hard-coded**. The script fetches the latest available annual company metrics from the Financial Datasets API and then uses the OpenAI Responses API to generate, critique, and refine the investment thesis.

The default model is `gpt-4.1-mini` to keep the lab inexpensive.

> Educational use only. This lab does not provide investment advice.

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
OPENAI_API_KEY=sk-...
FINANCIAL_DATASETS_API_KEY=...
```

## Run

```bash
python self-refine.py --ticker AAPL
python self-refine.py --ticker TSLA
python self-refine.py --ticker NVDA --max-iterations 2
```

## Data used

`financial_data.py` requests:

1. Latest annual income statements, up to 2 periods.
2. Current price snapshot.
3. Current financial-metrics snapshot.

It supplies report period, revenue, YoY revenue growth when available, net income, diluted EPS, gross margin, operating margin, current price, P/E and market capitalization.

If only one annual period is returned, revenue growth is marked unavailable rather than invented.

## OpenAI calls

The loop uses 1 call for the initial thesis, then up to 2 calls per iteration: evaluator + refiner. With 3 iterations the maximum is 7 OpenAI calls.

The prompt history grows with each iteration. This is deliberate for the lab so the Self-Refine pattern is visible.

## Output

The script prints the full process and saves a Markdown report under `output/`.

## Files

- `self-refine.py` — OpenAI Self-Refine loop and CLI.
- `financial_data.py` — Financial Datasets integration.
- `.env.example` — environment template.
- `.gitignore` — excludes secrets, venvs, caches and generated output.
- `requirements.txt` — dependencies.
