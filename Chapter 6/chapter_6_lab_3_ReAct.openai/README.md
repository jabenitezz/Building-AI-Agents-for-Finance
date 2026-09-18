# ReAct Lab — OpenAI version

OpenAI adaptation of `chapter_6_lab_3_ReAct`.

The purpose and tools are intentionally kept the same as the original lab. The main change is the model/provider layer:

- Claude Agent SDK → OpenAI Python SDK
- Claude Sonnet → `gpt-4.1-mini`
- Claude MCP tool wrappers → OpenAI Responses API function tools

The financial tools themselves still use live Yahoo Finance data through `yfinance`.

## ReAct flow

```text
Question
   ↓
gpt-4.1-mini
   ↓
ACTION: choose financial tool
   ↓
Yahoo Finance tool executes
   ↓
OBSERVATION: tool result
   ↓
gpt-4.1-mini evaluates the observation
   ↓
another tool if needed
   ↓
final answer
```

The script prints each tool call as `[ACTION]` and each result as `[OBSERVATION]`. It may also print concise user-visible model text as `[MODEL]`. It does not expose private chain-of-thought.

## Tools preserved from the original lab

The same three tools and the same underlying logic are retained:

- `get_valuation_ratios` — P/E, EV/EBITDA, P/B, PEG and Yahoo Finance sector.
- `get_sector_median_pe` — median P/E and EV/EBITDA across the same curated large-cap sector peers.
- `get_balance_sheet` — net cash in USD billions and debt-to-equity.

The `SECTOR_TICKERS` peer lists are also unchanged.

## Requirements

- Python 3.10+
- OpenAI API key

No Node.js or Claude Code installation is required in this version.

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
OPENAI_MODEL=gpt-4.1-mini
```

## Run

```bash
python react.py
```

The question at the bottom of `react.py` is intentionally the same as in the original lab:

```text
Is Apple (AAPL) overvalued relative to the Technology sector?
What does this imply for a long-only portfolio?
```

## Model

The default model is:

```text
gpt-4.1-mini
```

You can override it in `.env` without changing the source:

```env
OPENAI_MODEL=gpt-4.1-mini
```

## Notes

The implementation uses the OpenAI Responses API with custom function tools. Because `store=False` is used, the script carries forward the response output and tool observations explicitly in the next request.

Yahoo Finance values are live and can change between executions. Sector medians are based only on the curated peer lists in `SECTOR_TICKERS`, exactly as in the original lab.
