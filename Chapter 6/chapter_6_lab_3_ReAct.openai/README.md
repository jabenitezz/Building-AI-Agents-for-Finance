# ReAct Lab — OpenAI version

OpenAI adaptation of `chapter_6_lab_3_ReAct`.

The purpose and financial tools are intentionally kept the same as the original lab. The main change is the model/provider layer:

- Claude Agent SDK → OpenAI Python SDK
- Claude Sonnet → `gpt-4.1-mini`
- Claude MCP tool wrappers → OpenAI Responses API function tools

The financial tools themselves still use live Yahoo Finance data through `yfinance`.

## ReAct flow

This version preserves the pedagogical trace of the original lab:

```text
Question
   ↓
gpt-4.1-mini
   ↓
[THOUGHT] short public rationale
   ↓
[ACTION] choose financial tool
   ↓
Yahoo Finance tool executes
   ↓
[OBSERVATION] tool result
   ↓
gpt-4.1-mini evaluates the new context
   ↓
[THOUGHT] next public rationale
   ↓
[ACTION] next tool if needed
   ↓
...
   ↓
final answer in Spanish
```

The `[THOUGHT]` line is deliberately a **short user-facing rationale** explaining what information is needed next or how the latest observation affects the next action. It is not private chain-of-thought.

To make the trace reliable with OpenAI function calling, every financial tool schema includes a required display-only `thought` argument. The program prints it before the action and removes it before calling the underlying Python function. Therefore the actual financial tool behavior is unchanged.

Example:

```text
[THOUGHT] Primero necesito conocer los múltiplos de NVIDIA y confirmar su sector.
[ACTION] get_valuation_ratios({'ticker': 'NVDA'})
[OBSERVATION] {"PE": ..., "sector": "Technology"}

[THOUGHT] Con el sector ya identificado, necesito comparar esos múltiplos con su mediana.
[ACTION] get_sector_median_pe({'sector': 'Technology'})
[OBSERVATION] {...}
```

## Tools preserved from the original lab

The same three financial tools and the same underlying logic are retained:

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

If you do not pass a question, the script uses this default example in Spanish:

```text
¿Está Apple (AAPL) sobrevalorada en relación con el sector tecnológico?
¿Qué implica esto para una cartera long-only?
```

Run it with:

```bash
python react.py
```

You can also pass any other question as a positional parameter. Put it in quotes when it contains spaces:

```bash
python react.py "¿Está Microsoft (MSFT) sobrevalorada en relación con el sector tecnológico?"
```

Another example:

```bash
python react.py "Analiza la valoración de NVIDIA (NVDA) frente a su sector y explica qué implica para una cartera long-only."
```

The script prints the selected question first and the final answer is always requested in Spanish (Spain).

## Model

The default model is:

```text
gpt-4.1-mini
```

You can override it in `.env` without changing the source.

## Notes

The implementation uses the OpenAI Responses API with custom function tools. Because `store=False` is used, the script carries forward the response output and tool observations explicitly in the next request.

Yahoo Finance values are live and can change between executions. Sector medians are based only on the curated peer lists in `SECTOR_TICKERS`, exactly as in the original lab.
