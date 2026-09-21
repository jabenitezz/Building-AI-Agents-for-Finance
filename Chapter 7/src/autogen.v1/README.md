# AutoGen v1 — robust yfinance recovery

This is a sibling of `src/autogen/` that preserves the Chapter 7
**conversational multi-agent** architecture:

```text
code_executor -> assistant -> generated Python -> code_executor
      ^                                              |
      +----------------------------------------------+
```

The original demo can fail when generated code guesses a yfinance financial
statement row that is not present (for example `Total Stockholder Equity`).
It may then repeat the same failing lookup and exhaust `max_turns`.

## What v1 changes

- Keeps the same `AssistantAgent` + `UserProxyAgent` + local code executor.
- Adds `financial_helpers.py`, copied into the temporary execution directory.
- The assistant is instructed to use `get_health_snapshot()` first for
  standard financial-health metrics.
- Statement fallbacks accept multiple yfinance row-label variants.
- Derived statement ratios use the **latest non-null period only**; they never
  sum balance-sheet values across years.
- If generated code fails, the assistant must use the traceback to change the
  next attempt rather than repeating the same failing code.
- `max_turns` increases from 5 to 10 by default.
- `TERMINATE` is allowed only after a successful execution result has been
  observed and the final analysis has been written.

## Run

Because this folder is deliberately named `autogen.v1` (the dot is part of
the directory name), run the file directly rather than with `python -m`:

```bash
cd "Chapter 7"
source .venv312/bin/activate
python "src/autogen.v1/run.py" "Assess the financial health of AAPL"
```

You can override the turn budget:

```bash
AUTOGEN_MAX_TURNS=12 python "src/autogen.v1/run.py" "Assess the financial health of AAPL"
```

The shared Chapter 7 requirements still apply. For the classic AutoGen code
path, use Python 3.12 (or another Python version supported by the installed
AutoGen 0.2-lineage package).
