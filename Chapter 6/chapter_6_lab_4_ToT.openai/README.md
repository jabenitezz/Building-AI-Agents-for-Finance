# Tree of Thoughts — Hedge Selection Lab (OpenAI)

OpenAI port of `chapter_6_lab_4_ToT`.

The exercise behavior is intentionally kept the same. Only the model/provider layer is changed:

- Anthropic Python SDK → OpenAI Python SDK
- `claude-sonnet-4-6` → `gpt-4.1-mini`
- Anthropic structured output → OpenAI Responses API Structured Outputs

The prompts, JSON schemas, tree structure, branching factors, beam search, scoring criteria, console output, default portfolio, and default stress scenario are preserved.

## Tree of Thoughts behavior

The program explores hedge candidates as a two-level tree and uses the LLM as evaluator:

```text
Portfolio + stress scenario
          |
          v
Level 1: generate 4 hedge candidates
          |
          v
score every candidate (1-10)
          |
          v
sort by score
          |
          v
retain top 2 (beam_width=2)
          |
          +------------------+
          |                  |
          v                  v
Level 2: 3 sizings      Level 2: 3 sizings
for retained hedge 1    for retained hedge 2
          |                  |
          v                  v
score all sizings       score all sizings
          |                  |
          v                  v
keep best sizing        keep best sizing
          \__________________/
                   |
                   v
            final retained hedges
```

As in the original lab:

- `generate_hedges(..., n=4)` creates four Level-1 branches.
- `score_hedge(...)` scores protection, cost, and basis risk.
- Beam search keeps the best `beam_width=2` branches.
- `generate_sizings(..., n=3)` creates three Level-2 branches for each retained hedge.
- `score_sizing(...)` scores residual catalyst exposure, premium cost, and complementarity with the parent hedge.
- The highest-scoring sizing is attached to each retained hedge.

## Structured outputs

The original JSON schemas are preserved:

- `HEDGES_SCHEMA` for candidate lists.
- `SCORE_SCHEMA` for score + reasoning.

The OpenAI version uses the Responses API with `text.format.type = "json_schema"` and strict schema adherence, so the rest of the Python logic can continue to parse the model output with `json.loads(...)` just like the original exercise.

## Requirements

- Python 3.9+
- OpenAI API key

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
python tot.py
```

The default exercise remains:

```text
Portfolio:
5,000 shares of NVDA held outright (~$5M concentrated single-name position)

Stress scenario:
NVDA reports earnings tonight after the close; weekly implied volatility is elevated
```

The script prints the same stages as the original:

```text
CANDIDATES
SCORED
RETAINED
SIZING
SCORED SIZING
```

and finishes with one summary line for each retained hedge plus its best sizing.

## Files

- `tot.py` — OpenAI port of the original ToT exercise.
- `requirements.txt` — OpenAI and dotenv dependencies.
- `.env.example` — environment template.
- `.gitignore` — excludes secrets, virtual environments, caches, and editor files.

The original `tot-example.md` is a captured Anthropic run, so it is left in the original lab rather than copied as if it were OpenAI output.
