# Chapter 8 — OpenAI-only v1

This directory is a complete OpenAI-only translation of the Chapter 8 exercise.

The original Chapter 8 mixes Anthropic and OpenAI models. This version preserves
the same LangGraph architecture, data fetchers, prompts, risk rules, debate
structure, and backtest harness, while replacing every Anthropic model with an
OpenAI model chosen for the closest **role/capability tier**.

## Model mapping

| Original role | Original model | OpenAI-only v1 | Why |
| --- | --- | --- | --- |
| Portfolio Manager / Risk / Bull | Claude Opus 4.7 | `gpt-5.6-sol` | Premium reasoning / complex professional work |
| Fundamental / Technical / Macro / Devil / Judge | Claude Sonnet 4.6 | `gpt-5.6-terra` | Balance of intelligence and cost |
| Sentiment | Claude Haiku 4.5 | `gpt-5.6-luna` | Cost-sensitive, high-volume workload |

The mapping is intentionally by **tier and role**, not by claiming exact
cross-vendor benchmark equivalence.

### Important difference in the adversarial debate

The original debate deliberately used two providers (Anthropic + OpenAI) for
Bull and Bear to reduce correlated model errors.

An OpenAI-only version cannot preserve provider diversity. Here:

- Bull uses GPT-5.6 Sol.
- Bear uses GPT-5.6 Terra.
- Devil's Advocate uses GPT-5.6 Terra.
- Judge uses GPT-5.6 Terra.

So the debate remains adversarial by prompt and uses different capability tiers,
but may have more correlated errors than the original cross-provider design.

## Project layout

```text
openai.v1/
├── models.py
├── investment_committee.py
├── adversarial_debate.py
├── backtest.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+
- One OpenAI API key
- No Anthropic key is required

Install:

```bash
cd "Chapter 8/openai.v1"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already have a compatible Chapter 8 virtual environment, installing the
requirements into that environment is enough.

## Configure

```bash
cp .env.example .env
```

Put your own key in `.env`:

```env
OPENAI_API_KEY=...
```

Do not commit `.env`.

Optional model overrides:

```env
OPENAI_OPUS_EQUIVALENT_MODEL=gpt-5.6-sol
OPENAI_SONNET_EQUIVALENT_MODEL=gpt-5.6-terra
OPENAI_HAIKU_EQUIVALENT_MODEL=gpt-5.6-luna
```

## Run

From inside `Chapter 8/openai.v1`:

```bash
python investment_committee.py
python adversarial_debate.py
python backtest.py
```

The committee and debate keep the original default ticker, TSLA.

## Architecture

Stage 1:

```text
Fundamentals ─┐
Technicals   ─┼──> Portfolio Manager (GPT-5.6 Sol)
Sentiment    ─┤             |
Macro        ─┘             v
                       Risk Officer (GPT-5.6 Sol)
```

Specialists:
- Fundamentals: GPT-5.6 Terra
- Technicals: GPT-5.6 Terra
- Sentiment: GPT-5.6 Luna
- Macro: GPT-5.6 Terra

Stage 2:

```text
Committee thesis
      |
      +--> Bull (GPT-5.6 Sol) ----┐
      |                           |
      +--> Bear (GPT-5.6 Terra) --+--> Devil (Terra) --> Judge (Terra)
```

## API behavior

The GPT-5.6 models are instantiated with LangChain's `ChatOpenAI` using the
OpenAI Responses API.

Reasoning effort is tiered as well:

- Sol: `high`
- Terra: `medium`
- Luna: `low`

This mirrors the chapter's cost/capability idea: spend more reasoning budget on
the PM/risk decisions, less on high-volume specialist work.

## Backtest warning

The original warning still applies. The included backtest is a framework
demonstration, not a research-grade point-in-time backtest. The committee
fetchers use current fundamentals/news/macro data, so historical runs can leak
future information.
