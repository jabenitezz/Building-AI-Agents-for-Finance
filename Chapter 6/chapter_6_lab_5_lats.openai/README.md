# LATS: Language Agent Tree Search — OpenAI port

This directory is a **platform port** of `chapter_6_lab_5_lats` from the Claude Agent SDK to OpenAI.

The goal is to preserve the original exercise as closely as possible:

- same LATS search algorithm;
- same `Node`, UCT, selection, expansion, evaluation, backpropagation, reflection, and final-tree functions;
- same simulated portfolio environment;
- same one-shot `k` expansion;
- same parsers for proposals, scores, and lessons;
- same search defaults: 3 iterations, `k_expand=2`, `max_depth=3`;
- same evaluator and reflector skill instructions;
- same console behavior and tree output.

Only the model/platform integration is changed.

## Platform mapping

Original lab:

```text
Claude Agent SDK
claude-sonnet-4-6
Claude project Skill tool
.claude/skills/...
```

OpenAI port:

```text
OpenAI Python SDK
gpt-4.1-mini
OpenAI Responses API
.openai/skills/...
```

The original Claude implementation invokes project skills through the Claude `Skill` tool. The OpenAI version preserves the two skill files and loads their instructions locally into the corresponding OpenAI request:

- `trade-evaluator` for trajectory scoring;
- `trade-reflector` for failure reflection.

This changes the provider mechanism, not the role those skills play in the algorithm.

## How LATS works

Each iteration still performs:

1. **Selection** — descend from the root via UCT.
2. **Expansion** — ask the model for `k` materially distinct trim candidates in one call.
3. **Evaluation** — score every new child with the trade-evaluator rubric.
4. **Success check** — if a new child reaches the 30% target, return it.
5. **Backpropagation** — on failure, propagate each child score back through its ancestors.
6. **Reflection** — reflect on the best failed child and save a `When ..., do ...` lesson for future expansions.

Conceptually:

```text
root (40%)
   |
   +-- candidate 1
   |
   +-- candidate 2
          |
          v
      evaluate
          |
          v
     backpropagate
          |
          v
       reflect
          |
          v
lesson added to the next expansion prompt
```

UCT continues to balance exploitation and exploration:

```text
UCT(s) = V(s) + w * sqrt(ln(N(parent)) / N(s))
```

with `w = 1.41`.

## Project structure

```text
chapter_6_lab_5_lats.openai/
├── lats_search.py
├── lats_helpers.py
├── requirements.txt
├── .env.example
├── .gitignore
└── .openai/
    └── skills/
        ├── trade-evaluator/
        │   └── SKILL.md
        └── trade-reflector/
            └── SKILL.md
```

## Requirements

- Python 3.10+
- OpenAI API key

Unlike the Claude Agent SDK version, this port does not require Node.js or the Claude Code CLI.

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
python lats_search.py
```

The default invocation remains:

```python
anyio.run(lats_search, 3, 2, 3)
```

which means:

```text
n_iterations = 3
k_expand     = 2
max_depth    = 3
```

## Google Colab

No Node.js installation is needed.

```python
!pip install -q openai anyio python-dotenv
```

Stage the complete directory, including `.openai/skills/`, and set the API key in the environment.

Because Colab already runs an event loop, import and call:

```python
await lats_search(3, 2, 3)
```

instead of calling `anyio.run(...)` from a notebook cell.

## Important implementation detail

The helper function `_run_query(...)` keeps the same broad role and compatible arguments as the Claude implementation.

In the original:

```text
use_skills=True
    ↓
Claude Agent SDK
    ↓
Skill tool
    ↓
.claude/skills/<skill>/SKILL.md
```

In this port:

```text
use_skills=True
    ↓
lats_helpers.py detects the requested skill
    ↓
loads .openai/skills/<skill>/SKILL.md
    ↓
injects those instructions into the OpenAI request
```

The evaluator and reflector therefore use the same rubrics and output contracts as the original exercise.

## Model

The default model is deliberately the same one used by the other OpenAI Chapter 6 ports:

```text
gpt-4.1-mini
```

It can be overridden through `OPENAI_MODEL`, but the lab defaults to `gpt-4.1-mini`.
