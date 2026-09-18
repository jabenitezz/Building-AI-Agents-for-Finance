"""Chapter 6 Lab 1 — Self-Refine with OpenAI + Financial Datasets.

Pattern:
    financial data -> initial thesis -> evaluator -> refiner -> evaluator ...

The LLM is deliberately grounded only in the supplied Financial Datasets
snapshot. Missing information must be called out rather than invented.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from financial_data import get_financial_data


load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_MAX_ITERATIONS = 3

client = OpenAI()


def call_llm(prompt: str, model: str, max_tokens: int) -> str:
    """Send one grounded text request through the OpenAI Responses API."""
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_tokens,
        store=False,
    )
    return response.output_text.strip()


def generate_initial_thesis(financials: str, model: str) -> str:
    prompt = f"""You are a senior equity analyst at a long-only asset management firm.

Draft a concise investment thesis based ONLY on the financial data below.

Financial data:
{financials}

Cover these sections:
1. Key business drivers visible in the supplied metrics.
2. Valuation using only the supplied valuation/price metrics.
3. Two or three key risks that are supported by the supplied data, including
   important information gaps where the data is insufficient.

Ground every factual statement in the supplied data.
Do not use outside knowledge about the company.
If a claim cannot be supported by the supplied data, label it [unverified].
Do not invent missing metrics.
Write 1-2 sentences per section.
Be concise."""

    return call_llm(prompt, model=model, max_tokens=1024)


def get_feedback(
    thesis: str,
    financials: str,
    history_feedback: str,
    model: str,
) -> str:
    prompt = f"""You are a critical investment committee reviewer.

Review the investment thesis below and provide specific, actionable feedback.
Identify missing information, weak arguments, unsupported claims, misuse of
financial metrics, or conclusions that are stronger than the supplied evidence.

Use ONLY the supplied financial data as ground truth.
Do not add outside knowledge about the company.
If the available data is insufficient for a conclusion, require the analyst to
state that limitation rather than invent information.

The thesis should cover:
- key business drivers
- valuation
- 2-3 key risks or material information gaps

If no issue would block investment-committee review, respond with exactly:
NO_FURTHER_FEEDBACK

Be concise.

Financial data:
{financials}

Previous theses and feedback:
{history_feedback or "[none]"}

Thesis to review:
{thesis}

Feedback:"""

    return call_llm(prompt, model=model, max_tokens=1024)


def refine_thesis(
    thesis: str,
    feedback: str,
    financials: str,
    history_feedback: str,
    model: str,
) -> str:
    prompt = f"""You are a senior equity analyst.

Revise the thesis using the reviewer feedback below.

Use ONLY the supplied financial data as ground truth.
Do not add outside knowledge about the company.
Do not invent missing metrics.
If the supplied data cannot support a conclusion, state the limitation clearly.

Keep the same sections:
1. Key business drivers.
2. Valuation.
3. Two or three key risks or material information gaps.

Directly address every material reviewer comment.
Write 1-2 sentences per section.
Be concise.

Financial data:
{financials}

Previous theses and feedback:
{history_feedback or "[none]"}

Thesis to revise:
{thesis}

Reviewer feedback:
{feedback}

Revised thesis:"""

    return call_llm(prompt, model=model, max_tokens=1536)


def self_refine_thesis(
    financials: str,
    model: str,
    max_iterations: int,
) -> tuple[str, str]:
    print("[STEP 1] Generating initial thesis ...")
    thesis = generate_initial_thesis(financials, model)
    history_feedback = ""
    print(f"\nInitial Thesis:\n{thesis}\n")

    for iteration in range(1, max_iterations + 1):
        print(f"[STEP 2] Iteration {iteration} — Getting feedback...")
        feedback = get_feedback(
            thesis=thesis,
            financials=financials,
            history_feedback=history_feedback,
            model=model,
        )
        print(f"\nFeedback:\n{feedback}\n")

        if feedback.strip().startswith("NO_FURTHER_FEEDBACK"):
            print("Stopping: evaluator reports no further blocking improvements.")
            break

        history_feedback += (
            f"Iteration: {iteration}\n"
            f"Thesis:\n{thesis}\n"
            f"Feedback:\n{feedback}\n\n"
        )

        print(f"[STEP 3] Iteration {iteration} — Refining thesis...")
        thesis = refine_thesis(
            thesis=thesis,
            feedback=feedback,
            financials=financials,
            history_feedback=history_feedback,
            model=model,
        )
        print(f"\nRefined Thesis:\n{thesis}\n")

    return thesis, history_feedback


def save_report(
    ticker: str,
    model: str,
    financials: str,
    final_thesis: str,
    history_feedback: str,
) -> Path:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{ticker}_self_refine_{stamp}.md"

    path.write_text(
        f"""# Self-Refine Equity Thesis — {ticker}

- Model: `{model}`
- Generated: {datetime.now().isoformat(timespec="seconds")}

## Financial data used

```text
{financials.strip()}
```

## Final thesis

{final_thesis}

## Refinement history

{history_feedback.strip() or "No refinement was required."}
""",
        encoding="utf-8",
    )

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Refine equity thesis using OpenAI and Financial Datasets."
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Public-company ticker, e.g. AAPL, NVDA, AMD, TSLA.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum evaluator/refiner rounds (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Copy .env.example to .env and add your key."
        )

    if args.max_iterations < 0:
        raise ValueError("--max-iterations must be >= 0")

    ticker = args.ticker.strip().upper()

    print(f"[DATA] Fetching Financial Datasets snapshot for {ticker} ...")
    financials = get_financial_data(ticker)
    print(f"\nFinancial data supplied to the model:\n{financials}")

    final_thesis, history_feedback = self_refine_thesis(
        financials=financials,
        model=args.model,
        max_iterations=args.max_iterations,
    )

    print("\nFINAL THESIS:\n")
    print(final_thesis)

    report_path = save_report(
        ticker=ticker,
        model=args.model,
        financials=financials,
        final_thesis=final_thesis,
        history_feedback=history_feedback,
    )
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
