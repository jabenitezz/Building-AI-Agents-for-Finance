"""Chapter 6 Lab 2 v1.1 — Self-Consistency with OpenAI + Financial Datasets.

Pattern:
    enriched financial data -> N independent analyses -> parse labels -> vote

Each path returns a concise, user-facing rationale plus a final BUY/HOLD/SELL
label. Hidden chain-of-thought is neither requested nor exposed.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from financial_data import get_financial_data


load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_N_PATHS = 5
DEFAULT_TEMPERATURE = 0.8

client = OpenAI()


def language_instruction(language: str) -> str:
    if language == "es":
        return (
            "Respond entirely in Spanish (Spain), but keep standard financial "
            "abbreviations such as P/E, EPS, FCF, and EBITDA unchanged. "
            "IMPORTANT: regardless of the response language, the final "
            "machine-readable line must remain exactly in English as "
            "RECOMMENDATION: <BUY|HOLD|SELL>."
        )
    return (
        "Respond entirely in English. The final machine-readable line must be "
        "exactly RECOMMENDATION: <BUY|HOLD|SELL>."
    )


def build_prompt(financials: str, language: str) -> str:
    return f"""You are a financial analyst.

Analyze the supplied financial snapshot independently and choose one final
label: BUY, HOLD, or SELL.

Grounding rules:
- Use ONLY the supplied financial data.
- Do not use outside knowledge about the company.
- Do not invent peer comparisons, historical multiples, catalysts, news,
  management guidance, macro conditions, or missing metrics.
- Treat provider growth fields exactly as supplied; do not infer an unstated
  time period for a growth metric.
- Missing information should reduce confidence, but must not automatically
  force a HOLD label.
- Choose the label whose case is best supported by the supplied evidence.
- The label is an educational model output, not personalized investment advice.

Provide a concise rationale covering:
1. Profitability and earnings quality.
2. Growth evidence across revenue, earnings, EPS, FCF, operating income and
   EBITDA when available.
3. Financial strength and cash-flow evidence, including debt/equity,
   current ratio and FCF/share when available.
4. Valuation using only supplied metrics such as P/E and FCF yield.
5. Key limitations and the overall risk/reward balance.

Do not provide private chain-of-thought or hidden reasoning. Give only a short,
user-facing rationale: 1-2 sentences per point.

End with exactly one plain-text line:
RECOMMENDATION: <BUY|HOLD|SELL>

{language_instruction(language)}

Financial data:
{financials}
"""


def sample_analysis_path(
    financials: str,
    model: str,
    temperature: float,
    language: str,
) -> str:
    """Generate one independent analysis path."""
    response = client.responses.create(
        model=model,
        input=build_prompt(financials, language),
        max_output_tokens=1400,
        temperature=temperature,
        store=False,
    )
    return response.output_text.strip()


def extract_recommendation(path: str) -> str | None:
    """Parse BUY/HOLD/SELL robustly, including a translated Spanish header."""
    prefixes = ("RECOMMENDATION:", "RECOMENDACIÓN:", "RECOMENDACION:")

    for line in reversed(path.strip().splitlines()):
        cleaned = line.strip().lstrip("*_`# ").upper()
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                label = cleaned.removeprefix(prefix).strip().strip("*_`. ")
                if label in {"BUY", "HOLD", "SELL"}:
                    return label
    return None


def aggregate_recommendations(
    recommendations: list[str | None],
) -> tuple[str, dict[str, int]]:
    """Return plurality label; report NO_CONSENSUS on an exact top-vote tie."""
    valid = [r for r in recommendations if r is not None]
    if not valid:
        return "NO_VALID_VOTES", {}

    vote_counts = Counter(valid)
    ranked = vote_counts.most_common()
    top_count = ranked[0][1]
    winners = [label for label, count in ranked if count == top_count]

    if len(winners) > 1:
        return "NO_CONSENSUS", dict(vote_counts)
    return winners[0], dict(vote_counts)


def self_consistency_recommend(
    financials: str,
    model: str,
    n_paths: int,
    temperature: float,
    language: str,
) -> dict:
    paths: list[str] = []
    recommendations: list[str | None] = []

    print(
        f"[STEP 1] Sampling {n_paths} independent paths "
        f"with model={model}, temperature={temperature}.\n"
    )

    for i in range(1, n_paths + 1):
        print(f"[STEP 2] Sampling analysis path {i}/{n_paths} ...")
        path = sample_analysis_path(
            financials=financials,
            model=model,
            temperature=temperature,
            language=language,
        )
        recommendation = extract_recommendation(path)
        paths.append(path)
        recommendations.append(recommendation)
        print(f"--- Path {i} ---\n{path}\n")
        print(f"[Path {i}] Parsed recommendation: {recommendation}\n")

    print("[STEP 3] Aggregating labels via plurality vote ...")
    final_recommendation, vote_counts = aggregate_recommendations(recommendations)

    return {
        "final_recommendation": final_recommendation,
        "vote_counts": vote_counts,
        "recommendations": recommendations,
        "paths": paths,
    }


def save_report(
    ticker: str,
    model: str,
    temperature: float,
    financials: str,
    result: dict,
) -> Path:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{ticker}_self_consistency_{stamp}.md"

    path_sections = []
    for i, (analysis, recommendation) in enumerate(
        zip(result["paths"], result["recommendations"]), start=1
    ):
        path_sections.append(
            f"## Path {i}\n\nParsed recommendation: `{recommendation}`\n\n{analysis}"
        )

    path.write_text(
        f"""# Self-Consistency Equity Analysis v1.1 — {ticker}

- Model: `{model}`
- Temperature: `{temperature}`
- Generated: {datetime.now().isoformat(timespec="seconds")}

## Financial data used

```text
{financials.strip()}
```

## Aggregate result

- Final label: **{result['final_recommendation']}**
- Vote breakdown: `{result['vote_counts']}`

{chr(10).join(path_sections)}
""",
        encoding="utf-8",
    )

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Self-Consistency v1.1 stock analysis using OpenAI "
            "and Financial Datasets."
        )
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
        "--n-paths",
        type=int,
        default=DEFAULT_N_PATHS,
        help=f"Independent samples (default: {DEFAULT_N_PATHS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--language",
        choices=("en", "es"),
        default="en",
        help="Output language: en or es (default: en).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Copy .env.example to .env and add your key."
        )
    if args.n_paths < 1:
        raise ValueError("--n-paths must be >= 1")
    if not 0 <= args.temperature <= 2:
        raise ValueError("--temperature must be between 0 and 2")

    ticker = args.ticker.strip().upper()

    print(f"[DATA] Fetching enriched Financial Datasets snapshot for {ticker} ...")
    financials = get_financial_data(ticker)
    print(f"\nFinancial data supplied to each path:\n{financials}")

    result = self_consistency_recommend(
        financials=financials,
        model=args.model,
        n_paths=args.n_paths,
        temperature=args.temperature,
        language=args.language,
    )

    print(f"\n=== AGGREGATE LABEL: {result['final_recommendation']} ===")
    print(f"Vote breakdown: {result['vote_counts']}")

    report_path = save_report(
        ticker=ticker,
        model=args.model,
        temperature=args.temperature,
        financials=financials,
        result=result,
    )
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
