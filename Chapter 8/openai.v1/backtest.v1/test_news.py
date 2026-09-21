"""Test Alpha Vantage historical news without any LLM calls."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
OPENAI_V1_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

load_dotenv(OPENAI_V1_ROOT / ".env")
load_dotenv(HERE / ".env", override=False)

from point_in_time_data import fetch_historical_news, fetch_next_session_open


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prueba NEWS_SENTIMENT histórico de Alpha Vantage sin llamar a ningún LLM."
    )
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--start", default="2025-12-31")
    p.add_argument("--end", default="2026-01-30")
    p.add_argument("--limit", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = fetch_historical_news(
        args.ticker.upper(),
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        max_records=args.limit,
    )

    print("=" * 100)
    print("PRUEBA DE NOTICIAS HISTÓRICAS")
    print("=" * 100)
    print(f"Proveedor : {result.get('source')}")
    print(f"Ticker    : {args.ticker.upper()}")
    print(f"Ventana   : {args.start} -> {args.end}")
    print(f"Raw       : {result.get('raw_count', 'N/D')}")
    print(f"Elegibles : {result.get('eligible_count', 'N/D')}")
    print(f"Final     : {len(result.get('headlines') or [])}")
    print(f"Rechazadas: {result.get('filtered_out_count', 'N/D')}")
    print(f"Dup/vacías: {result.get('duplicate_or_empty_count', 'N/D')}")
    print(f"Umbral T  : {result.get('title_relevance_threshold', 'N/D')}")
    print(f"Umbral R  : {result.get('summary_relevance_threshold', 'N/D')}")
    print(f"Error     : {result.get('error') or 'NINGUNO'}")
    print("-" * 100)

    for i, article in enumerate(result.get("articles") or [], start=1):
        print(
            f"{i:02d}. {article.get('time_published') or ''} | "
            f"{article.get('source') or ''} | "
            f"relevance={article.get('ticker_relevance_score')} | "
            f"match={article.get('match_scope')} | "
            f"aliases={','.join(article.get('matched_aliases') or [])}\n"
            f"    {article.get('title') or ''}"
        )

    if result.get("error"):
        raise SystemExit(2)

    execution = fetch_next_session_open(
        args.ticker.upper(),
        date.fromisoformat(args.end),
    )
    print("-" * 100)
    print(
        "Ejecución  : "
        f"{execution['execution_rule']} | "
        f"{execution['execution_date']} | "
        f"open={execution['execution_open_price']:.2f}"
    )


if __name__ == "__main__":
    main()
