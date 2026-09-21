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

from point_in_time_data import fetch_historical_news


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
    print(f"Titulares : {len(result.get('headlines') or [])}")
    print(f"Error     : {result.get('error') or 'NINGUNO'}")
    print("-" * 100)

    for i, article in enumerate(result.get("articles") or [], start=1):
        print(
            f"{i:02d}. {article.get('time_published') or ''} | "
            f"{article.get('source') or ''}\n"
            f"    {article.get('title') or ''}"
        )

    if result.get("error"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
