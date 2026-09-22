"""Audit point-in-time TTM fundamentals without any LLM calls."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
OPENAI_V1_ROOT = HERE.parent
if str(OPENAI_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENAI_V1_ROOT))

load_dotenv(OPENAI_V1_ROOT / ".env")
load_dotenv(HERE / ".env", override=False)

from point_in_time_data import fetch_fundamentals, fetch_technicals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Muestra el snapshot fundamental TTM SEC point-in-time."
    )
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--as-of", default="2026-01-30")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    as_of = date.fromisoformat(args.as_of)

    technicals = fetch_technicals(ticker, as_of)
    fundamentals = fetch_fundamentals(
        ticker,
        as_of,
        float(technicals["last_close"]),
    )

    print("=" * 100)
    print(f"FUNDAMENTALS PIT TTM | {ticker} | as_of={as_of}")
    print("=" * 100)
    print(f"price_at_decision : {fundamentals.get('price_at_decision')}")
    print(f"mode              : {fundamentals.get('fundamental_mode')}")
    print(f"source            : {fundamentals.get('source')}")
    print(f"available_by      : {fundamentals.get('filing_date')}")
    print(f"ttm_end           : {fundamentals.get('ttm_end')}")
    print(f"ttm_method        : {fundamentals.get('ttm_method')}")
    print(f"prior_ttm_end     : {fundamentals.get('prior_ttm_end')}")
    print(f"balance_end       : {fundamentals.get('balance_period_end')}")
    print(f"balance_filed     : {fundamentals.get('balance_filing_date')}")
    print(f"balance_forms     : {fundamentals.get('balance_forms')}")
    print("-" * 100)
    print(f"revenue_ttm       : {fundamentals.get('revenue')}")
    print(f"net_income_ttm    : {fundamentals.get('net_income')}")
    print(f"revenue_growth    : {fundamentals.get('revenueGrowth')}")
    print(f"earnings_growth   : {fundamentals.get('earningsGrowth')}")
    print(f"profit_margin     : {fundamentals.get('profitMargins')}")
    print(f"ROE               : {fundamentals.get('returnOnEquity')}")
    print(f"P/E TTM           : {fundamentals.get('trailingPE')}")
    print(f"P/B               : {fundamentals.get('priceToBook')}")
    print(f"Debt/Equity       : {fundamentals.get('debtToEquity')}")
    print(f"Current Ratio     : {fundamentals.get('currentRatio')}")
    print("-" * 100)
    print("REVENUE TTM COMPONENTS")
    print(json.dumps(
        fundamentals.get("revenue_ttm_components"),
        ensure_ascii=False,
        indent=2,
    ))
    print("-" * 100)
    print("NET INCOME TTM COMPONENTS")
    print(json.dumps(
        fundamentals.get("net_income_ttm_components"),
        ensure_ascii=False,
        indent=2,
    ))
    print("-" * 100)
    print("FULL AUDIT JSON")
    print(json.dumps(fundamentals, ensure_ascii=False, indent=2))
    print("=" * 100)
    print("No se ha realizado ninguna llamada a un LLM.")


if __name__ == "__main__":
    main()
