"""Financial data helper for the OpenAI Self-Consistency lab.

Fetches structured company metrics from Financial Datasets and converts them
into a compact, grounded text block for independent LLM analysis paths.
"""

from __future__ import annotations

import math
import os
from typing import Any

import httpx

API_BASE = "https://api.financialdatasets.ai"


def _number(data: dict[str, Any] | None, field: str) -> float | None:
    """Return a finite numeric value or None when unavailable."""
    if not data:
        return None
    value = data.get(field)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _money(value: float | None, scale: float, suffix: str) -> str:
    if value is None:
        return "unavailable"
    return f"${value / scale:,.2f}{suffix}"


def _percent(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.1f}%"


def get_financial_snapshot(ticker: str) -> dict[str, Any]:
    """Fetch a compact financial snapshot for one public company."""
    api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FINANCIAL_DATASETS_API_KEY is not configured. "
            "Copy .env.example to .env and add your key."
        )

    ticker = ticker.strip().upper()
    headers = {"X-API-Key": api_key}

    with httpx.Client(timeout=30.0) as client:
        income_resp = client.get(
            f"{API_BASE}/financials/income-statements",
            params={"ticker": ticker, "period": "annual", "limit": 2},
            headers=headers,
        )
        income_resp.raise_for_status()
        income_data = income_resp.json().get("income_statements", [])
        if not income_data:
            raise ValueError(f"No annual income-statement data found for {ticker}")

        price_resp = client.get(
            f"{API_BASE}/prices/snapshot",
            params={"ticker": ticker},
            headers=headers,
        )
        price_resp.raise_for_status()
        price_snapshot = price_resp.json().get("snapshot", {})

        metrics_resp = client.get(
            f"{API_BASE}/financial-metrics/snapshot",
            params={"ticker": ticker},
            headers=headers,
        )
        metrics_resp.raise_for_status()
        metrics_snapshot = metrics_resp.json().get("snapshot", {})

    current = income_data[0]
    previous = income_data[1] if len(income_data) > 1 else None

    revenue = _number(current, "revenue")
    previous_revenue = _number(previous, "revenue")
    revenue_growth = None
    if revenue is not None and previous_revenue is not None and previous_revenue > 0:
        revenue_growth = ((revenue - previous_revenue) / previous_revenue) * 100

    gross_profit = _number(current, "gross_profit")
    operating_income = _number(current, "operating_income")
    gross_margin = None
    operating_margin = None
    if revenue is not None and revenue > 0:
        if gross_profit is not None:
            gross_margin = gross_profit / revenue * 100
        if operating_income is not None:
            operating_margin = operating_income / revenue * 100

    eps = _number(current, "earnings_per_share_diluted")
    if eps is None:
        eps = _number(current, "earnings_per_share")

    price = _number(price_snapshot, "price")
    pe_ratio = None
    if eps is not None and eps > 0 and price is not None and price > 0:
        pe_ratio = price / eps

    return {
        "ticker": ticker,
        "company_name": current.get("company_name") or ticker,
        "report_period": current.get("report_period") or "unavailable",
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "net_income": _number(current, "net_income"),
        "eps": eps,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "price": price,
        "pe_ratio": pe_ratio,
        "market_cap": _number(metrics_snapshot, "market_cap"),
    }


def format_financial_snapshot(snapshot: dict[str, Any]) -> str:
    """Convert the structured snapshot into grounded text for the LLM."""
    growth = (
        _percent(snapshot["revenue_growth"])
        if snapshot["revenue_growth"] is not None
        else "unavailable (provider returned only one comparable annual period)"
    )
    eps = f"${snapshot['eps']:.2f}" if snapshot["eps"] is not None else "unavailable"
    price = f"${snapshot['price']:.2f}" if snapshot["price"] is not None else "unavailable"
    pe = f"{snapshot['pe_ratio']:.1f}x" if snapshot["pe_ratio"] is not None else "unavailable"

    return f"""----- {snapshot['ticker']} / {snapshot['company_name']} -----
Report period: {snapshot['report_period']}
Revenue: {_money(snapshot['revenue'], 1_000_000_000, 'B')}
Revenue growth YoY: {growth}
Net income: {_money(snapshot['net_income'], 1_000_000_000, 'B')}
Diluted EPS: {eps}
Gross margin: {_percent(snapshot['gross_margin'])}
Operating margin: {_percent(snapshot['operating_margin'])}
Current price snapshot: {price}
P/E based on current price and reported EPS: {pe}
Market capitalization: {_money(snapshot['market_cap'], 1_000_000_000, 'B')}
"""


def get_financial_data(ticker: str) -> str:
    """Fetch and format the financial data used by the Self-Consistency loop."""
    return format_financial_snapshot(get_financial_snapshot(ticker))
