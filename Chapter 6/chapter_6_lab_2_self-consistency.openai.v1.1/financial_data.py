"""Financial data helper for Self-Consistency v1.1.

Fetches structured company data from Financial Datasets and exposes a richer
snapshot than v1.0, including growth, cash-flow and balance-strength metrics
already present in the financial-metrics snapshot.
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


def _ratio_to_percent(data: dict[str, Any] | None, field: str) -> float | None:
    """Convert Financial Datasets ratio-style growth fields to percentage points."""
    value = _number(data, field)
    return None if value is None else value * 100.0


def _money(value: float | None, scale: float, suffix: str) -> str:
    if value is None:
        return "unavailable"
    return f"${value / scale:,.2f}{suffix}"


def _percent(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.{decimals}f}%"


def _multiple(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.2f}x"


def _dollars(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"${value:,.2f}"


def get_financial_snapshot(ticker: str) -> dict[str, Any]:
    """Fetch an enriched financial snapshot for one public company."""
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
    revenue_growth_source = "unavailable"
    if revenue is not None and previous_revenue is not None and previous_revenue > 0:
        revenue_growth = ((revenue - previous_revenue) / previous_revenue) * 100.0
        revenue_growth_source = "calculated from two annual income statements"
    else:
        revenue_growth = _ratio_to_percent(metrics_snapshot, "revenue_growth")
        if revenue_growth is not None:
            revenue_growth_source = "Financial Datasets metrics snapshot"

    gross_profit = _number(current, "gross_profit")
    operating_income = _number(current, "operating_income")
    gross_margin = None
    operating_margin = None
    if revenue is not None and revenue > 0:
        if gross_profit is not None:
            gross_margin = gross_profit / revenue * 100.0
        if operating_income is not None:
            operating_margin = operating_income / revenue * 100.0

    eps = _number(current, "earnings_per_share_diluted")
    if eps is None:
        eps = _number(current, "earnings_per_share")

    price = _number(price_snapshot, "price")

    pe_ratio = None
    if eps is not None and eps > 0 and price is not None and price > 0:
        pe_ratio = price / eps

    free_cash_flow_per_share = _number(metrics_snapshot, "free_cash_flow_per_share")
    fcf_yield = None
    if (
        free_cash_flow_per_share is not None
        and free_cash_flow_per_share > 0
        and price is not None
        and price > 0
    ):
        fcf_yield = free_cash_flow_per_share / price * 100.0

    return {
        "ticker": ticker,
        "company_name": current.get("company_name") or ticker,
        "report_period": current.get("report_period") or "unavailable",
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "revenue_growth_source": revenue_growth_source,
        "net_income": _number(current, "net_income"),
        "eps": eps,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "price": price,
        "pe_ratio": pe_ratio,
        "market_cap": _number(metrics_snapshot, "market_cap"),
        "earnings_growth": _ratio_to_percent(metrics_snapshot, "earnings_growth"),
        "eps_growth": _ratio_to_percent(
            metrics_snapshot, "earnings_per_share_growth"
        ),
        "free_cash_flow_growth": _ratio_to_percent(
            metrics_snapshot, "free_cash_flow_growth"
        ),
        "operating_income_growth": _ratio_to_percent(
            metrics_snapshot, "operating_income_growth"
        ),
        "ebitda_growth": _ratio_to_percent(metrics_snapshot, "ebitda_growth"),
        "debt_to_equity": _number(metrics_snapshot, "debt_to_equity"),
        "current_ratio": _number(metrics_snapshot, "current_ratio"),
        "free_cash_flow_per_share": free_cash_flow_per_share,
        "fcf_yield": fcf_yield,
    }


def format_financial_snapshot(snapshot: dict[str, Any]) -> str:
    """Convert the structured snapshot into grounded text for each LLM path."""
    eps = _dollars(snapshot["eps"])
    price = _dollars(snapshot["price"])
    pe = (
        f"{snapshot['pe_ratio']:.1f}x"
        if snapshot["pe_ratio"] is not None
        else "unavailable"
    )

    return f"""----- {snapshot['ticker']} / {snapshot['company_name']} -----
Report period: {snapshot['report_period']}

Income statement / profitability:
Revenue: {_money(snapshot['revenue'], 1_000_000_000, 'B')}
Net income: {_money(snapshot['net_income'], 1_000_000_000, 'B')}
Diluted EPS: {eps}
Gross margin: {_percent(snapshot['gross_margin'], 1)}
Operating margin: {_percent(snapshot['operating_margin'], 1)}

Growth:
Revenue growth: {_percent(snapshot['revenue_growth'])} ({snapshot['revenue_growth_source']})
Earnings growth: {_percent(snapshot['earnings_growth'])}
Diluted EPS growth: {_percent(snapshot['eps_growth'])}
Free cash flow growth: {_percent(snapshot['free_cash_flow_growth'])}
Operating income growth: {_percent(snapshot['operating_income_growth'])}
EBITDA growth: {_percent(snapshot['ebitda_growth'])}

Financial strength / cash flow:
Debt to equity: {_multiple(snapshot['debt_to_equity'])}
Current ratio: {_multiple(snapshot['current_ratio'])}
Free cash flow per share: {_dollars(snapshot['free_cash_flow_per_share'])}
FCF yield based on current price: {_percent(snapshot['fcf_yield'])}

Valuation / market:
Current price snapshot: {price}
P/E based on current price and reported EPS: {pe}
Market capitalization: {_money(snapshot['market_cap'], 1_000_000_000, 'B')}
"""


def get_financial_data(ticker: str) -> str:
    """Fetch and format the data used by the Self-Consistency paths."""
    return format_financial_snapshot(get_financial_snapshot(ticker))
