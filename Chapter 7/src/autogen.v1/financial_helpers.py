"""Robust helpers for the Chapter 7 AutoGen v1 demo.

The original lab intentionally lets the LLM write yfinance code from scratch.
That is pedagogically useful, but yfinance statement row labels can change
(e.g. "Stockholders Equity" vs. "Total Stockholder Equity").  These helpers
provide a stable first path for common health metrics while preserving the
AssistantAgent -> code_executor conversational loop.

Important: when a ratio is derived from statements, this module uses the
latest non-null period only.  It never sums balance-sheet values across years.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import yfinance as yf


def _number(value: Any) -> float | None:
    """Return a finite float or None."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _latest_statement_value(frame, labels: Iterable[str]) -> float | None:
    """Return the latest non-null value for the first matching row label."""
    if frame is None or frame.empty:
        return None

    for label in labels:
        if label not in frame.index:
            continue

        row = frame.loc[label]
        # yfinance normally returns newest statement columns first.  Prefer
        # explicit date ordering when possible, otherwise preserve its order.
        try:
            columns = sorted(frame.columns, reverse=True)
            values = [row[col] for col in columns]
        except Exception:
            values = list(row)

        for value in values:
            number = _number(value)
            if number is not None:
                return number

    return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def get_health_snapshot(symbol: str) -> dict[str, Any]:
    """Fetch a robust financial-health snapshot for a ticker.

    Preferred source:
      - yfinance Ticker.info for standard published ratios.

    Fallback:
      - latest-period income statement / balance sheet values selected using
        multiple known row-label variants.

    The returned dictionary is deliberately plain so generated code can print
    it directly and the assistant can reason over observed execution output.
    """
    symbol = symbol.upper().strip()
    ticker = yf.Ticker(symbol)

    info = ticker.info or {}
    financials = ticker.financials
    balance_sheet = ticker.balance_sheet

    history = ticker.history(period="5d", auto_adjust=False)
    history_close = None
    if history is not None and not history.empty and "Close" in history:
        closes = history["Close"].dropna()
        if not closes.empty:
            history_close = _number(closes.iloc[-1])

    price = _number(info.get("currentPrice")) or history_close
    trailing_eps = _number(info.get("trailingEps"))
    trailing_pe = _number(info.get("trailingPE"))
    if trailing_pe is None:
        trailing_pe = _safe_ratio(price, trailing_eps)

    revenue = _latest_statement_value(
        financials,
        ["Total Revenue", "Operating Revenue"],
    )
    net_income = _latest_statement_value(
        financials,
        ["Net Income", "Net Income Common Stockholders"],
    )
    gross_profit = _latest_statement_value(financials, ["Gross Profit"])

    equity = _latest_statement_value(
        balance_sheet,
        [
            "Stockholders Equity",
            "Total Stockholder Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        ],
    )
    total_assets = _latest_statement_value(balance_sheet, ["Total Assets"])
    current_assets = _latest_statement_value(
        balance_sheet,
        ["Current Assets", "Total Current Assets"],
    )
    current_liabilities = _latest_statement_value(
        balance_sheet,
        ["Current Liabilities", "Total Current Liabilities"],
    )
    inventory = _latest_statement_value(
        balance_sheet,
        ["Inventory", "Inventories"],
    )
    total_debt = _latest_statement_value(balance_sheet, ["Total Debt"])

    roe = _number(info.get("returnOnEquity"))
    if roe is None:
        roe = _safe_ratio(net_income, equity)

    roa = _number(info.get("returnOnAssets"))
    if roa is None:
        roa = _safe_ratio(net_income, total_assets)

    net_margin = _number(info.get("profitMargins"))
    if net_margin is None:
        net_margin = _safe_ratio(net_income, revenue)

    gross_margin = _number(info.get("grossMargins"))
    if gross_margin is None:
        gross_margin = _safe_ratio(gross_profit, revenue)

    current_ratio = _number(info.get("currentRatio"))
    if current_ratio is None:
        current_ratio = _safe_ratio(current_assets, current_liabilities)

    quick_ratio = _number(info.get("quickRatio"))
    if quick_ratio is None and current_assets is not None:
        adjusted_current_assets = current_assets - (inventory or 0.0)
        quick_ratio = _safe_ratio(adjusted_current_assets, current_liabilities)

    # Derive a normalized D/E ratio from the latest balance sheet when
    # possible. This avoids ambiguity in Yahoo's debtToEquity field, whose
    # scale may be presented as a percentage-like value.
    debt_to_equity = _safe_ratio(total_debt, equity)
    if debt_to_equity is None:
        raw_de = _number(info.get("debtToEquity"))
        if raw_de is not None:
            debt_to_equity = raw_de / 100.0

    return {
        "symbol": symbol,
        "company_name": info.get("longName") or info.get("shortName") or symbol,
        "currency": info.get("currency"),
        "current_price": price,
        "trailing_pe": trailing_pe,
        "return_on_assets": roa,
        "return_on_equity": roe,
        "net_profit_margin": net_margin,
        "gross_margin": gross_margin,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "debt_to_equity_ratio": debt_to_equity,
        "latest_revenue": revenue,
        "latest_net_income": net_income,
        "latest_total_assets": total_assets,
        "latest_stockholders_equity": equity,
    }


def save_price_plot(
    symbol: str,
    period: str = "1y",
    filename: str | None = None,
) -> str:
    """Save a closing-price plot and return its path."""
    symbol = symbol.upper().strip()
    ticker = yf.Ticker(symbol)
    history = ticker.history(period=period, auto_adjust=False)

    if history is None or history.empty or "Close" not in history:
        raise RuntimeError(f"No price history returned for {symbol}.")

    filename = filename or f"{symbol.lower()}_stock_price.png"
    output_path = Path(filename)

    plt.figure(figsize=(10, 6))
    plt.plot(history.index, history["Close"], label="Close Price")
    plt.title(f"{symbol} Stock Price - {period}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return str(output_path)
