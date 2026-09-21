"""Point-in-time data layer for Chapter 8 OpenAI backtest.v1.

Goal: when the committee evaluates a historical date, every data source is
bounded by that date. Current/future news or current technical indicators are
never used as fallbacks.

Data sources:
- Prices/technicals/macro: yfinance historical series, cut at as_of_date.
- Fundamentals: SEC Company Facts, using only facts filed on/before as_of_date.
- News: GDELT DOC API, explicitly bounded to a historical date window.

This is still an educational research harness. SEC annual facts are used for
fundamentals to keep the point-in-time logic auditable and conservative.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / "cache"
SEC_CACHE = CACHE_DIR / "sec"
SEC_CACHE.mkdir(parents=True, exist_ok=True)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

COMPANY_NAMES = {
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}


def _trace(message: str) -> None:
    print(f"[PIT-DATA] {message}", flush=True)


def _as_date(value: str | date | datetime | pd.Timestamp) -> date:
    return pd.Timestamp(value).date()


def _naive_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def _history(
    ticker: str,
    start: date,
    end_inclusive: date,
) -> pd.DataFrame:
    # yfinance treats end= as exclusive.
    df = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(end_inclusive + timedelta(days=1)).isoformat(),
        auto_adjust=False,
    )
    if not df.empty:
        df = df.copy()
        df.index = _naive_index(df.index)
        df = df[df.index.date <= end_inclusive]
    return df


def resolve_trading_date(
    anchor: str | date | datetime | pd.Timestamp,
    proxy: str = "SPY",
) -> date:
    """Resolve an anchor date to the latest US trading date on/before it."""
    d = _as_date(anchor)
    df = _history(proxy, d - timedelta(days=10), d)
    if df.empty:
        raise RuntimeError(f"No se pudo resolver una sesión bursátil para {d}.")
    return df.index[-1].date()


def fetch_technicals(ticker: str, as_of_date: date) -> dict[str, Any]:
    """Historical technical snapshot using only closes available by as_of_date."""
    df = _history(ticker, as_of_date - timedelta(days=450), as_of_date)
    if df.empty or "Close" not in df:
        raise RuntimeError(f"Sin precios históricos para {ticker} hasta {as_of_date}.")

    close = df["Close"].dropna()
    if close.empty:
        raise RuntimeError(f"Sin cierres históricos para {ticker} hasta {as_of_date}.")

    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = (100 - 100 / (1 + rs)).iloc[-1] if len(close) >= 15 else None
    vol_30d = (
        close.pct_change().rolling(30).std().iloc[-1] * (252 ** 0.5)
        if len(close) >= 31
        else None
    )

    def _num(v: Any) -> float | None:
        if v is None or pd.isna(v):
            return None
        return float(v)

    return {
        "ticker": ticker,
        "data_end": close.index[-1].date().isoformat(),
        "last_close": float(close.iloc[-1]),
        "sma_50": _num(sma_50),
        "sma_200": _num(sma_200),
        "rsi_14": _num(rsi),
        "vol_30d_annualized": _num(vol_30d),
        "observations": int(len(close)),
    }


def _last_close(ticker: str, as_of_date: date, lookback_days: int = 45) -> tuple[date, float]:
    df = _history(ticker, as_of_date - timedelta(days=lookback_days), as_of_date)
    if df.empty:
        raise RuntimeError(f"Sin datos para {ticker} hasta {as_of_date}.")
    close = df["Close"].dropna()
    return close.index[-1].date(), float(close.iloc[-1])


def fetch_macro(as_of_date: date) -> dict[str, Any]:
    """Historical VIX, US 10Y and S&P 500 1-month return as of the decision date."""
    vix_date, vix = _last_close("^VIX", as_of_date)
    tnx_date, tnx = _last_close("^TNX", as_of_date)

    spx = _history("^GSPC", as_of_date - timedelta(days=45), as_of_date)
    spx_close = spx["Close"].dropna()
    if len(spx_close) < 2:
        raise RuntimeError(f"Sin historial suficiente del S&P 500 hasta {as_of_date}.")

    cutoff = pd.Timestamp(as_of_date - timedelta(days=30))
    older = spx_close[spx_close.index <= cutoff]
    base = float(older.iloc[-1] if len(older) else spx_close.iloc[0])
    last = float(spx_close.iloc[-1])
    ret_1m = (last / base - 1.0) * 100.0

    return {
        "data_end": min(vix_date, tnx_date, spx_close.index[-1].date()).isoformat(),
        "vix": vix,
        "us_10y_yield_pct": tnx,
        "spx_1m_return_pct": ret_1m,
    }


def _sec_headers() -> dict[str, str]:
    email = os.getenv("SEC_EDGAR_EMAIL", "").strip()
    if not email:
        raise RuntimeError(
            "Falta SEC_EDGAR_EMAIL. Añade SEC_EDGAR_EMAIL=tu-correo@dominio "
            "a Chapter 8/openai.v1/.env o backtest.v1/.env. No es una clave API."
        )
    return {
        "User-Agent": f"Building-AI-Agents-for-Finance Chapter8 Backtest {email}",
        "Accept-Encoding": "gzip, deflate",
    }


def _cached_json(path: Path, url: str) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    with httpx.Client(headers=_sec_headers(), timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(0.12)
    return data


def _ticker_map() -> dict[str, int]:
    data = _cached_json(SEC_CACHE / "company_tickers.json", SEC_TICKERS_URL)
    result: dict[str, int] = {}
    for item in data.values():
        ticker = str(item.get("ticker", "")).upper()
        cik = item.get("cik_str")
        if ticker and cik is not None:
            result[ticker] = int(cik)
    return result


def _company_facts(ticker: str) -> dict[str, Any]:
    ticker = ticker.upper()
    cik = _ticker_map().get(ticker)
    if cik is None:
        raise RuntimeError(f"No se encontró CIK SEC para {ticker}.")
    return _cached_json(
        SEC_CACHE / f"CIK{cik:010d}.json",
        SEC_FACTS_URL.format(cik=cik),
    )


def _concept_units(
    facts: dict[str, Any],
    concepts: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    root = facts.get("facts", {})
    for namespace, concept in concepts:
        node = root.get(namespace, {}).get(concept)
        if not node:
            continue
        units = node.get("units", {})
        for unit in ("USD", "USD/shares", "shares"):
            if unit in units:
                return list(units[unit])
        if units:
            return list(next(iter(units.values())))
    return []


def _filed_on_or_before(entry: dict[str, Any], as_of_date: date) -> bool:
    filed = entry.get("filed")
    end = entry.get("end")
    if not filed or not end:
        return False
    try:
        return _as_date(filed) <= as_of_date and _as_date(end) <= as_of_date
    except Exception:
        return False


def _annual_flow_records(
    entries: list[dict[str, Any]],
    as_of_date: date,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for e in entries:
        if e.get("form") not in {"10-K", "10-K/A"}:
            continue
        if not _filed_on_or_before(e, as_of_date):
            continue
        start = e.get("start")
        end = e.get("end")
        if not start or not end:
            continue
        try:
            duration = (_as_date(end) - _as_date(start)).days
        except Exception:
            continue
        if duration < 250:
            continue
        filtered.append(e)

    best: dict[str, dict[str, Any]] = {}
    for e in filtered:
        key = str(e["end"])
        old = best.get(key)
        if old is None or str(e.get("filed", "")) > str(old.get("filed", "")):
            best[key] = e

    return sorted(best.values(), key=lambda x: str(x["end"]), reverse=True)


def _instant_record(
    entries: list[dict[str, Any]],
    as_of_date: date,
    preferred_end: str | None = None,
) -> dict[str, Any] | None:
    candidates = [
        e for e in entries
        if e.get("form") in {"10-K", "10-K/A"} and _filed_on_or_before(e, as_of_date)
    ]
    if preferred_end:
        matching = [e for e in candidates if str(e.get("end")) == preferred_end]
        if matching:
            candidates = matching
    if not candidates:
        return None
    return max(candidates, key=lambda e: (str(e.get("end", "")), str(e.get("filed", ""))))


def _value(entry: dict[str, Any] | None) -> float | None:
    if not entry:
        return None
    try:
        return float(entry["val"])
    except Exception:
        return None


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def fetch_fundamentals(ticker: str, as_of_date: date, price: float) -> dict[str, Any]:
    """Conservative point-in-time fundamentals from SEC annual filings."""
    facts = _company_facts(ticker)

    revenue_entries = _concept_units(facts, [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
    ])
    net_income_entries = _concept_units(facts, [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "ProfitLoss"),
    ])
    eps_entries = _concept_units(facts, [
        ("us-gaap", "EarningsPerShareDiluted"),
    ])
    equity_entries = _concept_units(facts, [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ])
    current_assets_entries = _concept_units(facts, [("us-gaap", "AssetsCurrent")])
    current_liabilities_entries = _concept_units(facts, [("us-gaap", "LiabilitiesCurrent")])
    debt_noncurrent_entries = _concept_units(facts, [
        ("us-gaap", "LongTermDebtNoncurrent"),
        ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"),
    ])
    debt_current_entries = _concept_units(facts, [
        ("us-gaap", "LongTermDebtCurrent"),
        ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsCurrent"),
        ("us-gaap", "CurrentPortionOfLongTermDebt"),
    ])
    shares_entries = _concept_units(facts, [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ])

    revenues = _annual_flow_records(revenue_entries, as_of_date)
    earnings = _annual_flow_records(net_income_entries, as_of_date)
    eps = _annual_flow_records(eps_entries, as_of_date)

    if not revenues or not earnings:
        raise RuntimeError(
            f"No hay suficientes datos anuales SEC point-in-time para {ticker} a {as_of_date}."
        )

    rev0 = revenues[0]
    ni0 = earnings[0]
    period_end = str(rev0.get("end"))
    filing_date = max(str(rev0.get("filed", "")), str(ni0.get("filed", "")))

    revenue = _value(rev0)
    net_income = _value(ni0)
    prior_revenue = _value(revenues[1]) if len(revenues) > 1 else None
    prior_net_income = _value(earnings[1]) if len(earnings) > 1 else None
    diluted_eps = _value(eps[0]) if eps else None

    equity = _value(_instant_record(equity_entries, as_of_date, period_end))
    current_assets = _value(_instant_record(current_assets_entries, as_of_date, period_end))
    current_liabilities = _value(_instant_record(current_liabilities_entries, as_of_date, period_end))
    debt_noncurrent = _value(_instant_record(debt_noncurrent_entries, as_of_date, period_end)) or 0.0
    debt_current = _value(_instant_record(debt_current_entries, as_of_date, period_end)) or 0.0
    total_debt = debt_noncurrent + debt_current

    shares_rec = _instant_record(shares_entries, as_of_date)
    shares = _value(shares_rec)
    market_cap = price * shares if shares else None

    return {
        "ticker": ticker,
        "source": "SEC Company Facts (annual 10-K, point-in-time)",
        "filing_date": filing_date or None,
        "report_period": period_end,
        "price_at_decision": price,
        "revenue": revenue,
        "net_income": net_income,
        "diluted_eps": diluted_eps,
        "equity": equity,
        "total_debt": total_debt if total_debt != 0 else None,
        "shares_outstanding": shares,
        "trailingPE": _safe_div(price, diluted_eps),
        "priceToBook": _safe_div(market_cap, equity),
        "returnOnEquity": _safe_div(net_income, equity),
        "profitMargins": _safe_div(net_income, revenue),
        "debtToEquity": _safe_div(total_debt, equity),
        "currentRatio": _safe_div(current_assets, current_liabilities),
        "revenueGrowth": (
            revenue / prior_revenue - 1.0
            if revenue is not None and prior_revenue not in (None, 0)
            else None
        ),
        "earningsGrowth": (
            net_income / prior_net_income - 1.0
            if net_income is not None and prior_net_income not in (None, 0)
            else None
        ),
    }


def fetch_historical_news(
    ticker: str,
    start_date: date,
    end_date: date,
    max_records: int = 12,
) -> dict[str, Any]:
    """Historical headlines bounded by the requested window via GDELT."""
    company = COMPANY_NAMES.get(ticker.upper(), ticker.upper())
    query = f'("{company}" OR {ticker.upper()})'
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": max_records,
        "format": "json",
        "sort": "HybridRel",
        "startdatetime": start_date.strftime("%Y%m%d000000"),
        "enddatetime": end_date.strftime("%Y%m%d235959"),
    }

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(GDELT_DOC_URL, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        _trace(f"GDELT no disponible para {ticker} {start_date}..{end_date}: {exc}")
        return {
            "source": "GDELT",
            "window_start": start_date.isoformat(),
            "window_end": end_date.isoformat(),
            "headlines": [],
            "articles": [],
            "error": str(exc),
        }

    articles = data.get("articles") or []
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for article in articles:
        title = str(article.get("title") or "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        cleaned.append({
            "title": title,
            "url": article.get("url"),
            "domain": article.get("domain"),
            "seendate": article.get("seendate"),
            "language": article.get("language"),
        })
        if len(cleaned) >= max_records:
            break

    headlines = [
        f"[{a.get('seendate') or ''}, {a.get('domain') or ''}] {a['title']}"
        for a in cleaned
    ]
    return {
        "source": "GDELT",
        "window_start": start_date.isoformat(),
        "window_end": end_date.isoformat(),
        "headlines": headlines,
        "articles": cleaned,
        "error": None,
    }
