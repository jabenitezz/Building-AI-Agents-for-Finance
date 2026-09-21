"""Point-in-time data layer for Chapter 8 OpenAI backtest.v1.

Goal: when the committee evaluates a historical date, every data source is
bounded by that date. Current/future news or current technical indicators are
never used as fallbacks.

Data sources:
- Prices/technicals/macro: yfinance historical series, cut at as_of_date.
- Fundamentals: SEC Company Facts, using only facts filed on/before as_of_date.
- News: Alpha Vantage NEWS_SENTIMENT, bounded by ticker and historical date window.

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
ALPHAVANTAGE_CACHE = CACHE_DIR / "alphavantage_news"
SEC_CACHE.mkdir(parents=True, exist_ok=True)
ALPHAVANTAGE_CACHE.mkdir(parents=True, exist_ok=True)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"

# Deterministic lexical aliases used to reject stories where a ticker is only
# tangentially tagged by the provider. Product aliases are intentionally
# specific; broad terms such as "AI" or "cloud" are excluded.
NEWS_ALIASES: dict[str, tuple[str, ...]] = {
    "NVDA": (
        "nvidia", "nvda", "geforce", "cuda", "blackwell", "hopper",
        "b200", "gb200", "h200", "h100",
    ),
    "MSFT": (
        "microsoft", "msft", "azure", "windows", "office 365",
        "microsoft 365", "github copilot", "xbox",
    ),
    "TSLA": (
        "tesla", "tsla", "cybertruck", "model 3", "model y", "model s",
        "model x", "supercharger", "full self-driving", "optimus",
    ),
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


def fetch_next_session_open(
    ticker: str,
    after_date: date,
    lookahead_days: int = 14,
) -> dict[str, Any]:
    """Return the first tradable session strictly AFTER after_date.

    The signal is assumed to be created after all information from
    `after_date` has been observed. Executing at the next session open avoids
    using late same-day news together with the same day's closing price.
    """
    start = after_date + timedelta(days=1)
    end = after_date + timedelta(days=lookahead_days)
    df = _history(ticker, start, end)
    if df.empty or "Open" not in df:
        raise RuntimeError(
            f"No se encontró una apertura posterior para {ticker} después de {after_date}."
        )

    valid = df["Open"].dropna()
    if valid.empty:
        raise RuntimeError(
            f"No hay precios de apertura válidos para {ticker} después de {after_date}."
        )

    first_ts = valid.index[0]
    return {
        "execution_date": first_ts.date().isoformat(),
        "execution_open_price": float(valid.iloc[0]),
        "execution_rule": "next_session_open",
    }


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


def _concept_entries(
    facts: dict[str, Any],
    namespace: str,
    concept: str,
) -> list[dict[str, Any]]:
    """Return SEC Company Facts entries for one exact taxonomy concept."""
    node = facts.get("facts", {}).get(namespace, {}).get(concept)
    if not node:
        return []
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


def _best_annual_series(
    facts: dict[str, Any],
    concepts: list[tuple[str, str]],
    as_of_date: date,
) -> tuple[list[dict[str, Any]], str | None]:
    """Choose the concept whose available annual series has the newest period.

    SEC taxonomy concepts can change over time. Choosing the *first concept that
    exists* can silently select a stale series (this happened with NVDA, where a
    legacy revenue concept ended in 2022 although newer annual facts existed
    under another concept).
    """
    best_records: list[dict[str, Any]] = []
    best_name: str | None = None
    best_key = ("", "")
    for namespace, concept in concepts:
        records = _annual_flow_records(
            _concept_entries(facts, namespace, concept),
            as_of_date,
        )
        if not records:
            continue
        key = (str(records[0].get("end", "")), str(records[0].get("filed", "")))
        if key > best_key:
            best_records = records
            best_name = f"{namespace}:{concept}"
            best_key = key
    return best_records, best_name


def _record_for_end(
    records: list[dict[str, Any]],
    period_end: str,
) -> dict[str, Any] | None:
    matches = [r for r in records if str(r.get("end")) == period_end]
    if not matches:
        return None
    return max(matches, key=lambda r: str(r.get("filed", "")))


def _prior_record(
    records: list[dict[str, Any]],
    period_end: str,
) -> dict[str, Any] | None:
    older = [r for r in records if str(r.get("end", "")) < period_end]
    if not older:
        return None
    return max(older, key=lambda r: str(r.get("end", "")))


def _best_instant_from_concepts(
    facts: dict[str, Any],
    concepts: list[tuple[str, str]],
    as_of_date: date,
    preferred_end: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Pick the newest matching instant fact across alternate SEC concepts."""
    best_record: dict[str, Any] | None = None
    best_name: str | None = None
    best_key = ("", "")
    for namespace, concept in concepts:
        rec = _instant_record(
            _concept_entries(facts, namespace, concept),
            as_of_date,
            preferred_end,
        )
        if rec is None:
            continue
        # With a preferred annual period, do not silently fall back to a
        # different period: mixing fiscal years corrupts ratios.
        if preferred_end is not None and str(rec.get("end")) != preferred_end:
            continue
        key = (str(rec.get("end", "")), str(rec.get("filed", "")))
        if key > best_key:
            best_record = rec
            best_name = f"{namespace}:{concept}"
            best_key = key
    return best_record, best_name


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
    """Conservative point-in-time fundamentals from SEC annual filings.

    All flow metrics are aligned to the SAME fiscal-year end. Alternate XBRL
    concepts are selected by recency, not by taxonomy-list order.
    """
    facts = _company_facts(ticker)

    revenue_concepts = [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
    ]
    net_income_concepts = [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "ProfitLoss"),
    ]
    eps_concepts = [("us-gaap", "EarningsPerShareDiluted")]
    equity_concepts = [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ]
    current_assets_concepts = [("us-gaap", "AssetsCurrent")]
    current_liabilities_concepts = [("us-gaap", "LiabilitiesCurrent")]
    debt_noncurrent_concepts = [
        ("us-gaap", "LongTermDebtNoncurrent"),
        ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"),
    ]
    debt_current_concepts = [
        ("us-gaap", "LongTermDebtCurrent"),
        ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsCurrent"),
        ("us-gaap", "CurrentPortionOfLongTermDebt"),
    ]
    shares_concepts = [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ]

    revenues, revenue_concept = _best_annual_series(
        facts, revenue_concepts, as_of_date
    )
    earnings, net_income_concept = _best_annual_series(
        facts, net_income_concepts, as_of_date
    )
    eps, eps_concept = _best_annual_series(facts, eps_concepts, as_of_date)

    if not revenues or not earnings:
        raise RuntimeError(
            f"No hay suficientes datos anuales SEC point-in-time para {ticker} a {as_of_date}."
        )

    # Revenue and net income MUST refer to the same fiscal year.
    revenue_ends = {str(r.get("end")) for r in revenues}
    earnings_ends = {str(r.get("end")) for r in earnings}
    common_periods = sorted(revenue_ends & earnings_ends, reverse=True)
    if not common_periods:
        raise RuntimeError(
            f"SEC sin periodo anual común de ingresos/beneficio para {ticker} a {as_of_date}."
        )

    period_end = common_periods[0]
    rev0 = _record_for_end(revenues, period_end)
    ni0 = _record_for_end(earnings, period_end)
    eps0 = _record_for_end(eps, period_end) if eps else None
    if rev0 is None or ni0 is None:
        raise RuntimeError(
            f"No se pudo alinear el periodo fiscal {period_end} para {ticker}."
        )

    period_end_date = _as_date(period_end)
    age_days = (as_of_date - period_end_date).days
    if age_days > 550:
        raise RuntimeError(
            f"Datos fundamentales SEC demasiado antiguos para {ticker}: "
            f"periodo={period_end}, as_of={as_of_date}, antigüedad={age_days} días."
        )

    prior_rev = _prior_record(revenues, period_end)
    prior_ni = _prior_record(earnings, period_end)

    filing_dates = [
        str(x.get("filed", ""))
        for x in (rev0, ni0, eps0)
        if x is not None and x.get("filed")
    ]
    filing_date = max(filing_dates) if filing_dates else None

    revenue = _value(rev0)
    net_income = _value(ni0)
    prior_revenue = _value(prior_rev)
    prior_net_income = _value(prior_ni)
    diluted_eps = _value(eps0)

    equity_rec, equity_concept = _best_instant_from_concepts(
        facts, equity_concepts, as_of_date, period_end
    )
    current_assets_rec, _ = _best_instant_from_concepts(
        facts, current_assets_concepts, as_of_date, period_end
    )
    current_liabilities_rec, _ = _best_instant_from_concepts(
        facts, current_liabilities_concepts, as_of_date, period_end
    )
    debt_noncurrent_rec, _ = _best_instant_from_concepts(
        facts, debt_noncurrent_concepts, as_of_date, period_end
    )
    debt_current_rec, _ = _best_instant_from_concepts(
        facts, debt_current_concepts, as_of_date, period_end
    )
    shares_rec, shares_concept = _best_instant_from_concepts(
        facts, shares_concepts, as_of_date, None
    )

    equity = _value(equity_rec)
    current_assets = _value(current_assets_rec)
    current_liabilities = _value(current_liabilities_rec)
    debt_noncurrent = _value(debt_noncurrent_rec)
    debt_current = _value(debt_current_rec)
    if debt_noncurrent is None and debt_current is None:
        total_debt = None
    else:
        total_debt = (debt_noncurrent or 0.0) + (debt_current or 0.0)

    shares = _value(shares_rec)
    market_cap = price * shares if shares else None

    profit_margin = _safe_div(net_income, revenue)
    roe = _safe_div(net_income, equity)

    # Catch obvious cross-period/taxonomy corruption before an LLM sees it.
    if profit_margin is not None and abs(profit_margin) > 1.5:
        raise RuntimeError(
            f"Margen fundamental no plausible para {ticker} ({profit_margin:.2%}); "
            "se aborta para evitar contaminar el backtest."
        )

    return {
        "ticker": ticker,
        "source": "SEC Company Facts (annual 10-K, point-in-time)",
        "filing_date": filing_date,
        "report_period": period_end,
        "report_age_days": age_days,
        "revenue_concept": revenue_concept,
        "net_income_concept": net_income_concept,
        "eps_concept": eps_concept,
        "equity_concept": equity_concept,
        "shares_concept": shares_concept,
        "price_at_decision": price,
        "revenue": revenue,
        "net_income": net_income,
        "diluted_eps": diluted_eps,
        "equity": equity,
        "total_debt": total_debt,
        "shares_outstanding": shares,
        "trailingPE": _safe_div(price, diluted_eps),
        "priceToBook": _safe_div(market_cap, equity),
        "returnOnEquity": roe,
        "profitMargins": profit_margin,
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


def _alphavantage_api_key() -> str:
    key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Falta ALPHAVANTAGE_API_KEY en Chapter 8/openai.v1/.env "
            "o backtest.v1/.env."
        )
    return key


def _ticker_sentiment_for(article: dict[str, Any], ticker: str) -> dict[str, Any] | None:
    """Return Alpha Vantage ticker-specific metadata for audit only.

    We store Alpha Vantage's own sentiment/relevance fields in the raw article
    record, but the LLM sentiment agent is intentionally fed the article text,
    not Alpha Vantage's sentiment label, so the experiment still evaluates our
    own agent.
    """
    for item in article.get("ticker_sentiment") or []:
        if str(item.get("ticker", "")).upper() == ticker.upper():
            return item
    return None


def _matched_aliases(text: str, ticker: str) -> list[str]:
    haystack = " ".join(str(text or "").casefold().split())
    matches: list[str] = []
    for alias in NEWS_ALIASES.get(ticker.upper(), (ticker.casefold(),)):
        if alias.casefold() in haystack:
            matches.append(alias)
    return matches


def _classify_news_directness(
    article: dict[str, Any],
    ticker: str,
    provider_relevance: float,
    title_min_relevance: float,
    summary_min_relevance: float,
) -> tuple[bool, str, list[str]]:
    """Decide whether a provider-tagged article is direct enough for the ticker.

    Rules:
    - ticker/company/product alias in TITLE: accept from the normal relevance floor.
    - alias only in SUMMARY: require a much higher provider relevance score.
    - no deterministic alias in either: reject, even if provider relevance is high.

    This deliberately favors precision over recall for backtesting: a missing
    sentiment item is safer than injecting unrelated news into a historical signal.
    """
    title_matches = _matched_aliases(str(article.get("title") or ""), ticker)
    if title_matches and provider_relevance >= title_min_relevance:
        return True, "title_match", title_matches

    summary_matches = _matched_aliases(str(article.get("summary") or ""), ticker)
    if summary_matches and provider_relevance >= summary_min_relevance:
        return True, "summary_match_high_relevance", summary_matches

    if title_matches or summary_matches:
        return False, "alias_match_but_relevance_too_low", title_matches + summary_matches
    return False, "no_ticker_alias", []


def fetch_historical_news(
    ticker: str,
    start_date: date,
    end_date: date,
    max_records: int = 12,
) -> dict[str, Any]:
    """Historical ticker news from Alpha Vantage NEWS_SENTIMENT.

    Provider relevance alone is not trusted. We inspect all raw candidates,
    require a deterministic ticker/company/product alias in title or summary,
    apply a stricter threshold to summary-only matches, rank the surviving
    candidates, and only then keep the best `max_records` items.
    """
    ticker = ticker.upper()
    title_min_relevance = float(
        os.getenv("ALPHAVANTAGE_MIN_RELEVANCE_SCORE", "0.20")
    )
    summary_min_relevance = float(
        os.getenv("ALPHAVANTAGE_SUMMARY_ONLY_MIN_RELEVANCE_SCORE", "0.85")
    )
    provider_limit = max(
        max_records,
        int(os.getenv("ALPHAVANTAGE_PROVIDER_NEWS_LIMIT", "50")),
    )
    provider_limit = min(provider_limit, 1000)

    # v3 changes only local filtering/ranking; the raw provider payload is still
    # safe to cache independently of the thresholds used on a given run.
    cache_path = ALPHAVANTAGE_CACHE / (
        f"v3_{ticker}_{start_date.isoformat()}_{end_date.isoformat()}_"
        f"raw{provider_limit}.json"
    )
    legacy_v2_cache = ALPHAVANTAGE_CACHE / (
        f"v2_{ticker}_{start_date.isoformat()}_{end_date.isoformat()}_"
        f"raw{provider_limit}.json"
    )

    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        _trace(
            f"Alpha Vantage cache HIT {ticker} {start_date}..{end_date} "
            f"raw_limit={provider_limit}"
        )
    elif legacy_v2_cache.exists():
        data = json.loads(legacy_v2_cache.read_text(encoding="utf-8"))
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        _trace(
            f"Alpha Vantage cache MIGRATED v2->v3 {ticker} "
            f"{start_date}..{end_date} raw_limit={provider_limit}"
        )
    else:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": start_date.strftime("%Y%m%dT0000"),
            "time_to": end_date.strftime("%Y%m%dT2359"),
            "sort": "RELEVANCE",
            "limit": provider_limit,
            "apikey": _alphavantage_api_key(),
        }

        timeout = float(os.getenv("ALPHAVANTAGE_TIMEOUT_SECONDS", "30"))
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.get(ALPHAVANTAGE_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            error_text = f"Alpha Vantage HTTP/JSON error: {exc}"
            _trace(
                f"Alpha Vantage no disponible para {ticker} "
                f"{start_date}..{end_date}: {error_text}"
            )
            return {
                "source": "Alpha Vantage NEWS_SENTIMENT",
                "window_start": start_date.isoformat(),
                "window_end": end_date.isoformat(),
                "headlines": [],
                "articles": [],
                "raw_count": 0,
                "eligible_count": 0,
                "relevant_count": 0,
                "filtered_out_count": 0,
                "title_relevance_threshold": title_min_relevance,
                "summary_relevance_threshold": summary_min_relevance,
                "error": error_text,
            }

        api_error = (
            data.get("Error Message")
            or data.get("Note")
            or data.get("Information")
        )
        if api_error:
            error_text = str(api_error)
            _trace(
                f"Alpha Vantage rechazó la consulta para {ticker} "
                f"{start_date}..{end_date}: {error_text}"
            )
            return {
                "source": "Alpha Vantage NEWS_SENTIMENT",
                "window_start": start_date.isoformat(),
                "window_end": end_date.isoformat(),
                "headlines": [],
                "articles": [],
                "raw_count": 0,
                "eligible_count": 0,
                "relevant_count": 0,
                "filtered_out_count": 0,
                "title_relevance_threshold": title_min_relevance,
                "summary_relevance_threshold": summary_min_relevance,
                "error": error_text,
            }

        if "feed" not in data:
            error_text = (
                "Respuesta Alpha Vantage sin campo 'feed'; "
                f"claves recibidas={sorted(data.keys())}"
            )
            _trace(error_text)
            return {
                "source": "Alpha Vantage NEWS_SENTIMENT",
                "window_start": start_date.isoformat(),
                "window_end": end_date.isoformat(),
                "headlines": [],
                "articles": [],
                "raw_count": 0,
                "eligible_count": 0,
                "relevant_count": 0,
                "filtered_out_count": 0,
                "title_relevance_threshold": title_min_relevance,
                "summary_relevance_threshold": summary_min_relevance,
                "error": error_text,
            }

        cache_path.write_text(json.dumps(data), encoding="utf-8")
        delay = float(os.getenv("ALPHAVANTAGE_REQUEST_DELAY_SECONDS", "1.0"))
        if delay > 0:
            time.sleep(delay)

    feed = data.get("feed") or []
    raw_count = len(feed)
    direct_candidates: list[dict[str, Any]] = []
    summary_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected_count = 0
    duplicate_or_empty_count = 0

    direct_target = max(
        1,
        int(os.getenv("ALPHAVANTAGE_DIRECT_NEWS_TARGET", "5")),
    )

    for article in feed:
        title = str(article.get("title") or "").strip()
        if not title or title.casefold() in seen:
            duplicate_or_empty_count += 1
            continue
        seen.add(title.casefold())

        ticker_meta = _ticker_sentiment_for(article, ticker)
        try:
            provider_relevance = float(
                ticker_meta.get("relevance_score")
                if ticker_meta is not None
                else 0.0
            )
        except (TypeError, ValueError):
            provider_relevance = 0.0

        accepted, match_scope, matched_aliases = _classify_news_directness(
            article,
            ticker,
            provider_relevance,
            title_min_relevance,
            summary_min_relevance,
        )
        if not accepted:
            rejected_count += 1
            continue

        published = str(article.get("time_published") or "").strip()
        cleaned_article = {
            "title": title,
            "url": article.get("url"),
            "source": article.get("source"),
            "source_domain": article.get("source_domain"),
            "time_published": published,
            "summary": article.get("summary"),
            "authors": article.get("authors") or [],
            "ticker_relevance_score": provider_relevance,
            "match_scope": match_scope,
            "matched_aliases": matched_aliases,
            # Provider sentiment is stored for audit only and is NOT passed as
            # a sentiment label to our LLM agent.
            "provider_ticker_sentiment_score": (
                ticker_meta.get("ticker_sentiment_score")
                if ticker_meta else None
            ),
            "provider_ticker_sentiment_label": (
                ticker_meta.get("ticker_sentiment_label")
                if ticker_meta else None
            ),
        }

        if match_scope == "title_match":
            direct_candidates.append(cleaned_article)
        else:
            summary_candidates.append(cleaned_article)

    def _news_rank(article: dict[str, Any]) -> tuple[float, str]:
        return (
            float(article.get("ticker_relevance_score") or 0.0),
            str(article.get("time_published") or ""),
        )

    direct_candidates.sort(key=_news_rank, reverse=True)
    summary_candidates.sort(key=_news_rank, reverse=True)

    # Title-first policy:
    # - If we already have at least N direct title matches, use ONLY direct news.
    # - Otherwise, keep all direct matches and use summary-only items merely to
    #   fill the set up to N items (never to pad toward max_records).
    if len(direct_candidates) >= direct_target:
        cleaned = direct_candidates[:max_records]
        fallback_used_count = 0
    else:
        needed = min(max_records, direct_target) - len(direct_candidates)
        fallback = summary_candidates[:max(0, needed)]
        cleaned = (direct_candidates + fallback)[:max_records]
        fallback_used_count = len(fallback)

    direct_count = len(direct_candidates)
    summary_candidate_count = len(summary_candidates)
    eligible_count = direct_count + summary_candidate_count

    headlines = [
        (
            f"[{a.get('time_published') or ''}, {a.get('source') or ''}, "
            f"relevance={a.get('ticker_relevance_score'):.3f}, "
            f"match={a.get('match_scope')}] {a['title']}"
        )
        for a in cleaned
    ]

    _trace(
        f"Alpha Vantage {ticker}: raw={raw_count} direct={direct_count} "
        f"summary_candidates={summary_candidate_count} final={len(cleaned)} "
        f"fallback_used={fallback_used_count} rejected={rejected_count} "
        f"dup/empty={duplicate_or_empty_count} title_min={title_min_relevance:.2f} "
        f"summary_min={summary_min_relevance:.2f} direct_target={direct_target}"
    )

    return {
        "source": "Alpha Vantage NEWS_SENTIMENT",
        "window_start": start_date.isoformat(),
        "window_end": end_date.isoformat(),
        "headlines": headlines,
        "articles": cleaned,
        "raw_count": raw_count,
        "eligible_count": eligible_count,
        "direct_count": direct_count,
        "summary_candidate_count": summary_candidate_count,
        "fallback_used_count": fallback_used_count,
        "direct_target": direct_target,
        "relevant_count": len(cleaned),
        "filtered_out_count": rejected_count,
        "duplicate_or_empty_count": duplicate_or_empty_count,
        "title_relevance_threshold": title_min_relevance,
        "summary_relevance_threshold": summary_min_relevance,
        "relevance_threshold": title_min_relevance,
        "error": None,
    }

