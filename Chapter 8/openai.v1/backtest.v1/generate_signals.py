"""Generate auditable point-in-time multi-agent signals to CSV."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
OPENAI_V1_ROOT = HERE.parent
if str(OPENAI_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENAI_V1_ROOT))

load_dotenv(OPENAI_V1_ROOT / ".env")
load_dotenv(HERE / ".env", override=False)

from point_in_time_committee import parse_pm_decision, run_committee_as_of
from point_in_time_data import resolve_trading_date


CSV_FIELDS = [
    "generated_at_utc", "decision_date", "next_rebalance_date", "ticker",
    "strategy_mode", "action", "confidence", "size_pct", "final_decision",
    "price_at_decision", "fundamentals_source", "fundamentals_filing_date",
    "fundamentals_report_period", "trailing_pe", "price_to_book",
    "return_on_equity", "profit_margin", "debt_to_equity", "current_ratio",
    "revenue_growth", "earnings_growth", "technical_data_end", "sma_50",
    "sma_200", "rsi_14", "vol_30d_annualized", "news_source",
    "news_window_start", "news_window_end", "news_count", "headlines_json",
    "macro_data_end", "vix", "us_10y_yield_pct", "spx_1m_return_pct",
    "fundamentals_json", "technicals_json", "macro_json",
    "fundamentals_report", "technicals_report", "sentiment_report",
    "macro_report", "pm_thesis", "risk_verdict",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera señales point-in-time del comité y las guarda en CSV.")
    p.add_argument("--tickers", nargs="+", default=["TSLA", "MSFT", "NVDA"])
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default="2026-04-01")
    p.add_argument("--freq", choices=["monthly", "biweekly", "weekly"], default="monthly")
    p.add_argument("--output", default=str(HERE / "output" / "signals.csv"))
    p.add_argument("--show-reports", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _anchors(start: str, end: str, freq: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if freq == "weekly":
        return list(pd.date_range(start_ts, end_ts, freq="W-FRI"))
    if freq == "biweekly":
        return list(pd.date_range(start_ts, end_ts, freq="W-FRI"))[::2]
    return list(pd.date_range(start_ts, end_ts, freq="ME"))


def build_schedule(start: str, end: str, freq: str) -> list[date]:
    resolved: list[date] = []
    for anchor in _anchors(start, end, freq):
        d = resolve_trading_date(anchor.date())
        if d not in resolved:
            resolved.append(d)
    return resolved


def _fmt_pct(v: Any) -> str:
    return "N/D" if v is None else f"{float(v):.2%}"


def _fmt_num(v: Any, ndigits: int = 2) -> str:
    return "N/D" if v is None else f"{float(v):.{ndigits}f}"


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row.get("decision_date", ""), row.get("ticker", "").upper()))
    return keys


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _flatten(result: dict[str, Any], next_date: date | None) -> dict[str, Any]:
    f = result["fundamentals_data"]
    t = result["technicals_data"]
    s = result["sentiment_data"]
    m = result["macro_data"]

    action, confidence, size_pct = parse_pm_decision(result["pm_thesis"])
    if result["final_decision"].startswith("HOLD (risk"):
        action, size_pct = "HOLD", 0.0

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_date": result["as_of_date"],
        "next_rebalance_date": next_date.isoformat() if next_date else "",
        "ticker": result["ticker"],
        "strategy_mode": "long_only",
        "action": action,
        "confidence": confidence,
        "size_pct": size_pct,
        "final_decision": result["final_decision"],
        "price_at_decision": t.get("last_close"),
        "fundamentals_source": f.get("source"),
        "fundamentals_filing_date": f.get("filing_date"),
        "fundamentals_report_period": f.get("report_period"),
        "trailing_pe": f.get("trailingPE"),
        "price_to_book": f.get("priceToBook"),
        "return_on_equity": f.get("returnOnEquity"),
        "profit_margin": f.get("profitMargins"),
        "debt_to_equity": f.get("debtToEquity"),
        "current_ratio": f.get("currentRatio"),
        "revenue_growth": f.get("revenueGrowth"),
        "earnings_growth": f.get("earningsGrowth"),
        "technical_data_end": t.get("data_end"),
        "sma_50": t.get("sma_50"),
        "sma_200": t.get("sma_200"),
        "rsi_14": t.get("rsi_14"),
        "vol_30d_annualized": t.get("vol_30d_annualized"),
        "news_source": s.get("source"),
        "news_window_start": s.get("window_start"),
        "news_window_end": s.get("window_end"),
        "news_count": len(s.get("headlines") or []),
        "headlines_json": json.dumps(s.get("articles") or [], ensure_ascii=False),
        "macro_data_end": m.get("data_end"),
        "vix": m.get("vix"),
        "us_10y_yield_pct": m.get("us_10y_yield_pct"),
        "spx_1m_return_pct": m.get("spx_1m_return_pct"),
        "fundamentals_json": json.dumps(f, ensure_ascii=False),
        "technicals_json": json.dumps(t, ensure_ascii=False),
        "macro_json": json.dumps(m, ensure_ascii=False),
        "fundamentals_report": result["fundamentals_report"],
        "technicals_report": result["technicals_report"],
        "sentiment_report": result["sentiment_report"],
        "macro_report": result["macro_report"],
        "pm_thesis": result["pm_thesis"],
        "risk_verdict": result["risk_verdict"],
    }


def print_signal(result: dict[str, Any], row: dict[str, Any], show_reports: bool) -> None:
    f = result["fundamentals_data"]
    t = result["technicals_data"]
    s = result["sentiment_data"]
    m = result["macro_data"]

    print("\n" + "=" * 100)
    print(f"SEÑAL POINT-IN-TIME | {result['as_of_date']} | {result['ticker']} | siguiente={row['next_rebalance_date'] or 'N/D'}")
    print("=" * 100)
    print(
        "FUNDAMENTALES "
        f"(10-K presentado {f.get('filing_date')}, periodo {f.get('report_period')}): "
        f"P/E={_fmt_num(f.get('trailingPE'))} | P/B={_fmt_num(f.get('priceToBook'))} | "
        f"ROE={_fmt_pct(f.get('returnOnEquity'))} | Margen={_fmt_pct(f.get('profitMargins'))} | "
        f"D/E={_fmt_num(f.get('debtToEquity'))} | Current={_fmt_num(f.get('currentRatio'))} | "
        f"RevGrowth={_fmt_pct(f.get('revenueGrowth'))} | EarnGrowth={_fmt_pct(f.get('earningsGrowth'))}"
    )
    print(
        "TÉCNICO "
        f"(datos hasta {t.get('data_end')}): Close={_fmt_num(t.get('last_close'))} | "
        f"SMA50={_fmt_num(t.get('sma_50'))} | SMA200={_fmt_num(t.get('sma_200'))} | "
        f"RSI14={_fmt_num(t.get('rsi_14'))} | Vol30d={_fmt_pct(t.get('vol_30d_annualized'))}"
    )
    print(
        "NOTICIAS "
        f"{s.get('window_start')}..{s.get('window_end')} | fuente={s.get('source')} | "
        f"titulares={len(s.get('headlines') or [])}"
    )
    for i, headline in enumerate((s.get("headlines") or [])[:5], start=1):
        print(f"  {i}. {headline}")
    print(
        "MACRO "
        f"(datos hasta {m.get('data_end')}): VIX={_fmt_num(m.get('vix'))} | "
        f"US10Y={_fmt_num(m.get('us_10y_yield_pct'))}% | S&P1m={_fmt_num(m.get('spx_1m_return_pct'))}%"
    )
    print("-" * 100)
    print(
        f"DECISIÓN: {row['action']} | confianza={row['confidence']} | "
        f"size={float(row['size_pct']):.1f}% | final={row['final_decision']}"
    )

    if show_reports:
        print("\n--- Fundamental ---\n" + result["fundamentals_report"])
        print("\n--- Técnico ---\n" + result["technicals_report"])
        print("\n--- Sentimiento ---\n" + result["sentiment_report"])
        print("\n--- Macro ---\n" + result["macro_report"])
        print("\n--- Portfolio Manager ---\n" + result["pm_thesis"])
        print("\n--- Risk Officer ---\n" + result["risk_verdict"])


def main() -> None:
    args = parse_args()
    tickers = [t.upper() for t in args.tickers]
    schedule = build_schedule(args.start, args.end, args.freq)

    if not schedule:
        raise SystemExit("No hay fechas de rebalanceo en el intervalo.")

    runs = len(schedule) * len(tickers)
    print("=" * 100)
    print("PLAN DE GENERACIÓN DE SEÑALES POINT-IN-TIME")
    print("=" * 100)
    print(f"Tickers           : {', '.join(tickers)}")
    print(f"Intervalo         : {args.start} -> {args.end}")
    print(f"Frecuencia        : {args.freq}")
    print(f"Fechas reales     : {', '.join(d.isoformat() for d in schedule)}")
    print(f"Comités           : {runs}")
    print(f"Llamadas LLM aprox: {runs * 6} (6 por comité)")
    print(f"CSV               : {args.output}")
    print("Nota              : el CSV se escribe fila a fila y puede reanudarse.")
    print("=" * 100)

    if args.plan_only:
        return

    output = Path(args.output)
    if args.overwrite and output.exists():
        output.unlink()

    existing = _existing_keys(output)
    previous_date: date | None = None

    for i, decision_date in enumerate(schedule):
        next_date = schedule[i + 1] if i + 1 < len(schedule) else None
        news_start = (
            previous_date + timedelta(days=1)
            if previous_date is not None
            else decision_date - timedelta(days=30)
        )

        print("\n" + "#" * 100)
        print(f"REBALANCEO {i + 1}/{len(schedule)} | fecha={decision_date} | noticias={news_start}..{decision_date}")
        print("#" * 100)

        for ticker in tickers:
            key = (decision_date.isoformat(), ticker)
            if key in existing:
                print(f"[RESUME] {decision_date} {ticker}: ya existe en CSV, se omite.")
                continue

            result = run_committee_as_of(ticker, decision_date, news_start)
            row = _flatten(result, next_date)
            print_signal(result, row, args.show_reports)
            _append_row(output, row)
            existing.add(key)
            print(f"[CSV] guardado -> {output}")

        previous_date = decision_date

    print("\n" + "=" * 100)
    print(f"FIN. Señales guardadas en: {output}")
    print("Este CSV será la entrada del motor de backtest/benchmarks.")
    print("=" * 100)


if __name__ == "__main__":
    main()
