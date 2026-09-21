"""Parallel multi-company comparison — Chapter 7 extensions.v1.

This version keeps the chapter's parallelization idea, but produces a clean
cross-company comparison in Spanish instead of five independent, interleaved
agent reports.

Usage from Chapter 7:
    python "src/extensions.v1/parallel_comparison.py"
    python "src/extensions.v1/parallel_comparison.py" AAPL MSFT GOOGL NVDA JPM

Architecture:
    tickers
      -> concurrent FinanceToolkit ratio collection
      -> deterministic threshold scoring
      -> comparison table + strengths/weaknesses
      -> one final Spanish synthesis call

The final "strongest" company means strongest ONLY under this didactic
8-ratio framework. It is not an investment recommendation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from financetoolkit import Toolkit
from llama_index.llms.openai import OpenAI

# The folder name contains a dot ("extensions.v1"), so this file is executed
# directly instead of with "python -m". Add Chapter 7 to sys.path so imports
# from src work reliably.
CHAPTER_DIR = Path(__file__).resolve().parents[2]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from src.common import (  # noqa: E402
    FINANCIAL_MODELING_PREP_API_KEY,
    PROFITABILITY_THRESHOLDS,
    LIQUIDITY_THRESHOLDS,
)


# Keep FinanceToolkit's INFO chatter from burying the comparison output.
logging.getLogger("financetoolkit").setLevel(logging.WARNING)

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"]
COMPARATOR_MODEL = os.getenv("COMPARATOR_MODEL", "gpt-4.1-mini")

METRICS = [
    "Return on Assets",
    "Return on Equity",
    "Net Profit Margin",
    "Gross Margin",
    "Current Ratio",
    "Quick Ratio",
    "Debt-to-Equity Ratio",
    "Interest Coverage Ratio",
]

SPANISH_NAMES = {
    "Return on Assets": "ROA",
    "Return on Equity": "ROE",
    "Net Profit Margin": "Margen neto",
    "Gross Margin": "Margen bruto",
    "Current Ratio": "Ratio corriente",
    "Quick Ratio": "Ratio rápido",
    "Debt-to-Equity Ratio": "Deuda/Patrimonio",
    "Interest Coverage Ratio": "Cobertura intereses",
}

PERCENT_METRICS = {
    "Return on Assets",
    "Return on Equity",
    "Net Profit Margin",
    "Gross Margin",
}


def _as_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _latest_value(ratios: pd.DataFrame, metric: str) -> float | None:
    """Return the most recent finite value available for one metric."""
    if metric not in ratios.index:
        return None

    row = ratios.loc[metric]

    if isinstance(row, pd.DataFrame):
        values = row.stack().dropna().tolist()
    elif isinstance(row, pd.Series):
        values = row.dropna().tolist()
    else:
        values = [row]

    for value in reversed(values):
        number = _as_finite_float(value)
        if number is not None:
            return number
    return None


def _profitability_score(metric: str, value: float | None) -> int | None:
    if value is None:
        return None

    cfg = PROFITABILITY_THRESHOLDS[metric]
    healthy = cfg["healthy"]
    moderate = cfg["moderate"]

    if value >= healthy:
        return 9
    if value >= moderate:
        return 6
    return 3


def _liquidity_score(metric: str, value: float | None) -> int | None:
    if value is None:
        return None

    cfg = LIQUIDITY_THRESHOLDS[metric]

    if metric == "Current Ratio":
        if cfg["healthy_low"] <= value <= cfg["healthy_high"]:
            return 9
        if value >= cfg["warning"]:
            return 6
        return 3

    if metric == "Quick Ratio":
        return 9 if value >= cfg["healthy"] else 4

    if metric == "Debt-to-Equity Ratio":
        if value < 0:
            return 3
        if cfg["healthy_low"] <= value <= cfg["healthy_high"]:
            return 9
        if value <= cfg["warning"]:
            return 6
        return 3

    if metric == "Interest Coverage Ratio":
        if value >= cfg["healthy"]:
            return 9
        if value >= cfg["moderate"]:
            return 6
        return 3

    return None


def score_metric(metric: str, value: float | None) -> int | None:
    if metric in PROFITABILITY_THRESHOLDS:
        return _profitability_score(metric, value)
    if metric in LIQUIDITY_THRESHOLDS:
        return _liquidity_score(metric, value)
    return None


def _collect_company_sync(ticker: str) -> dict[str, Any]:
    """Fetch the real ratios for one company using FinanceToolkit/FMP."""
    companies = Toolkit(
        [ticker],
        api_key=FINANCIAL_MODELING_PREP_API_KEY,
        start_date="2022-01-01",
    )
    ratios = companies.ratios.collect_all_ratios()

    metrics = {metric: _latest_value(ratios, metric) for metric in METRICS}
    scores = {metric: score_metric(metric, value) for metric, value in metrics.items()}

    valid_scores = [score for score in scores.values() if score is not None]
    overall_score = (
        sum(valid_scores) / len(valid_scores)
        if valid_scores
        else None
    )

    strengths = [
        SPANISH_NAMES[metric]
        for metric, score in scores.items()
        if score is not None and score >= 9
    ]
    weaknesses = [
        SPANISH_NAMES[metric]
        for metric, score in scores.items()
        if score is not None and score <= 4
    ]

    return {
        "ticker": ticker,
        "metrics": metrics,
        "scores": scores,
        "overall_score": overall_score,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


async def collect_company(ticker: str) -> dict[str, Any]:
    """Run blocking FinanceToolkit work in a thread so tickers can overlap."""
    try:
        result = await asyncio.to_thread(_collect_company_sync, ticker)
        result["status"] = "success"
        return result
    except Exception as exc:
        return {
            "ticker": ticker,
            "status": "failed",
            "error": str(exc),
        }


async def collect_all(tickers: list[str]) -> list[dict[str, Any]]:
    print(f"Analizando {len(tickers)} compañías en paralelo...")
    print(f"Tickers: {', '.join(tickers)}\n")

    start = time.time()
    results = await asyncio.gather(
        *(collect_company(ticker) for ticker in tickers)
    )
    elapsed = time.time() - start

    print(f"Recogida de datos completada en {elapsed:.1f}s.\n")
    return results


def _format_value(metric: str, value: float | None) -> str:
    if value is None:
        return "N/D"
    if metric in PERCENT_METRICS:
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        return

    rows = []
    for result in successful:
        row = {"Ticker": result["ticker"]}
        for metric in METRICS:
            row[SPANISH_NAMES[metric]] = _format_value(
                metric,
                result["metrics"][metric],
            )
        row["Puntuación"] = (
            f"{result['overall_score']:.2f}/10"
            if result["overall_score"] is not None
            else "N/D"
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    print("=" * 120)
    print("TABLA COMPARATIVA — DATOS REALES MÁS RECIENTES DISPONIBLES")
    print("=" * 120)
    print(df.to_string(index=False))
    print()

    summary_rows = []
    for result in successful:
        summary_rows.append({
            "Ticker": result["ticker"],
            "Puntuación": (
                f"{result['overall_score']:.2f}/10"
                if result["overall_score"] is not None
                else "N/D"
            ),
            "Fortalezas": ", ".join(result["strengths"]) or "Ninguna destacada",
            "Debilidades": ", ".join(result["weaknesses"]) or "Ninguna grave",
        })

    summary_df = pd.DataFrame(summary_rows)
    print("=" * 120)
    print("RESUMEN DE FORTALEZAS Y DEBILIDADES")
    print("=" * 120)
    print(summary_df.to_string(index=False))
    print()


def print_ranking(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [
        r for r in results
        if r.get("status") == "success" and r.get("overall_score") is not None
    ]
    ranked = sorted(successful, key=lambda r: r["overall_score"], reverse=True)

    print("=" * 80)
    print("RANKING SEGÚN EL MARCO DIDÁCTICO DE 8 RATIOS")
    print("=" * 80)

    for position, result in enumerate(ranked, start=1):
        strengths = ", ".join(result["strengths"]) or "ninguna destacada"
        weaknesses = ", ".join(result["weaknesses"]) or "ninguna grave"
        print(
            f"{position}. {result['ticker']}: "
            f"{result['overall_score']:.2f}/10 | "
            f"Fortalezas: {strengths} | "
            f"Debilidades: {weaknesses}"
        )
    print()

    return ranked


def _serializable_payload(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for result in results:
        if result.get("status") != "success":
            continue

        payload.append({
            "ticker": result["ticker"],
            "overall_score": round(result["overall_score"], 3)
            if result["overall_score"] is not None
            else None,
            "metrics": {
                SPANISH_NAMES[metric]: (
                    round(value, 6) if value is not None else None
                )
                for metric, value in result["metrics"].items()
            },
            "scores": {
                SPANISH_NAMES[metric]: score
                for metric, score in result["scores"].items()
            },
            "strengths": result["strengths"],
            "weaknesses": result["weaknesses"],
        })
    return payload


async def spanish_comparative_analysis(
    results: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
) -> str:
    """Use one final LLM call only for a clear Spanish synthesis."""
    payload = _serializable_payload(results)

    if not payload or not ranked:
        return "No hay suficientes resultados válidos para realizar la comparación."

    strongest = ranked[0]["ticker"]

    prompt = f"""
Eres un analista financiero que debe explicar una comparación ya calculada.

DATOS ESTRUCTURADOS:
{json.dumps(payload, ensure_ascii=False, indent=2)}

La clasificación determinista coloca a {strongest} en primer lugar según una
media simple de ocho ratios. Esa puntuación ya está calculada: NO la cambies.

Escribe TODO en español y usa únicamente los datos proporcionados.

Quiero una salida clara con estas partes:
1. "Comparación general": explica las diferencias más importantes entre las
   compañías sin repetir toda la tabla.
2. "Fortalezas y debilidades": una línea breve por compañía.
3. "Compañía más fuerte según este marco": indica {strongest} y explica por qué,
   citando los ratios/puntuaciones concretos que sustentan esa conclusión.
4. "Matiz de comparabilidad": recuerda que comparar sectores distintos tiene
   limitaciones y que este ranking mide fortaleza financiera bajo estos ocho
   ratios, no atractivo bursátil ni recomendación de inversión.

Reglas estrictas:
- No inventes ratios, precios, crecimiento, noticias ni datos externos.
- No añadas métricas que no estén en DATOS ESTRUCTURADOS.
- No cambies el ranking calculado.
- Si un dato es null, di que no está disponible.
- Sé concreto y legible.
"""

    llm = OpenAI(model=COMPARATOR_MODEL, temperature=0)
    response = await llm.acomplete(prompt)
    return str(response)


async def main(tickers: list[str]) -> int:
    results = await collect_all(tickers)

    failures = [r for r in results if r.get("status") == "failed"]
    if failures:
        print("ERRORES DE DATOS:")
        for failure in failures:
            print(f"- {failure['ticker']}: {failure['error']}")
        print()

    print_comparison_table(results)
    ranked = print_ranking(results)

    if not ranked:
        print("No se pudo construir una comparación válida.")
        return 1

    print("=" * 80)
    print("ANÁLISIS COMPARATIVO EN ESPAÑOL")
    print("=" * 80)

    try:
        synthesis = await spanish_comparative_analysis(results, ranked)
        print(synthesis)
    except Exception as exc:
        # The deterministic table/ranking remains useful even if the final
        # language-model synthesis fails.
        print(f"No se pudo generar la síntesis con el LLM: {exc}")

    print()
    print(
        "Nota: la puntuación global es una media simple de 8 ratios y tiene "
        "fines didácticos. No equivale a una recomendación de inversión."
    )
    return 0


if __name__ == "__main__":
    tickers = [ticker.upper() for ticker in sys.argv[1:]] or DEFAULT_TICKERS
    raise SystemExit(asyncio.run(main(tickers)))
