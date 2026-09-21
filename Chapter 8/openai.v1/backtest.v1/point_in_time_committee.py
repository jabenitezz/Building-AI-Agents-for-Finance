"""Point-in-time Investment Committee for backtest.v1."""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

HERE = Path(__file__).resolve().parent
OPENAI_V1_ROOT = HERE.parent
if str(OPENAI_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENAI_V1_ROOT))

load_dotenv(OPENAI_V1_ROOT / ".env")
load_dotenv(HERE / ".env", override=False)

from models import (
    make_haiku_equivalent,
    make_opus_equivalent,
    make_sonnet_equivalent,
    message_text,
    trace,
)
from point_in_time_data import (
    fetch_fundamentals,
    fetch_historical_news,
    fetch_macro,
    fetch_technicals,
)

premium_model = make_opus_equivalent(max_tokens=2000)
balanced_model = make_sonnet_equivalent(max_tokens=1500)
fast_model = make_haiku_equivalent(max_tokens=800)


class PITCommitteeState(TypedDict):
    ticker: str
    as_of_date: str
    news_start_date: str
    fundamentals_data: dict[str, Any]
    technicals_data: dict[str, Any]
    sentiment_data: dict[str, Any]
    macro_data: dict[str, Any]
    fundamentals_report: str
    technicals_report: str
    sentiment_report: str
    macro_report: str
    pm_thesis: str
    risk_verdict: str
    final_decision: str


def _d(value: str) -> date:
    return date.fromisoformat(value)


def fundamentals_analyst(state: PITCommitteeState) -> dict:
    ticker = state["ticker"]
    as_of = _d(state["as_of_date"])
    price = state["technicals_data"]["last_close"]
    trace("pit/fundamentals", f"START {ticker} as_of={as_of}")
    data = fetch_fundamentals(ticker, as_of, price)
    trace(
        "pit/fundamentals",
        f"DATA filing={data.get('filing_date')} period={data.get('report_period')} "
        f"PE={data.get('trailingPE')} ROE={data.get('returnOnEquity')}",
    )
    sys = SystemMessage(content=(
        "Eres un analista fundamental senior. Estás realizando un backtest histórico. "
        "NO conoces información posterior a la fecha indicada. Los datos proceden del "
        "último 10-K que ya había sido presentado a la SEC en esa fecha. Escribe TODO "
        "en español, máximo 160 palabras. Evalúa valoración, rentabilidad, balance y "
        "crecimiento. No inventes datos ni utilices conocimiento posterior. Concluye "
        "con BULLISH / NEUTRAL / BEARISH en fundamentales."
    ))
    msg = HumanMessage(content=(
        f"Fecha histórica: {as_of}\nTicker: {ticker}\n"
        f"Datos point-in-time: {data}"
    ))
    out = balanced_model.invoke([sys, msg])
    report = message_text(out)
    trace("pit/fundamentals", f"DONE preview={report[:150]!r}")
    return {"fundamentals_data": data, "fundamentals_report": report}


def technicals_analyst(state: PITCommitteeState) -> dict:
    ticker = state["ticker"]
    as_of = _d(state["as_of_date"])
    data = state["technicals_data"]
    trace("pit/technicals", f"START {ticker} as_of={as_of} data_end={data.get('data_end')}")
    sys = SystemMessage(content=(
        "Eres un analista técnico en un backtest histórico. Usa EXCLUSIVAMENTE los "
        "indicadores calculados con precios disponibles hasta la fecha histórica. "
        "Escribe TODO en español, máximo 120 palabras. Evalúa tendencia frente a "
        "SMA50/SMA200, RSI y volatilidad. No uses información posterior. Concluye "
        "BULLISH / NEUTRAL / BEARISH en análisis técnico."
    ))
    msg = HumanMessage(content=(
        f"Fecha histórica: {as_of}\nTicker: {ticker}\nIndicadores: {data}"
    ))
    out = balanced_model.invoke([sys, msg])
    report = message_text(out)
    trace("pit/technicals", f"DONE preview={report[:150]!r}")
    return {"technicals_report": report}


def sentiment_analyst(state: PITCommitteeState) -> dict:
    ticker = state["ticker"]
    start = _d(state["news_start_date"])
    end = _d(state["as_of_date"])
    trace("pit/sentiment", f"START {ticker} window={start}..{end}")
    data = fetch_historical_news(ticker, start, end)
    if data.get("error"):
        raise RuntimeError(
            f"Fallo operativo recuperando noticias históricas para {ticker} "
            f"{start}..{end}: {data['error']}. "
            "No se genera una señal degradada; reanuda la ejecución más tarde."
        )
    headlines = data.get("headlines") or []
    trace(
        "pit/sentiment",
        f"DATA source={data.get('source')} raw={data.get('raw_count')} "
        f"eligible={data.get('eligible_count')} final={len(headlines)} "
        f"rejected={data.get('filtered_out_count')} "
        f"title_min={data.get('title_relevance_threshold')} "
        f"summary_min={data.get('summary_relevance_threshold')}",
    )

    prompt_items: list[str] = []
    for article in data.get("articles") or []:
        summary = " ".join(str(article.get("summary") or "").split())
        if len(summary) > 420:
            summary = summary[:417] + "..."
        prompt_items.append(
            f"[{article.get('time_published') or ''}, "
            f"{article.get('source') or ''}, "
            f"relevance={article.get('ticker_relevance_score')}, "
            f"match={article.get('match_scope')}] "
            f"{article.get('title') or ''}"
            + (f" | Resumen: {summary}" if summary else "")
        )

    body = "\n- ".join(prompt_items) if prompt_items else "(sin noticias históricas relevantes disponibles)"
    sys = SystemMessage(content=(
        "Eres un analista de sentimiento en un backtest histórico. Solo puedes usar "
        "las noticias suministradas, todas acotadas a la ventana indicada y filtradas "
        "por relevancia directa para el ticker. Cada elemento puede incluir título y "
        "un resumen breve. Si no hay noticias relevantes, dilo explícitamente y usa "
        "MIXED/insuficiente en vez de inventar información. No utilices el sentimiento "
        "calculado por el proveedor aunque exista en los datos de auditoría. Escribe "
        "TODO en español, máximo 120 palabras. Concluye POSITIVE / MIXED / NEGATIVE."
    ))
    msg = HumanMessage(content=(
        f"Ticker: {ticker}\nVentana histórica: {start}..{end}\nNoticias filtradas:\n- {body}"
    ))
    out = fast_model.invoke([sys, msg])
    report = message_text(out)
    trace("pit/sentiment", f"DONE preview={report[:150]!r}")
    return {"sentiment_data": data, "sentiment_report": report}


def macro_analyst(state: PITCommitteeState) -> dict:
    as_of = _d(state["as_of_date"])
    trace("pit/macro", f"START as_of={as_of}")
    data = fetch_macro(as_of)
    trace("pit/macro", f"DATA {data}")
    sys = SystemMessage(content=(
        "Eres un estratega macro en un backtest histórico. Los datos están cortados "
        "en la fecha indicada. Escribe TODO en español, máximo 120 palabras. Evalúa "
        "risk-on/risk-off/neutral a partir de VIX, Treasury 10Y y S&P 500 a un mes. "
        "No utilices información posterior."
    ))
    msg = HumanMessage(content=f"Fecha histórica: {as_of}\nMacro: {data}")
    out = balanced_model.invoke([sys, msg])
    report = message_text(out)
    trace("pit/macro", f"DONE preview={report[:150]!r}")
    return {"macro_data": data, "macro_report": report}


def portfolio_manager(state: PITCommitteeState) -> dict:
    trace("pit/portfolio_manager", f"START {state['ticker']} as_of={state['as_of_date']}")
    sys = SystemMessage(content=(
        "Eres el Portfolio Manager de un backtest histórico LONG-ONLY. La señal se "
        "genera DESPUÉS de terminar el día histórico indicado y puede usar toda la "
        "información disponible durante ese día. Cualquier operación se ejecutará en "
        "la APERTURA DE LA SIGUIENTE SESIÓN bursátil, nunca al cierre del mismo día. "
        "No uses hechos posteriores a la fecha indicada. Sintetiza los cuatro informes "
        "en español, máximo 250 palabras. "
        "Termina ESTRICTAMENTE con:\n"
        "ACTION=<BUY|HOLD|SELL>; CONFIDENCE=<0-100>; SIZE_PCT=<0.0-5.0>\n"
        "BUY abre/aumenta un largo; HOLD no añade; SELL significa salir/no mantener "
        "un largo, NO abrir un short. BUY debe tener SIZE_PCT normalmente entre "
        "0.5 y 3.0; HOLD y SELL deben tener SIZE_PCT=0.0."
    ))
    msg = HumanMessage(content=(
        f"Fecha de señal EOD: {state['as_of_date']}\n"
        f"Regla de ejecución: apertura de la siguiente sesión\n"
        f"Ticker: {state['ticker']}\n\n"
        f"--- Fundamental ---\n{state['fundamentals_report']}\n\n"
        f"--- Técnico ---\n{state['technicals_report']}\n\n"
        f"--- Sentimiento ---\n{state['sentiment_report']}\n\n"
        f"--- Macro ---\n{state['macro_report']}"
    ))
    out = premium_model.invoke([sys, msg])
    thesis = message_text(out)
    trace("pit/portfolio_manager", f"DONE preview={thesis[:180]!r}")
    return {"pm_thesis": thesis}


_ACTION_RE = re.compile(r"ACTION\s*=\s*(BUY|SELL|HOLD)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE\s*=\s*(\d+)", re.IGNORECASE)
_SIZE_RE = re.compile(r"SIZE_PCT\s*=\s*([\d.]+)", re.IGNORECASE)


def parse_pm_decision(thesis: str) -> tuple[str, int, float]:
    action_m = _ACTION_RE.search(thesis)
    conf_m = _CONF_RE.search(thesis)
    size_m = _SIZE_RE.search(thesis)
    action = action_m.group(1).upper() if action_m else "HOLD"
    confidence = int(conf_m.group(1)) if conf_m else 0
    size_pct = float(size_m.group(1)) if size_m else 0.0
    if action in {"HOLD", "SELL"}:
        size_pct = 0.0
    return action, confidence, size_pct


def risk_officer(state: PITCommitteeState) -> dict:
    action, confidence, size_pct = parse_pm_decision(state["pm_thesis"])
    vol = state["technicals_data"].get("vol_30d_annualized")
    trace("pit/risk_officer", f"START action={action} confidence={confidence} size={size_pct:.1f}% vol={vol}")

    if vol is not None and vol > 0.60 and action == "BUY":
        verdict = f"VETO: volatilidad anualizada 30d {vol:.1%} supera 60%."
        trace("pit/risk_officer", "HARD VETO volatility > 60%")
        return {"risk_verdict": verdict, "final_decision": "HOLD (risk veto)"}

    sys = SystemMessage(content=(
        "Eres el Chief Risk Officer en un backtest histórico LONG-ONLY. Rechaza un BUY "
        "si ignora vientos macro claros, SIZE_PCT > 3.0, o CONFIDENCE < 50. "
        "HOLD/SELL con SIZE_PCT=0 son coherentes. Responde ESTRICTAMENTE: "
        "VERDICT=<APPROVED|REJECTED>; REASON=<explicación breve en español>."
    ))
    msg = HumanMessage(content=(
        f"Fecha: {state['as_of_date']}\nTesis PM:\n{state['pm_thesis']}\n"
        f"Volatilidad 30d anualizada: {vol}"
    ))
    out = premium_model.invoke([sys, msg])
    verdict = message_text(out)
    final = "HOLD (risk rejected)" if "REJECTED" in verdict.upper() else action
    trace("pit/risk_officer", f"DONE final={final} verdict={verdict!r}")
    return {"risk_verdict": verdict, "final_decision": final}


def build_committee():
    g = StateGraph(PITCommitteeState)
    g.add_node("fundamentals", fundamentals_analyst)
    g.add_node("technicals", technicals_analyst)
    g.add_node("sentiment", sentiment_analyst)
    g.add_node("macro", macro_analyst)
    g.add_node("portfolio_manager", portfolio_manager)
    g.add_node("risk_officer", risk_officer)

    for analyst in ("fundamentals", "technicals", "sentiment", "macro"):
        g.add_edge(START, analyst)
        g.add_edge(analyst, "portfolio_manager")
    g.add_edge("portfolio_manager", "risk_officer")
    g.add_edge("risk_officer", END)
    return g.compile()


def run_committee_as_of(ticker: str, as_of_date: date, news_start_date: date) -> PITCommitteeState:
    ticker = ticker.upper()
    trace("pit/committee", f"START ticker={ticker} as_of={as_of_date} news={news_start_date}..{as_of_date}")
    technicals = fetch_technicals(ticker, as_of_date)

    initial: PITCommitteeState = {
        "ticker": ticker,
        "as_of_date": as_of_date.isoformat(),
        "news_start_date": news_start_date.isoformat(),
        "fundamentals_data": {},
        "technicals_data": technicals,
        "sentiment_data": {},
        "macro_data": {},
        "fundamentals_report": "",
        "technicals_report": "",
        "sentiment_report": "",
        "macro_report": "",
        "pm_thesis": "",
        "risk_verdict": "",
        "final_decision": "",
    }
    result = build_committee().invoke(initial)
    trace("pit/committee", f"END ticker={ticker} as_of={as_of_date} final={result['final_decision']}")
    return result
