"""Point-in-time committee + adversarial validation pipeline for backtest.v2.

Design goals
------------
- Preserve the Chapter 8 Portfolio Manager sizing contract:
  SIZE_PCT is a target percentage of total portfolio NAV.
- Keep Bull and Bear equally capable: same model tier, same reasoning effort,
  same max_tokens. Their only asymmetry is the role prompt.
- Use the debate as a validation layer, not as a second portfolio manager.
  The Judge can APPROVE or REJECT a BUY proposal but never resizes it.
- Apply deterministic hard-risk rules before paying for the debate.
- Keep a committee-only baseline alongside the adversarial result so the
  backtest can measure whether the extra layer adds value.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from models import (
    make_opus_equivalent,
    make_sonnet_equivalent,
    message_text,
    trace,
)
from point_in_time_committee import (
    PITCommitteeState,
    fundamentals_analyst,
    macro_analyst,
    parse_pm_decision,
    portfolio_manager,
    sentiment_analyst,
    technicals_analyst,
)
from point_in_time_data import fetch_technicals


# Bull and Bear MUST be equally capable. Two independent client instances use
# exactly the same model factory and configuration.
bull_model = make_opus_equivalent(max_tokens=1500)
bear_model = make_opus_equivalent(max_tokens=1500)

# Devil is a critic/summariser; Judge and qualitative risk are final gates.
devil_model = make_sonnet_equivalent(max_tokens=1200)
judge_model = make_opus_equivalent(max_tokens=1500)
risk_model = make_opus_equivalent(max_tokens=900)


class DebateState(TypedDict):
    ticker: str
    as_of_date: str
    fundamentals_report: str
    technicals_report: str
    sentiment_report: str
    macro_report: str
    thesis_narrative: str
    pm_thesis: str
    pm_action: str
    pm_confidence: int
    pm_size_pct: float
    bull_case: str
    bear_case: str
    devil_critique: str
    judge_verdict: str
    judge_decision: str
    judge_confidence: int
    judge_rationale: str


_CONTROL_LINE_RE = re.compile(
    r"^\s*ACTION\s*=\s*(BUY|HOLD|SELL)\s*;\s*"
    r"CONFIDENCE\s*=\s*\d+\s*;\s*"
    r"SIZE_PCT\s*=\s*[\d.]+\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_JUDGE_DECISION_RE = re.compile(
    r"VERDICT\s*=\s*(APPROVED|REJECTED)", re.IGNORECASE
)
_JUDGE_CONF_RE = re.compile(r"CONFIDENCE\s*=\s*(\d+)", re.IGNORECASE)
_JUDGE_RATIONALE_RE = re.compile(
    r"RATIONALE\s*=\s*(.*)", re.IGNORECASE | re.DOTALL
)


def _strip_pm_control_line(thesis: str) -> str:
    """Remove ACTION/CONFIDENCE/SIZE_PCT before Bull/Bear see the thesis."""
    return _CONTROL_LINE_RE.sub("", thesis).strip()


def build_thesis_committee():
    """Four analysts -> Portfolio Manager, deliberately stopping before risk."""
    g = StateGraph(PITCommitteeState)
    g.add_node("fundamentals", fundamentals_analyst)
    g.add_node("technicals", technicals_analyst)
    g.add_node("sentiment", sentiment_analyst)
    g.add_node("macro", macro_analyst)
    g.add_node("portfolio_manager", portfolio_manager)

    for analyst in ("fundamentals", "technicals", "sentiment", "macro"):
        g.add_edge(START, analyst)
        g.add_edge(analyst, "portfolio_manager")
    g.add_edge("portfolio_manager", END)
    return g.compile()


def run_thesis_as_of(
    ticker: str,
    as_of_date: date,
    news_start_date: date,
) -> PITCommitteeState:
    ticker = ticker.upper()
    trace(
        "pit-v2/thesis",
        f"START ticker={ticker} as_of={as_of_date} "
        f"news={news_start_date}..{as_of_date}",
    )
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
    result = build_thesis_committee().invoke(initial)
    trace("pit-v2/thesis", f"DONE ticker={ticker}")
    return result


def hard_risk_precheck(
    action: str,
    confidence: int,
    size_pct: float,
    vol_30d_annualized: float | None,
) -> tuple[str, str]:
    """Deterministic rules inherited from the original Chapter 8 risk policy."""
    if vol_30d_annualized is not None and vol_30d_annualized > 0.60:
        return (
            "REJECTED",
            f"Volatilidad anualizada 30d {vol_30d_annualized:.1%} supera 60%.",
        )
    if size_pct > 3.0:
        return (
            "REJECTED",
            f"SIZE_PCT={size_pct:.1f}% supera el límite firm-wide de 3.0%.",
        )
    if confidence < 50:
        return (
            "REJECTED",
            f"CONFIDENCE={confidence} es inferior al mínimo de 50.",
        )
    return ("APPROVED", "Supera las reglas deterministas de riesgo.")


def _evidence_block(state: DebateState) -> str:
    return (
        f"Fecha histórica: {state['as_of_date']}\n"
        f"Ticker: {state['ticker']}\n\n"
        f"--- Fundamentales ---\n{state['fundamentals_report']}\n\n"
        f"--- Técnicos ---\n{state['technicals_report']}\n\n"
        f"--- Sentimiento ---\n{state['sentiment_report']}\n\n"
        f"--- Macro ---\n{state['macro_report']}\n\n"
        f"--- Tesis narrativa del PM, sin línea ACTION/CONFIDENCE/SIZE_PCT ---\n"
        f"{state['thesis_narrative']}"
    )


def _side_system(side: str) -> str:
    if side == "BULL":
        stance = "A FAVOR DE entrar en una posición larga"
        failure = "qué tendría que ocurrir para invalidar el caso alcista"
        label = "BULL_CONVICTION"
    else:
        stance = "EN CONTRA DE entrar en una posición larga"
        failure = "qué tendría que ocurrir para invalidar el caso bajista"
        label = "BEAR_CONVICTION"

    return (
        f"Eres el analista adversarial {side}. Estás en un backtest histórico y solo "
        f"puedes usar la evidencia proporcionada. Tu tarea es construir el caso más "
        f"fuerte posible {stance}. Bull y Bear usan EXACTAMENTE el mismo modelo y la "
        f"misma potencia; la diferencia es únicamente el rol. Ignora cualquier "
        f"recomendación direccional que pudiera aparecer incidentalmente en la narrativa "
        f"del PM y razona desde la evidencia. No inventes hechos ni uses conocimiento "
        f"posterior a la fecha histórica.\n\n"
        f"Responde en español y con esta estructura simétrica:\n"
        f"1. ARGUMENTO_1=<afirmación concreta + evidencia>\n"
        f"2. ARGUMENTO_2=<afirmación concreta + evidencia>\n"
        f"3. ARGUMENTO_3=<afirmación concreta + evidencia>\n"
        f"SUPUESTO_CLAVE=<principal supuesto del caso>\n"
        f"CONDICION_DE_FALLO=<{failure}>\n"
        f"{label}=<0-100>\n"
        f"Máximo 260 palabras."
    )


def bull_node(state: DebateState) -> dict:
    trace("pit-v2/debate/bull", "START same-model symmetric debate")
    out = bull_model.invoke(
        [
            SystemMessage(content=_side_system("BULL")),
            HumanMessage(content=_evidence_block(state)),
        ]
    )
    value = message_text(out)
    trace("pit-v2/debate/bull", f"DONE preview={value[:140]!r}")
    return {"bull_case": value}


def bear_node(state: DebateState) -> dict:
    trace("pit-v2/debate/bear", "START same-model symmetric debate")
    out = bear_model.invoke(
        [
            SystemMessage(content=_side_system("BEAR")),
            HumanMessage(content=_evidence_block(state)),
        ]
    )
    value = message_text(out)
    trace("pit-v2/debate/bear", f"DONE preview={value[:140]!r}")
    return {"bear_case": value}


def devil_node(state: DebateState) -> dict:
    trace("pit-v2/debate/devil", "START")
    sys = SystemMessage(
        content=(
            "Eres el Abogado del Diablo en un backtest histórico. No debes elegir BULL "
            "ni BEAR. Tu trabajo es atacar la calidad del razonamiento de AMBOS lados "
            "contra la evidencia original. Identifica la afirmación más débil del Bull, "
            "la afirmación más débil del Bear, cualquier supuesto sin suficiente apoyo "
            "y una premisa compartida que pueda hacer fallar a ambos. No inventes hechos. "
            "Responde en español, máximo 190 palabras, con:\n"
            "WEAK_BULL=<...>\n"
            "WEAK_BEAR=<...>\n"
            "UNSUPPORTED_ASSUMPTION=<...>\n"
            "SHARED_RISK=<...>"
        )
    )
    msg = HumanMessage(
        content=(
            f"{_evidence_block(state)}\n\n"
            f"--- Caso BULL ---\n{state['bull_case']}\n\n"
            f"--- Caso BEAR ---\n{state['bear_case']}"
        )
    )
    out = devil_model.invoke([sys, msg])
    value = message_text(out)
    trace("pit-v2/debate/devil", f"DONE preview={value[:140]!r}")
    return {"devil_critique": value}


def _parse_judge(value: str) -> tuple[str, int, str]:
    dm = _JUDGE_DECISION_RE.search(value)
    cm = _JUDGE_CONF_RE.search(value)
    rm = _JUDGE_RATIONALE_RE.search(value)
    decision = dm.group(1).upper() if dm else "REJECTED"
    confidence = int(cm.group(1)) if cm else 0
    rationale = rm.group(1).strip() if rm else value.strip()
    return decision, confidence, rationale


def judge_node(state: DebateState) -> dict:
    trace("pit-v2/debate/judge", "START")
    sys = SystemMessage(
        content=(
            "Eres un Juez NEUTRAL que valida una tesis de inversión ya dimensionada por "
            "el Portfolio Manager. NO eres un segundo Portfolio Manager: no puedes "
            "cambiar ACTION, CONFIDENCE ni SIZE_PCT y no puedes proponer una posición "
            "alternativa. Tu única tarea es decidir si la propuesta BUY sobrevive al "
            "stress-test adversarial.\n\n"
            "APPROVED si la tesis central sigue apoyada por la evidencia después de "
            "considerar el mejor caso Bear y la crítica del Abogado del Diablo. "
            "REJECTED si una premisa central carece de apoyo, existe una contradicción "
            "material no resuelta o la evidencia es insuficiente para sostener el BUY. "
            "La mera existencia de riesgos no implica REJECTED.\n\n"
            "Responde ESTRICTAMENTE:\n"
            "VERDICT=<APPROVED|REJECTED>\n"
            "CONFIDENCE=<0-100>\n"
            "RATIONALE=<2-4 frases en español>"
        )
    )
    msg = HumanMessage(
        content=(
            f"Fecha histórica: {state['as_of_date']}\nTicker: {state['ticker']}\n\n"
            f"Propuesta del PM:\n{state['pm_thesis']}\n\n"
            f"--- Fundamentales ---\n{state['fundamentals_report']}\n\n"
            f"--- Técnicos ---\n{state['technicals_report']}\n\n"
            f"--- Sentimiento ---\n{state['sentiment_report']}\n\n"
            f"--- Macro ---\n{state['macro_report']}\n\n"
            f"--- BULL ---\n{state['bull_case']}\n\n"
            f"--- BEAR ---\n{state['bear_case']}\n\n"
            f"--- Abogado del Diablo ---\n{state['devil_critique']}"
        )
    )
    out = judge_model.invoke([sys, msg])
    verdict = message_text(out)
    decision, confidence, rationale = _parse_judge(verdict)
    trace("pit-v2/debate/judge", f"DONE decision={decision} confidence={confidence}")
    return {
        "judge_verdict": verdict,
        "judge_decision": decision,
        "judge_confidence": confidence,
        "judge_rationale": rationale,
    }


def build_debate():
    g = StateGraph(DebateState)
    g.add_node("bull", bull_node)
    g.add_node("bear", bear_node)
    g.add_node("devil", devil_node)
    g.add_node("judge", judge_node)
    g.add_edge(START, "bull")
    g.add_edge(START, "bear")
    g.add_edge("bull", "devil")
    g.add_edge("bear", "devil")
    g.add_edge("devil", "judge")
    g.add_edge("judge", END)
    return g.compile()


def qualitative_risk_review(
    *,
    ticker: str,
    as_of_date: str,
    pm_thesis: str,
    macro_report: str,
) -> tuple[str, str]:
    """Final qualitative risk gate after the adversarial review."""
    trace("pit-v2/risk", "START qualitative macro-risk review")
    sys = SystemMessage(
        content=(
            "Eres el Chief Risk Officer en un backtest histórico LONG-ONLY. Las reglas "
            "duras de volatilidad, SIZE_PCT y CONFIDENCE ya se comprobaron de forma "
            "determinista. Revisa únicamente el riesgo cualitativo original pendiente: "
            "REJECTED si el Portfolio Manager recomienda BUY pero su tesis ignora o "
            "minimiza de forma material vientos macro en contra que aparecen en el "
            "informe macro. En caso contrario APPROVED. No redimensiones la posición. "
            "Responde ESTRICTAMENTE:\n"
            "VERDICT=<APPROVED|REJECTED>; REASON=<explicación breve en español>"
        )
    )
    msg = HumanMessage(
        content=(
            f"Fecha: {as_of_date}\nTicker: {ticker}\n\n"
            f"Tesis PM:\n{pm_thesis}\n\n"
            f"Informe macro:\n{macro_report}"
        )
    )
    out = risk_model.invoke([sys, msg])
    verdict = message_text(out)
    decision = "REJECTED" if "REJECTED" in verdict.upper() else "APPROVED"
    trace("pit-v2/risk", f"DONE decision={decision}")
    return decision, verdict


def run_pipeline_as_of(
    ticker: str,
    as_of_date: date,
    news_start_date: date,
) -> dict[str, Any]:
    """Run committee thesis, adversarial review and final risk validation."""
    result = dict(run_thesis_as_of(ticker, as_of_date, news_start_date))

    action, confidence, size_pct = parse_pm_decision(result["pm_thesis"])
    vol = result["technicals_data"].get("vol_30d_annualized")
    hard_status, hard_reason = hard_risk_precheck(
        action, confidence, size_pct, vol
    )

    result.update(
        {
            "pm_action": action,
            "pm_confidence": confidence,
            "pm_size_pct": size_pct,
            "hard_risk_status": hard_status,
            "hard_risk_reason": hard_reason,
            "debate_status": "",
            "bull_case": "",
            "bear_case": "",
            "devil_critique": "",
            "judge_verdict": "",
            "judge_decision": "",
            "judge_confidence": 0,
            "judge_rationale": "",
            "qualitative_risk_status": "SKIPPED",
            "qualitative_risk_verdict": "",
            "committee_action": action,
            "committee_size_pct": size_pct if action == "BUY" else 0.0,
            "committee_final_decision": action,
            "final_action": action,
            "final_size_pct": size_pct if action == "BUY" else 0.0,
            "final_decision": action,
        }
    )

    if hard_status == "REJECTED":
        result.update(
            {
                "debate_status": "SKIPPED_HARD_RISK",
                "committee_action": "HOLD",
                "committee_size_pct": 0.0,
                "committee_final_decision": "HOLD (hard risk rejected)",
                "final_action": "HOLD",
                "final_size_pct": 0.0,
                "final_decision": "HOLD (hard risk rejected)",
                "risk_verdict": hard_reason,
            }
        )
        return result

    if action != "BUY":
        result.update(
            {
                "debate_status": "SKIPPED_NON_BUY",
                "committee_size_pct": 0.0,
                "final_size_pct": 0.0,
                "risk_verdict": "SKIPPED_NON_BUY",
            }
        )
        return result

    debate_initial: DebateState = {
        "ticker": result["ticker"],
        "as_of_date": result["as_of_date"],
        "fundamentals_report": result["fundamentals_report"],
        "technicals_report": result["technicals_report"],
        "sentiment_report": result["sentiment_report"],
        "macro_report": result["macro_report"],
        "thesis_narrative": _strip_pm_control_line(result["pm_thesis"]),
        "pm_thesis": result["pm_thesis"],
        "pm_action": action,
        "pm_confidence": confidence,
        "pm_size_pct": size_pct,
        "bull_case": "",
        "bear_case": "",
        "devil_critique": "",
        "judge_verdict": "",
        "judge_decision": "",
        "judge_confidence": 0,
        "judge_rationale": "",
    }
    debate = build_debate().invoke(debate_initial)
    result.update(
        {
            "debate_status": "RUN",
            "bull_case": debate["bull_case"],
            "bear_case": debate["bear_case"],
            "devil_critique": debate["devil_critique"],
            "judge_verdict": debate["judge_verdict"],
            "judge_decision": debate["judge_decision"],
            "judge_confidence": debate["judge_confidence"],
            "judge_rationale": debate["judge_rationale"],
        }
    )

    # Run the same qualitative risk gate regardless of the Judge result so the
    # committee-only baseline remains directly comparable. Risk does not see debate.
    risk_status, risk_verdict = qualitative_risk_review(
        ticker=result["ticker"],
        as_of_date=result["as_of_date"],
        pm_thesis=result["pm_thesis"],
        macro_report=result["macro_report"],
    )
    result["qualitative_risk_status"] = risk_status
    result["qualitative_risk_verdict"] = risk_verdict
    result["risk_verdict"] = risk_verdict

    if risk_status == "REJECTED":
        result["committee_action"] = "HOLD"
        result["committee_size_pct"] = 0.0
        result["committee_final_decision"] = "HOLD (qualitative risk rejected)"
    else:
        result["committee_action"] = "BUY"
        result["committee_size_pct"] = size_pct
        result["committee_final_decision"] = "BUY"

    if (
        result["committee_action"] == "BUY"
        and result["judge_decision"] == "APPROVED"
    ):
        result["final_action"] = "BUY"
        result["final_size_pct"] = size_pct
        result["final_decision"] = "BUY"
    elif result["judge_decision"] == "REJECTED":
        result["final_action"] = "HOLD"
        result["final_size_pct"] = 0.0
        result["final_decision"] = "HOLD (judge rejected)"
    else:
        result["final_action"] = "HOLD"
        result["final_size_pct"] = 0.0
        result["final_decision"] = result["committee_final_decision"]

    return result
