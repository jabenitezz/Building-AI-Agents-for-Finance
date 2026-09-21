"""
Adversarial Debate layer — implemented as a LangGraph state graph that runs
ON TOP of the Investment Committee from `investment_committee.py`.

The committee's thesis becomes the seed for a structured debate:
  - A BULL agent argues to enter the position.
  - A BEAR agent argues against.
  - A Devil's Advocate identifies the weakest claim on each side.
  - A neutral Judge issues a final verdict that may differ from the committee.

OpenAI-only variant:
  The original version deliberately used Anthropic + OpenAI for Bull/Bear
  diversity. This variant removes the Anthropic dependency. It uses different
  OpenAI capability tiers for the opposing roles. This preserves model-tier
  diversity, but it is NOT equivalent to cross-provider diversity and may
  produce more correlated errors than the original heterogeneous setup.

Run:
    export OPENAI_API_KEY=...
    python adversarial_debate.py
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from investment_committee import run_committee
from models import (
    make_opus_equivalent,
    make_sonnet_equivalent,
    message_text,
    trace,
)


# ---------------------------------------------------------------------------
# OpenAI-only model line-up.
# ---------------------------------------------------------------------------
# Original:
#   Bull  = Claude Opus 4.7
#   Bear  = OpenAI GPT-4o
#   Devil = Claude Sonnet 4.6
#   Judge = Claude Sonnet 4.6
#
# OpenAI-only v1:
#   Bull  = GPT-5.6 Sol   (premium / strongest reasoning role)
#   Bear  = GPT-5.6 Terra (balanced counter-view)
#   Devil = GPT-5.6 Terra
#   Judge = GPT-5.6 Terra
#
# Cross-provider diversity is intentionally lost in this variant; the prompts
# still force adversarial roles, while model tiers remain distinct.
bull_model = make_opus_equivalent(max_tokens=1500)
bear_model = make_sonnet_equivalent(max_tokens=1500)
devil_model = make_sonnet_equivalent(max_tokens=1000)
judge_model = make_sonnet_equivalent(max_tokens=1500)


class DebateState(TypedDict):
    ticker: str
    committee_thesis: str
    committee_decision: str
    bull_case: str
    bear_case: str
    devil_critique: str
    judge_verdict: str
    final_decision: str


def bull_node(state: DebateState) -> dict:
    trace("debate/bull", f"START ticker={state['ticker']} -> GPT-5.6 Sol")
    sys = SystemMessage(content=(
        "You are a BULL analyst. Construct the strongest possible case to ENTER a long "
        "position. Cite the strongest evidence from the committee thesis: catalysts, "
        "valuation support, technical confirmation, sentiment tailwinds. Be concrete and "
        "do NOT hedge. Max 200 words. End with: BULL_CONVICTION=<0-100>."
    ))
    msg = HumanMessage(content=(
        f"Ticker: {state['ticker']}\n"
        f"Committee thesis:\n{state['committee_thesis']}\n"
        f"Committee decision: {state['committee_decision']}"
    ))
    out = bull_model.invoke([sys, msg])
    case = message_text(out)
    trace("debate/bull", f"DONE chars={len(case)} preview={case[:140]!r}")
    return {"bull_case": case}


def bear_node(state: DebateState) -> dict:
    trace("debate/bear", f"START ticker={state['ticker']} -> GPT-5.6 Terra")
    sys = SystemMessage(content=(
        "You are a BEAR analyst. Construct the strongest possible case AGAINST entering "
        "a long position. Surface risks the committee may have under-weighted: tail "
        "risks, macro headwinds, valuation stretch, sentiment crowding, technical "
        "exhaustion. Be concrete and do NOT hedge. Max 200 words. End with: "
        "BEAR_CONVICTION=<0-100>."
    ))
    msg = HumanMessage(content=(
        f"Ticker: {state['ticker']}\n"
        f"Committee thesis:\n{state['committee_thesis']}\n"
        f"Committee decision: {state['committee_decision']}"
    ))
    out = bear_model.invoke([sys, msg])
    case = message_text(out)
    trace("debate/bear", f"DONE chars={len(case)} preview={case[:140]!r}")
    return {"bear_case": case}


def devil_advocate(state: DebateState) -> dict:
    trace("debate/devil", "START -> comparing Bull and Bear -> GPT-5.6 Terra")
    sys = SystemMessage(content=(
        "You are a Devil's Advocate. Identify the SINGLE weakest claim in the BULL case "
        "and the SINGLE weakest claim in the BEAR case, and explain in one sentence each "
        "why those claims are weak. Max 150 words."
    ))
    msg = HumanMessage(content=(
        f"BULL case:\n{state['bull_case']}\n\nBEAR case:\n{state['bear_case']}"
    ))
    out = devil_model.invoke([sys, msg])
    critique = message_text(out)
    trace("debate/devil", f"DONE chars={len(critique)} preview={critique[:140]!r}")
    return {"devil_critique": critique}


def _parse_verdict(text: str) -> str:
    upper = text.upper().replace(" ", "")
    for action in ("BUY", "SELL", "HOLD"):
        if f"VERDICT={action}" in upper:
            return action
    return "HOLD"


def judge_node(state: DebateState) -> dict:
    trace("debate/judge", "START -> final adjudication -> GPT-5.6 Terra")
    sys = SystemMessage(content=(
        "You are a NEUTRAL Judge. Weigh the bull case, the bear case, and the devil's "
        "critique. Issue a verdict that may differ from the committee's prior decision. "
        "Reply STRICTLY in the form:\n"
        "VERDICT=<BUY|HOLD|SELL>\nCONFIDENCE=<0-100>\nRATIONALE=<2-3 sentences>"
    ))
    msg = HumanMessage(content=(
        f"Committee prior: {state['committee_decision']}\n\n"
        f"BULL case:\n{state['bull_case']}\n\n"
        f"BEAR case:\n{state['bear_case']}\n\n"
        f"Devil's critique:\n{state['devil_critique']}"
    ))
    out = judge_model.invoke([sys, msg])
    verdict = message_text(out)
    final = _parse_verdict(verdict)
    trace("debate/judge", f"DONE final_decision={final} verdict={verdict!r}")
    return {"judge_verdict": verdict, "final_decision": final}


def build_debate():
    g = StateGraph(DebateState)
    g.add_node("bull", bull_node)
    g.add_node("bear", bear_node)
    g.add_node("devil", devil_advocate)
    g.add_node("judge", judge_node)

    # Bull and bear in parallel; both feed the devil; devil feeds the judge.
    g.add_edge(START, "bull")
    g.add_edge(START, "bear")
    g.add_edge("bull", "devil")
    g.add_edge("bear", "devil")
    g.add_edge("devil", "judge")
    g.add_edge("judge", END)
    return g.compile()


def run_pipeline(ticker: str) -> dict:
    """Run the full pipeline: committee first, then debate on the committee output."""
    trace("debate", f"START full pipeline ticker={ticker}")
    trace("debate", "STAGE 1 -> investment committee")
    committee_result = run_committee(ticker)
    trace("debate", f"STAGE 1 DONE committee={committee_result['final_decision']}")
    trace("debate", "STAGE 2 -> Bull + Bear in parallel -> Devil -> Judge")
    initial: DebateState = {
        "ticker": ticker,
        "committee_thesis": committee_result["pm_thesis"],
        "committee_decision": committee_result["final_decision"],
        "bull_case": "",
        "bear_case": "",
        "devil_critique": "",
        "judge_verdict": "",
        "final_decision": "",
    }
    debate_result = build_debate().invoke(initial)
    trace("debate", f"END judge={debate_result['final_decision']}")
    return {"committee": committee_result, "debate": debate_result}


if __name__ == "__main__":
    out = run_pipeline("TSLA")
    print("=" * 80)
    print(f"COMMITTEE said       : {out['committee']['final_decision']}")
    print(f"DEBATE  judge says   : {out['debate']['final_decision']}")
    print("=" * 80)
    print("\n--- Bull case ---\n",        out["debate"]["bull_case"])
    print("\n--- Bear case ---\n",        out["debate"]["bear_case"])
    print("\n--- Devil's critique ---\n", out["debate"]["devil_critique"])
    print("\n--- Judge verdict ---\n",    out["debate"]["judge_verdict"])
