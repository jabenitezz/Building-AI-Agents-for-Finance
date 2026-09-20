import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

HEDGES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["name", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


def call_structured(prompt: str, schema: dict, schema_name: str) -> str:
    """Call OpenAI and return JSON text constrained by the supplied schema."""
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=2048,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
        store=False,
    )
    return response.output_text


def generate_hedges(portfolio: str, scenario: str, n: int) -> list[dict]:
    prompt = (
        f"Portfolio: {portfolio}\nStress scenario: {scenario}\n"
        f"Propose {n} distinct hedge candidates, each with a short rationale."
    )
    text = call_structured(prompt, HEDGES_SCHEMA, "hedge_candidates")
    return json.loads(text)["candidates"]


def score_hedge(portfolio: str, scenario: str, hedge: dict) -> dict:
    prompt = (
        f"Portfolio: {portfolio}\nStress scenario: {scenario}\n"
        f"Candidate hedge: {hedge['name']} ({hedge['rationale']})\n"
        f"Score this hedge from 1 to 10 on protection, cost, and basis risk."
    )
    text = call_structured(prompt, SCORE_SCHEMA, "hedge_score")
    print(text)
    return json.loads(text)


def generate_sizings(
    portfolio: str, scenario: str, hedge: dict, n: int
) -> list[dict]:
    prompt = (
        f"Portfolio: {portfolio}\nScenario: {scenario}\n"
        f"Parent hedge: {hedge['name']} ({hedge['rationale']})\n"
        f"Propose {n} distinct sizings or structures for this hedge "
        f"(e.g. strike, notional, expiry, spread width), each with a short rationale."
    )
    text = call_structured(prompt, HEDGES_SCHEMA, "sizing_candidates")
    return json.loads(text)["candidates"]


def score_sizing(
    portfolio: str, scenario: str, hedge: dict, sizing: dict
) -> dict:
    prompt = (
        f"Portfolio: {portfolio}\nScenario: {scenario}\n"
        f"Parent hedge: {hedge['name']} ({hedge['rationale']})\n"
        f"Candidate sizing: {sizing['name']} ({sizing['rationale']})\n"
        f"Score this sizing from 1 to 10 on residual exposure to the catalyst, "
        f"premium cost, and how cleanly it complements the parent hedge."
    )
    text = call_structured(prompt, SCORE_SCHEMA, "sizing_score")
    return json.loads(text)


def best_hedges(
    portfolio: str, scenario: str, beam_width: int = 2
) -> list[dict]:
    # Level 1: pick the best hedge instruments
    candidates = generate_hedges(portfolio, scenario, n=4)
    print("\n\nCANDIDATES")
    print("=" * 10)
    print(candidates)
    print("=" * 10)

    scored = [{**c, **score_hedge(portfolio, scenario, c)} for c in candidates]
    scored.sort(key=lambda x: x["score"], reverse=True)

    print("\n\nSCORED")
    print("=" * 10)
    print(scored)
    print("=" * 10)

    retained = scored[:beam_width]
    print("\n\nRETAINED")
    print("=" * 10)
    print(retained)
    print("=" * 10)

    # Level 2: for each retained hedge, pick its best sizing
    for hedge in retained:
        sizings = generate_sizings(portfolio, scenario, hedge, n=3)

        print("\n\nSIZING")
        print("=" * 10)
        print(sizings)
        print("=" * 10)

        scored_sizings = [
            {**s, **score_sizing(portfolio, scenario, hedge, s)}
            for s in sizings
        ]
        scored_sizings.sort(key=lambda x: x["score"], reverse=True)

        print("\n\nSCORED SIZING")
        print("=" * 10)
        print(scored_sizings)
        print("=" * 10)

        hedge["best_sizing"] = scored_sizings[0]

    return retained


if __name__ == "__main__":
    portfolio = "5,000 shares of NVDA held outright (~$5M concentrated single-name position)"
    scenario = "NVDA reports earnings tonight after the close; weekly implied volatility is elevated"
    for h in best_hedges(portfolio, scenario):
        print(f"{h['score']:>2}  {h['name']}: {h['reasoning']}")
        s = h["best_sizing"]
        print(f"      └─ best sizing ({s['score']}): {s['name']}: {s['reasoning']}")
