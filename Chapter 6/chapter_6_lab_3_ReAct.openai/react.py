import argparse
import json
import os
import statistics

import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
client = OpenAI()

# Same curated sector peer lists as the original Claude lab.
SECTOR_TICKERS = {
    "Technology": ["MSFT", "NVDA", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO", "AVGO", "QCOM"],
    "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT", "LLY", "BMY", "AMGN"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "USB", "PNC"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "T", "VZ", "CMCSA", "TMUS"],
}

DEFAULT_QUESTION = (
    "¿Está Apple (AAPL) sobrevalorada en relación con el sector tecnológico? "
    "¿Qué implica esto para una cartera long-only?"
)


def get_valuation_ratios(ticker: str) -> dict:
    """Retrieve valuation ratios and sector from Yahoo Finance."""
    info = yf.Ticker(ticker).info
    return {
        "PE": info.get("trailingPE"),
        "EV_EBITDA": info.get("enterpriseToEbitda"),
        "PB": info.get("priceToBook"),
        "PEG": info.get("pegRatio") or info.get("trailingPegRatio"),
        "sector": info.get("sector"),
    }


def get_sector_median_pe(sector: str) -> dict:
    """Compute median P/E and EV/EBITDA for the configured sector peers."""
    pes, evs = [], []
    for tk in SECTOR_TICKERS.get(sector, []):
        info = yf.Ticker(tk).info
        if info.get("trailingPE"):
            pes.append(info["trailingPE"])
        if info.get("enterpriseToEbitda"):
            evs.append(info["enterpriseToEbitda"])
    return {
        "median_pe": round(statistics.median(pes), 2) if pes else None,
        "median_ev_ebitda": round(statistics.median(evs), 2) if evs else None,
        "sample_size": len(pes),
    }


def get_balance_sheet(ticker: str) -> dict:
    """Retrieve the same balance-sheet summary as the original lab."""
    info = yf.Ticker(ticker).info
    cash = info.get("totalCash") or 0
    debt = info.get("totalDebt") or 0
    de = info.get("debtToEquity")
    return {
        "net_cash_bn": round((cash - debt) / 1e9, 1),
        "debt_to_equity": round(de / 100, 2) if de else None,
    }


# Same three financial tools as the original lab.
#
# The extra "thought" field is NOT hidden chain-of-thought. It is a short,
# user-facing rationale explaining why the agent is about to use that tool.
# Requiring it lets the console preserve the pedagogical ReAct trace:
# THOUGHT -> ACTION -> OBSERVATION.
TOOLS = [
    {
        "type": "function",
        "name": "get_valuation_ratios",
        "description": "Retrieve valuation ratios (P/E, EV/EBITDA, P/B, PEG) and sector for a given ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker, for example AAPL.",
                },
                "thought": {
                    "type": "string",
                    "description": (
                        "One short user-facing sentence in Spanish explaining why "
                        "this tool is needed now. Do not provide private chain-of-thought."
                    ),
                },
            },
            "required": ["ticker", "thought"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_sector_median_pe",
        "description": "Retrieve the median P/E and EV/EBITDA for a given Yahoo Finance sector using the configured large-cap peer sample.",
        "parameters": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": "Yahoo Finance sector label, for example Technology.",
                },
                "thought": {
                    "type": "string",
                    "description": (
                        "One short user-facing sentence in Spanish explaining why "
                        "this tool is needed now, based on observations already obtained. "
                        "Do not provide private chain-of-thought."
                    ),
                },
            },
            "required": ["sector", "thought"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_balance_sheet",
        "description": "Retrieve balance-sheet summary (net cash in USD billions and debt-to-equity) for a ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker, for example AAPL.",
                },
                "thought": {
                    "type": "string",
                    "description": (
                        "One short user-facing sentence in Spanish explaining why "
                        "this tool is needed now, based on observations already obtained. "
                        "Do not provide private chain-of-thought."
                    ),
                },
            },
            "required": ["ticker", "thought"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

TOOL_FUNCTIONS = {
    "get_valuation_ratios": get_valuation_ratios,
    "get_sector_median_pe": get_sector_median_pe,
    "get_balance_sheet": get_balance_sheet,
}

SYSTEM_PROMPT = """You are a financial research analyst following the ReAct pattern.

Use the available tools to gather the evidence needed to answer the user's question.
Base factual financial claims on tool observations from this run. Do not invent missing
tool results or supplement them with unstated outside facts.

For EVERY tool call, include a short public-facing rationale in the tool's "thought"
argument. This rationale must explain only what information you need next or what the
previous observation implies for the next action. It must be concise, written in
Spanish (Spain), and must NOT contain private chain-of-thought or hidden reasoning.

The console will display that rationale as:
[THOUGHT] ...
followed by:
[ACTION] ...
[OBSERVATION] ...

Continue this Thought -> Action -> Observation loop until you have enough evidence.
Then conclude with a clear, well-grounded final answer.

The final answer must be written entirely in Spanish (Spain), while keeping standard
financial abbreviations such as P/E, EV/EBITDA, P/B and PEG unchanged.

For a valuation-vs-sector question, normally:
1. Retrieve the stock's valuation ratios and sector.
2. Retrieve the sector median valuation using the sector returned by the first tool.
3. Retrieve the stock's balance-sheet summary.
4. Compare the observed metrics and explain the implications for the portfolio.
"""


def _execute_tool(name: str, args: dict) -> dict:
    """Dispatch one OpenAI function call to the local Python tool."""
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool requested by model: {name}")

    # "thought" is display-only metadata; it is not passed to the financial function.
    tool_args = {key: value for key, value in args.items() if key != "thought"}
    return TOOL_FUNCTIONS[name](**tool_args)


def run_react_agent(question: str) -> str:
    """Run the ReAct loop with an explicit public THOUGHT before every ACTION."""
    input_items = [{"role": "user", "content": question}]

    while True:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=input_items,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=True,
            store=False,
        )

        tool_calls = [
            item
            for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]

        if not tool_calls:
            return (response.output_text or "").strip()

        # Stateless Responses API loop: preserve the model's function-call items,
        # then append each observation as a function_call_output.
        input_items += response.output

        for call in tool_calls:
            args = json.loads(call.arguments)
            thought = args.get(
                "thought",
                "Necesito consultar esta fuente antes de continuar con el análisis.",
            )
            action_args = {
                key: value for key, value in args.items() if key != "thought"
            }

            print(f"\n[THOUGHT] {thought}")
            print(f"[ACTION] {call.name}({action_args})")

            try:
                result = _execute_tool(call.name, args)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}

            observation = json.dumps(result, ensure_ascii=False)
            print(f"[OBSERVATION] {observation}")

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": observation,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agente financiero ReAct con OpenAI y datos de Yahoo Finance."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=(
            "Pregunta a analizar. Si se omite, se usa el ejemplo de Apple. "
            "Para preguntas con espacios, pásala entre comillas."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    question = args.question

    print("\n" + "=" * 60)
    print("PREGUNTA")
    print("=" * 60)
    print(question)

    answer = run_react_agent(question)

    print("\n" + "=" * 60)
    print("RESPUESTA FINAL")
    print("=" * 60)
    print(answer)
