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


# OpenAI function-tool definitions for the same three tools used by the Claude lab.
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
                }
            },
            "required": ["ticker"],
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
                }
            },
            "required": ["sector"],
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
                }
            },
            "required": ["ticker"],
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

SYSTEM_PROMPT = """You are a financial research analyst following a ReAct-style tool-use loop.

Use the available tools to gather the evidence needed to answer the user's question.
Base factual financial claims on tool observations from this run. Do not invent missing
tool results or supplement them with unstated outside facts.

You may briefly state a user-facing plan or interpretation, but do not reveal private
chain-of-thought. Call the appropriate tools, inspect the observations, and continue
until you have enough evidence. Conclude with a clear, well-grounded answer.

For a valuation-vs-sector question, normally:
1. Retrieve the stock's valuation ratios and sector.
2. Retrieve the sector median valuation using the sector returned by the first tool.
3. Retrieve the stock's balance-sheet summary.
4. Compare the observed metrics and explain the implications for the portfolio.
"""


def _execute_tool(name: str, arguments_json: str) -> dict:
    """Dispatch one OpenAI function call to the local Python tool."""
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool requested by model: {name}")

    args = json.loads(arguments_json)
    return TOOL_FUNCTIONS[name](**args)


def _print_model_text(response) -> None:
    """Print any user-visible text emitted alongside tool calls."""
    text = (response.output_text or "").strip()
    if text:
        print(f"\n[MODEL] {text}")


def run_react_agent(question: str) -> str:
    """Run the ReAct-style loop with OpenAI Responses API function calling."""
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

        _print_model_text(response)

        # Stateless Responses API loop: preserve the model's function-call items,
        # then append each observation as a function_call_output.
        input_items += response.output

        for call in tool_calls:
            args = json.loads(call.arguments)
            print(f"\n[ACTION] {call.name}({args})")

            try:
                result = _execute_tool(call.name, call.arguments)
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


if __name__ == "__main__":
    question = (
        "Is Apple (AAPL) overvalued relative to the Technology sector? "
        "What does this imply for a long-only portfolio?"
    )
    answer = run_react_agent(question)
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)
