"""AutoGen v1 — robust conversational financial analyst.

This keeps the original Chapter 7 coordination model:
    code_executor -> AssistantAgent -> generated Python -> code_executor -> ...

The v1 changes are intentionally narrow:
  * seed the executor work directory with robust yfinance helpers,
  * tell the assistant not to guess statement row labels,
  * require latest-period calculations rather than summing across years,
  * give the recovery loop more turns,
  * terminate only after successful code execution and a final analysis.

Usage from Chapter 7:
    python "src/autogen.v1/run.py" "Assess the financial health of AAPL"
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor


# Ensure generated code uses the same interpreter/venv as this launcher.
os.environ["PATH"] = (
    os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
)


MODEL = os.getenv("AUTOGEN_MODEL", "gpt-4o")
MAX_TURNS = int(os.getenv("AUTOGEN_MAX_TURNS", "10"))

config_list = [
    {
        "model": MODEL,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]

llm_config = {
    "config_list": config_list,
    "cache_seed": 43,
}


SYSTEM_MESSAGE = """You are a financial analyst assistant.

When asked to analyze a company, write Python code using yfinance and execute
it through the code_executor before giving the final analysis.

ROBUSTNESS RULES FOR FINANCIAL HEALTH ANALYSIS:
1. For standard financial-health metrics, FIRST use the helper already present
   in the execution directory:

       from financial_helpers import get_health_snapshot

   Call get_health_snapshot("TICKER") and print the returned dictionary.
   Use those observed values for the analysis.
2. If the user requests a price chart, use:

       from financial_helpers import save_price_plot

   Save plots to PNG files; never call plt.show().
3. Do not guess yfinance financial-statement row labels. If a metric not
   covered by the helper requires direct statement access, inspect and print
   the available DataFrame index labels first, then use an exact existing
   label.
4. Never sum balance-sheet values across multiple years to compute a
   single-period ratio. Use the latest non-null reporting period.
5. If code execution fails, read the traceback, change the next attempt, and
   do not repeat the same failing lookup unchanged.
6. Do not invent unavailable values. State when a metric is unavailable.
7. Never write TERMINATE in a message that contains code.
8. Only write TERMINATE after you have observed at least one successful code
   execution (exit code 0) and then written the final, data-backed analysis.

Present the final result clearly and distinguish observed data from your
interpretation.
"""


def _prepare_work_dir() -> str:
    """Create the executor directory and seed it with the robust helper."""
    work_dir = tempfile.mkdtemp(prefix="autogen_v1_")
    helper_source = Path(__file__).with_name("financial_helpers.py")
    helper_target = Path(work_dir) / "financial_helpers.py"
    shutil.copy2(helper_source, helper_target)
    return work_dir


def run_analysis(query: str) -> dict:
    """Run the two-agent conversation with error-recovery headroom."""
    work_dir = _prepare_work_dir()

    executor = LocalCommandLineCodeExecutor(
        timeout=90,
        work_dir=work_dir,
    )

    assistant = AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message=SYSTEM_MESSAGE,
    )

    code_executor = UserProxyAgent(
        name="code_executor",
        human_input_mode="NEVER",
        code_execution_config={"executor": executor},
        llm_config=False,
        is_termination_msg=lambda msg: (
            msg.get("content", "")
            and "TERMINATE" in msg.get("content", "")
        ),
    )

    print(f"Executor work directory: {work_dir}")
    print(f"Model: {MODEL} | max_turns: {MAX_TURNS}")

    chat_result = code_executor.initiate_chat(
        assistant,
        message=query,
        max_turns=MAX_TURNS,
    )

    print(f"\n{'=' * 50}")
    print("COST SUMMARY:")
    print(f"{'=' * 50}")
    print(chat_result.cost)

    return {
        "chat_history": chat_result.chat_history,
        "cost": chat_result.cost,
        "summary": chat_result.summary,
        "work_dir": work_dir,
    }


if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the current price and key financial metrics "
        "of NVIDIA? Plot the stock price over the last 6 months."
    )

    print(f"\nQuery: {query}\n")
    result = run_analysis(query)

    print(f"\n{'=' * 50}")
    print("SUMMARY:")
    print(f"{'=' * 50}")
    print(result["summary"])
    print(f"\nArtifacts/work directory: {result['work_dir']}")
