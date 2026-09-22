"""run.py
======
Entry point for the adversarial fraud-investigation workflow.

Usage:
    python -m src.fraud_investigation.run

Runs a three-agent debate:
    FraudAdvocate -> LegitimacyAdvocate -> InvestigationDecision
"""

import asyncio

from llama_index.core.agent.workflow import AgentWorkflow

from .agents import (
    fraud_advocate,
    legitimacy_advocate,
    investigation_decision,
)


SUSPICIOUS_CLAIM_SUMMARY = (
    "Claimant: J. Doe. Policy POL-2026-AUTO-99887 inception 2026-04-15. "
    "Date of loss: 2026-04-20 (5 days after inception). Description: "
    "Single-vehicle accident, total loss claimed. Vehicle reportedly stolen "
    "and burned in a remote area at night. No witnesses, no security footage. "
    "Prior claims: 3 in the past 14 months. Estimated payout would be 94% of "
    "the policy limit. Police report indicates no signs of forced entry, "
    "ignition intact, fire originated inside the passenger compartment."
)


async def run_fraud_investigation():
    """Run the complete adversarial fraud-investigation workflow."""

    workflow = AgentWorkflow(
        agents=[
            fraud_advocate,
            legitimacy_advocate,
            investigation_decision,
        ],
        root_agent=fraud_advocate.name,
        initial_state={
            "claim_summary": SUSPICIOUS_CLAIM_SUMMARY,
        },
    )

    user_msg = (
        "Investigate the following potentially fraudulent claim. "
        "Argue both sides before reaching a determination.\n\n"
        f"Claim summary:\n{SUSPICIOUS_CLAIM_SUMMARY}"
    )

    handler = workflow.run(
        user_msg=user_msg,
        max_iterations=30,
    )

    last_agent = None
    async for event in handler.stream_events():
        name = getattr(event, "current_agent_name", None)
        if name and name != last_agent:
            print(f"\n{'=' * 60}")
            print(f"  Agent: {name}")
            print(f"{'=' * 60}")
            last_agent = name

        delta = getattr(event, "delta", None)
        if delta:
            print(delta, end="", flush=True)

    result = await handler

    print(f"\n\n{'=' * 60}")
    print("  FINAL FRAUD DETERMINATION")
    print(f"{'=' * 60}")
    print(result)

    return result


if __name__ == "__main__":
    print("Investigating suspicious insurance claim...\n")
    asyncio.run(run_fraud_investigation())
