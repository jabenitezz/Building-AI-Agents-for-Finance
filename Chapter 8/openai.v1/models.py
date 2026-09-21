"""OpenAI model-tier translation for Chapter 8 openai.v1.

The original Chapter 8 code uses:
- Claude Opus 4.7 for premium reasoning roles.
- Claude Sonnet 4.6 for balanced specialist/reviewer roles.
- Claude Haiku 4.5 for high-volume, cost-sensitive sentiment work.

This OpenAI-only variant maps those *roles* to the current OpenAI tiers:
- Opus-like role   -> GPT-5.6 Sol
- Sonnet-like role -> GPT-5.6 Terra
- Haiku-like role  -> GPT-5.6 Luna

This is a capability/tier mapping, not a claim that vendor models are
benchmark-identical.
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


OPUS_EQUIVALENT_MODEL = os.getenv(
    "OPENAI_OPUS_EQUIVALENT_MODEL",
    "gpt-5.6-sol",
)
SONNET_EQUIVALENT_MODEL = os.getenv(
    "OPENAI_SONNET_EQUIVALENT_MODEL",
    "gpt-5.6-terra",
)
HAIKU_EQUIVALENT_MODEL = os.getenv(
    "OPENAI_HAIKU_EQUIVALENT_MODEL",
    "gpt-5.6-luna",
)


def _make_reasoning_model(
    model: str,
    *,
    max_tokens: int,
    effort: str,
) -> ChatOpenAI:
    """Create a GPT-5.6 model through the Responses API."""
    return ChatOpenAI(
        model=model,
        max_tokens=max_tokens,
        use_responses_api=True,
        reasoning={"effort": effort},
    )


def make_opus_equivalent(*, max_tokens: int = 2000) -> ChatOpenAI:
    """Premium reasoning tier: Claude Opus role -> GPT-5.6 Sol."""
    return _make_reasoning_model(
        OPUS_EQUIVALENT_MODEL,
        max_tokens=max_tokens,
        effort="high",
    )


def make_sonnet_equivalent(*, max_tokens: int = 1500) -> ChatOpenAI:
    """Balanced tier: Claude Sonnet role -> GPT-5.6 Terra."""
    return _make_reasoning_model(
        SONNET_EQUIVALENT_MODEL,
        max_tokens=max_tokens,
        effort="medium",
    )


def make_haiku_equivalent(*, max_tokens: int = 800) -> ChatOpenAI:
    """High-volume tier: Claude Haiku role -> GPT-5.6 Luna."""
    return _make_reasoning_model(
        HAIKU_EQUIVALENT_MODEL,
        max_tokens=max_tokens,
        effort="low",
    )
