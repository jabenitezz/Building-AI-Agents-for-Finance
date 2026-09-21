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
from typing import Any

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

TRACE_ENABLED = os.getenv("OPENAI_V1_TRACE", "1").strip().lower() not in {
    "0", "false", "no", "off",
}


def trace(scope: str, message: str) -> None:
    """Small stdout trace used by all Chapter 8 openai.v1 scripts."""
    if TRACE_ENABLED:
        print(f"[TRACE][{scope}] {message}", flush=True)


def message_text(message: Any) -> str:
    """Normalize LangChain/OpenAI Responses API message content to plain text.

    With ChatOpenAI + Responses API, AIMessage.content may be a list of
    structured content blocks instead of a single string. The original
    Anthropic-oriented code assumed a string, which breaks calls such as
    .upper(), regex parsing, and prompt interpolation.
    """
    if message is None:
        return ""

    # LangChain AIMessage exposes a text() helper in current versions.
    text_attr = getattr(message, "text", None)
    if callable(text_attr):
        try:
            value = text_attr()
            if isinstance(value, str) and value:
                return value
        except Exception:
            pass
    elif isinstance(text_attr, str) and text_attr:
        return text_attr

    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue

            if isinstance(block, dict):
                # Responses-style text/output_text blocks.
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
                    continue
                if isinstance(value, dict):
                    nested = value.get("value") or value.get("text")
                    if isinstance(nested, str):
                        parts.append(nested)
                        continue

                value = block.get("content")
                if isinstance(value, str):
                    parts.append(value)
                continue

            value = getattr(block, "text", None)
            if isinstance(value, str):
                parts.append(value)

        if parts:
            return "\n".join(parts)

    return str(content)


def _make_reasoning_model(
    model: str,
    *,
    max_tokens: int,
    effort: str,
) -> ChatOpenAI:
    """Create a GPT-5.6 model through the Responses API."""
    trace("models", f"CREATE model={model} reasoning={effort} max_tokens={max_tokens}")
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
