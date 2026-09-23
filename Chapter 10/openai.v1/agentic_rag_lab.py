"""
Agentic RAG over financial documents with LlamaIndex.

Companion lab for the chapter "Building Agentic RAG Systems in Finance".

Generation/reasoning runs on OpenAI GPT-5.6 Terra with medium reasoning effort;
the vector index uses the lower-cost OpenAI text-embedding-3-small model.

The file is organised into numbered steps that map 1:1 to the snippets in the
chapter prose. Run the whole thing with:

    python agentic_rag_lab.py

or import individual `step_*` functions from a notebook / REPL.

Each step prints its answers to the terminal AND writes a markdown report to
`results/lab_runs/`, recording both the retrieved context chunks and the final
answer so the behaviour of the different patterns can be compared after the fact.
A single shared question (`COMPARISON_QUESTION`) is run through the retrieval-pattern
steps (3-7 and the hybrid+rerank step 9) and collected into
`00_comparison_quick_ratio_policy.md` for a side-by-side view.

Requires OPENAI_API_KEY in the environment (see .env.example).
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

# Financial answers routinely contain non-Latin-1 characters (≥, ≤, —,
# en/em dashes). On Windows, stdout defaults to the cp1252 codepage and printing
# those raises UnicodeEncodeError, so force UTF-8 output here.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# -- Step 1: configuration -------------------------------------------------
# Use OpenAI for both generation/reasoning and embeddings. The original chapter
# uses Claude Sonnet 4.6 for generation; this OpenAI-only variant maps that role
# to GPT-5.6 Terra with medium reasoning effort (the same Sonnet-like mapping
# used in Chapter 8 openai.v1). Embeddings remain on text-embedding-3-small to
# keep vectorization inexpensive. Environment variables allow either tier to be
# changed without touching the lab code.

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5.6-terra")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "medium")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "180"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

Settings.llm = OpenAI(
    model=OPENAI_LLM_MODEL,
    reasoning_effort=OPENAI_REASONING_EFFORT,
    max_tokens=OPENAI_MAX_TOKENS,
    timeout=OPENAI_TIMEOUT_SECONDS,
    max_retries=OPENAI_MAX_RETRIES,
)
Settings.embed_model = OpenAIEmbedding(model=OPENAI_EMBED_MODEL)


# -- Reporting & chunk-inspection utilities --------------------------------
# This block is lab scaffolding, not part of the numbered chapter snippets. Its
# goal is to make each pattern's behaviour visible and comparable by
# capturing the chunks every step retrieves and recording the answers.
#
# Why bother? Every step below ends in an LLM answer, but the interesting
# difference between naive RAG, routing, decomposition, and the agent is what
# context each one retrieved before answering. These helpers surface that:
#
#   COMPARISON_QUESTION      one shared question, run through steps 3-7 and 9, so the
#                            patterns can be compared on an identical query.
#   _collect_source_nodes()  pull the retrieved chunks out of a response — from
#                            response.source_nodes for query engines, or from the
#                            agent's streamed tool outputs (see _run_agent_sync).
#   _render_chunks_md()      format those chunks (source file + relevance score +
#   _print_chunks()          text) for the markdown report / the terminal.
#   report_step()            print a step's answers AND write one markdown file
#                            per step to results/lab_runs/.
#   _log_comparison() +      accumulate each step's answer to COMPARISON_QUESTION
#   write_comparison_report()and emit a single side-by-side comparison file.
#
# So as you read each step, the printed/saved output tells you not just the
# answer but which document chunks (and at what score) produced it.

# A single comparative, multi-source question we run through steps 3-7 and 9. Each
# pattern retrieves and answers it differently, which is the whole point of the
# chapter — so collecting the answers side by side makes the progression obvious.
COMPARISON_QUESTION = "Does Acme's Q3 quick ratio satisfy the firm's risk policy?"

RESULTS_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "results", "lab_runs")
)

# Filled in as steps run COMPARISON_QUESTION; written out by main() at the end.
_COMPARISON_LOG: list[dict] = []


def _collect_source_nodes(response) -> list:
    """Return the retrieved chunks behind a response.

    Query engines expose them directly as `response.source_nodes`. A
    FunctionAgent retrieves indirectly via tool calls, so we dig the source
    nodes out of each tool output's `raw_output` instead.
    """
    direct = list(getattr(response, "source_nodes", None) or [])
    if direct:
        return direct
    collected: list = []
    for src in getattr(response, "sources", None) or []:
        raw = getattr(src, "raw_output", None)
        for node in getattr(raw, "source_nodes", None) or []:
            collected.append(node)
    return collected


def _node_source(node_with_score) -> str:
    meta = getattr(node_with_score.node, "metadata", {}) or {}
    name = meta.get("file_name") or meta.get("filename")
    if name:
        return name
    # SubQuestionQueryEngine adds its synthesized per-sub-question answers to
    # source_nodes as bare text nodes ("Sub question: ... Response: ..."), with no
    # file_name and no retrieval score. Label these so they are not mistaken for
    # retrieved document chunks.
    text = node_with_score.node.get_content() or ""
    if text.lstrip().startswith("Sub question:"):
        return "(synthesized sub-answer)"
    return "unknown"


def _node_score(node_with_score) -> str:
    score = getattr(node_with_score, "score", None)
    return f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"


def _render_chunks_md(nodes) -> str:
    """Full chunk text, for the markdown report."""
    if not nodes:
        return "_No retrieved chunks were exposed by this engine._"
    blocks = []
    for i, nws in enumerate(nodes, 1):
        text = nws.node.get_content().strip()
        quoted = "> " + text.replace("\n", "\n> ")
        blocks.append(
            f"**Chunk {i}** — source `{_node_source(nws)}`, score {_node_score(nws)}\n\n{quoted}"
        )
    return "\n\n".join(blocks)


def _print_chunks(nodes) -> None:
    """Condensed chunk view, for the terminal."""
    if not nodes:
        print("  retrieved chunks: (none exposed by this engine)")
        return
    print(f"  retrieved {len(nodes)} chunk(s):")
    for i, nws in enumerate(nodes, 1):
        snippet = " ".join(nws.node.get_content().split())[:160]
        print(f"   [{i}] {_node_source(nws)} (score {_node_score(nws)}): {snippet}…")


def report_step(filename: str, title: str, description: str, entries: list[dict]) -> str:
    """Print each entry to the terminal and write a markdown report.

    `entries` is a list of dicts with keys:
      - "question": the query string (or task) that was run
      - "response": the response object returned by the engine/agent
      - "note": optional extra markdown appended after the answer (e.g. eval verdicts)
      - "label": optional terminal heading (defaults to "Q")
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)

    md = [f"# {title}", "", f"_{description}_", ""]
    print(f"\n=== {title} ===")

    for entry in entries:
        question = entry["question"]
        response = entry["response"]
        label = entry.get("label", "Q")
        # An entry may carry chunks collected out-of-band (e.g. the agent's tool
        # retrievals); otherwise read them off the response.
        nodes = entry["nodes"] if entry.get("nodes") is not None else _collect_source_nodes(response)

        # Terminal
        print(f"\n{label}: {question}")
        _print_chunks(nodes)
        print(f"  answer:\n{response}")
        if entry.get("note"):
            print(f"  {entry['note']}")

        # Markdown
        md.append(f"## {label}: {question}\n")
        md.append(f"### Retrieved context ({len(nodes)} chunk(s))\n")
        md.append(_render_chunks_md(nodes) + "\n")
        md.append("### Answer\n")
        md.append(str(response).strip() + "\n")
        if entry.get("note"):
            md.append(f"> {entry['note']}\n")
        md.append("---\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    rel = os.path.relpath(path, os.path.dirname(__file__))
    print(f"\n→ saved {rel}")
    return path


def _log_comparison(step_label: str, response, nodes=None) -> None:
    """Record an answer to COMPARISON_QUESTION for the side-by-side report."""
    _COMPARISON_LOG.append({"step": step_label, "response": response, "nodes": nodes})


def write_comparison_report() -> str:
    """Aggregate every step's answer to COMPARISON_QUESTION into one file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "00_comparison_quick_ratio_policy.md")
    md = [
        "# Shared-question comparison",
        "",
        f"_The same question — **\"{COMPARISON_QUESTION}\"** — run through each "
        "retrieval pattern, so you can see how routing, decomposition, the agent, "
        "and hybrid search differ on an identical, cross-source query._",
        "",
    ]
    for item in _COMPARISON_LOG:
        response = item["response"]
        nodes = item["nodes"] if item.get("nodes") is not None else _collect_source_nodes(response)
        sources = sorted({_node_source(n) for n in nodes}) or ["(none exposed)"]
        md.append(f"## {item['step']}\n")
        md.append(f"**Sources touched:** {', '.join(sources)} "
                  f"({len(nodes)} chunk(s))\n")
        md.append("### Answer\n")
        md.append(str(response).strip() + "\n")
        md.append("---\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    rel = os.path.relpath(path, os.path.dirname(__file__))
    print(f"\n→ saved {rel}")
    return path


# -- Step 2: load the mixed corpus and build per-document indices -----------
# We load three heterogeneous financial documents: 1/ a 10-K excerpt, a 2/ Q3
# earnings-call transcript, and 3/ an internal risk policy. We build one vector
# index per document so the agent can later route to the right source.

def load_indices() -> dict[str, VectorStoreIndex]:
    """Build one VectorStoreIndex per source document."""
    sources = {
        "ten_k": "acme_10k_excerpt.md",
        "earnings_call": "acme_earnings_call_q3.md",
        "risk_policy": "firm_risk_policy.md",
    }
    indices: dict[str, VectorStoreIndex] = {}
    for key, filename in sources.items():
        docs = SimpleDirectoryReader(
            input_files=[os.path.join(DATA_DIR, filename)]
        ).load_data()
        indices[key] = VectorStoreIndex.from_documents(docs)
    return indices


# -- Step 3: baseline naive RAG --------------------------------------------
# A single combined index over all three documents, queried with one top-k
# retrieval pass. This is the "naive RAG" baseline: no routing, no
# decomposition, no self-check. It works for simple single-fact lookups but may
# struggles on comparative or multi-source questions across a large corpus of data.

def step_3_naive_rag() -> None:
    all_docs = SimpleDirectoryReader(input_dir=DATA_DIR).load_data()
    combined = VectorStoreIndex.from_documents(all_docs)
    engine = combined.as_query_engine(similarity_top_k=3)

    # A clean single-fact question the baseline handles well, plus the shared
    # comparative question, for which a single top-k "may" incorrectly retrieve both
    # the Q3 quick ratio and the policy threshold at once.
    single_fact = "What was Acme's full-year 2024 total revenue?"
    comparative = engine.query(COMPARISON_QUESTION)
    _log_comparison("Step 3 — Naive RAG (single combined index, top-k=3)", comparative)

    report_step(
        "Step_3_naive_RAG_baseline.md",
        "Step 3 — Naive RAG baseline",
        "One combined index over all three documents, a single top-k retrieval "
        "pass, no routing or self-checking.",
        [
            {"question": single_fact, "response": engine.query(single_fact)},
            {"question": COMPARISON_QUESTION, "response": comparative},
        ],
    )


# -- Step 4: routing with a RouterQueryEngine -------------------------------
# Wrap each per-document index as a tool with a natural-language description,
# then let an LLM selector pick the right tool for each query. This is the
# routing pattern: the agent chooses the source instead of blending all of them.

from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool


def build_query_engine_tools(indices: dict[str, VectorStoreIndex]) -> list[QueryEngineTool]:
    """Expose each document index as a described, callable tool."""
    return [
        QueryEngineTool.from_defaults(
            query_engine=indices["ten_k"].as_query_engine(similarity_top_k=3),
            name="annual_report_10k",
            description=(
                "Acme Robotics' FY2024 Form 10-K. Audited annual figures: full-year "
                "revenue, margins, balance-sheet items, year-end current and quick "
                "ratios, credit-facility covenants, and business risk factors."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=indices["earnings_call"].as_query_engine(similarity_top_k=3),
            name="q3_earnings_call",
            description=(
                "Acme Robotics' Q3 FY2024 earnings-call transcript. Quarterly figures "
                "and management commentary: Q3 revenue, Q3 quick ratio, inventory "
                "build, guidance updates, and analyst Q&A."
            ),
        ),
        QueryEngineTool.from_defaults(
            query_engine=indices["risk_policy"].as_query_engine(similarity_top_k=3),
            name="firm_risk_policy",
            description=(
                "Birchwood Asset Management's internal liquidity/counterparty risk "
                "policy. Defines required thresholds (current ratio, quick ratio, "
                "net-debt/EBITDA), covenant-awareness rules, and concentration caps."
            ),
        ),
    ]


def step_4_router(indices: dict[str, VectorStoreIndex]) -> None:
    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=build_query_engine_tools(indices),
    )

    # Three single-source questions show the selector picking the right tool;
    # the shared comparison question exposes routing's limitation, it selects
    # sources but adds no reasoning structure, so a cross-source question
    # can still get a confused single-pass answer (even when, as here, the selector
    # happens to pull both relevant sources).
    routing_questions = [
        "What is the firm's minimum required quick ratio for a new long position?",
        "What was Acme's Q3 quick ratio?",
        "What credit-facility covenants does Acme disclose in its 10-K?",
    ]
    entries = [{"question": q, "response": router.query(q)} for q in routing_questions]
    comparative = router.query(COMPARISON_QUESTION)
    entries.append({"question": COMPARISON_QUESTION, "response": comparative})
    _log_comparison("Step 4 — Router (picks ONE source via LLMSingleSelector)", comparative)

    report_step(
        "Step_4_router_query_engine.md",
        "Step 4 — Router query engine",
        "An LLMSingleSelector picks exactly one per-document tool per query.",
        entries,
    )


# -- Step 5: query decomposition with SubQuestionQueryEngine ---------------
# For a question that cannot be answered from a single document, the sub-question 
# query engine decomposes it into source-specific sub-questions, queries the appropriate 
# tool for each one, and synthesizes the results into a final answer.

from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator


def step_5_subquestion(indices: dict[str, VectorStoreIndex]) -> None:
    # By default SubQuestionQueryEngine reaches for an OpenAI-specific question
    # generator. We pass the LLM-based generator explicitly so decomposition runs
    # on our configured OpenAI reasoning model (Settings.llm) like every other step.
    engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=build_query_engine_tools(indices),
        question_gen=LLMQuestionGenerator.from_defaults(),
    )

    # A question that explicitly spans sources, plus the shared comparison
    # question, decomposition fans out to BOTH the issuer figure and the policy
    # threshold, so unlike routing it answers the cross-source question fully.
    spanning = (
        "How did Acme's quick ratio change from year-end 2024 to Q3, and does the "
        "Q3 level breach the firm's risk policy?"
    )
    comparative = engine.query(COMPARISON_QUESTION)
    _log_comparison("Step 5 — Sub-question decomposition (fans out to all sources)", comparative)

    report_step(
        "Step_5_subquestion_decomposition.md",
        "Step 5 — Sub-question decomposition",
        "SubQuestionQueryEngine splits a cross-source question into per-tool "
        "sub-questions, answers each, then synthesises.",
        [
            {"question": spanning, "response": engine.query(spanning)},
            {"question": COMPARISON_QUESTION, "response": comparative},
        ],
    )


# -- Step 6: self-correcting agentic retrieval ------------------------------
# A FunctionAgent wraps the same per-document query engines as tools. Unlike the
# fixed pipelines above, the agent decides which tools to call, reads the
# evidence, judges whether it is sufficient, and re-queries when it is not:
# the self-correcting loop. We give it a compliance-style task.

from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult

AGENT_SYSTEM_PROMPT = (
    "You are a buy-side compliance analyst. Answer using ONLY the document tools "
    "provided. First work out what evidence the question actually requires: for a "
    "policy or compliance check that usually means both the relevant figure from "
    "the issuer's filings and the matching threshold from the firm's risk policy, "
    "but a simpler factual question may need only a single figure. Before "
    "concluding, make sure you have gathered every piece of evidence the question "
    "requires; if one tool call does not give you everything, call another. If a "
    "required figure cannot be found, say so explicitly rather than estimating. "
    "Cite the source document and reporting period for every figure you use."
)


def _run_agent_sync(agent: FunctionAgent, task: str):
    """Run the agent and also collect the chunks it retrieved.

    FunctionAgent.run() must be awaited inside a running event loop, so we wrap it
    in a coroutine. The agent retrieves indirectly via tool calls, so we stream
    its ToolCallResult events and pull the source nodes out of each tool output's
    raw_output (the underlying query-engine Response). Returns (response, nodes).
    """
    collected: list = []

    async def run_agent():
        handler = agent.run(task)
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                raw = getattr(event.tool_output, "raw_output", None)
                for node in getattr(raw, "source_nodes", None) or []:
                    collected.append(node)
        return await handler

    response = asyncio.run(run_agent())
    return response, collected


def step_6_agent(indices: dict[str, VectorStoreIndex]) -> None:
    agent = FunctionAgent(
        tools=build_query_engine_tools(indices),
        llm=Settings.llm,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    # The richer compliance task shows the self-correcting loop; the shared
    # comparison question is also run so the agent appears in the side-by-side.
    task = (
        "Can we open a new long position in Acme Robotics under the firm's liquidity "
        "policy? Check the quick-ratio rule against Acme's most recent disclosed "
        "quick ratio."
    )
    task_response, task_nodes = _run_agent_sync(agent, task)
    comparative, comp_nodes = _run_agent_sync(agent, COMPARISON_QUESTION)
    _log_comparison(
        "Step 6 — Self-correcting agent (re-queries tools until sufficient)",
        comparative,
        nodes=comp_nodes,
    )

    report_step(
        "Step_6_self_correcting_agent.md",
        "Step 6 — Self-correcting agent",
        "A FunctionAgent calls the document tools, judges whether it has both the "
        "figure and the threshold, and re-queries when it does not.",
        [
            {"label": "Task", "question": task, "response": task_response, "nodes": task_nodes},
            {"question": COMPARISON_QUESTION, "response": comparative, "nodes": comp_nodes},
        ],
    )


# -- Step 7 : hybrid dense + sparse retrieval ------------------------
# Plain vector search matches on meaning but can miss an exact term; BM25 keyword
# search matches exact terms but misses paraphrases. Hybrid retrieval runs both
# over the same nodes and fuses the two ranked lists. We use QueryFusionRetriever
# with Reciprocal Rank Fusion (mode="reciprocal_rerank"), which fuses by rank and
# so sidesteps the fact that BM25 scores and cosine similarities live on different
# scales. (Requires: pip install llama-index-retrievers-bm25.)

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever


def build_hybrid_retriever() -> QueryFusionRetriever:
    """Dense + sparse (BM25) retrievers over one combined index, fused with RRF.

    Shared by Step 7 (hybrid recall on its own) and Step 9 (hybrid + reranking).
    """
    # One combined index over the whole corpus (like the naive baseline), so both
    # retrievers see every document.
    all_docs = SimpleDirectoryReader(input_dir=DATA_DIR).load_data()
    combined = VectorStoreIndex.from_documents(all_docs)

    # Dense (semantic) and sparse (lexical, BM25) retrievers over the SAME nodes, sharing the index docstore.
    vector_retriever = combined.as_retriever(similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=combined.docstore,
        similarity_top_k=5,
    )

    # num_queries=1 is pure dense+sparse hybrid with no LLM query rewriting, which
    # keeps the step cheap and deterministic. Raise it to let the configured OpenAI model generate paraphrases (the RAG-Fusion variant).
    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )


def step_7_hybrid() -> None:
    hybrid_engine = RetrieverQueryEngine.from_args(build_hybrid_retriever())

    # The shared comparison question, now answered over fused dense+sparse recall.
    comparative = hybrid_engine.query(COMPARISON_QUESTION)
    _log_comparison("Step 7 — Hybrid dense + sparse retrieval (fused recall)", comparative)

    report_step(
        "Step_7_hybrid_retrieval.md",
        "Step 7 — Hybrid dense + sparse retrieval",
        "A dense vector retriever and a BM25 sparse retriever over the same nodes, "
        "fused with reciprocal rank fusion.",
        [{"question": COMPARISON_QUESTION, "response": comparative}],
    )

# -- Step 8 : reranking ---------------------------------------------
# Vector top-k can surface loosely related chunks. A reranker re-scores the
# retrieved nodes for relevance to the query and keeps only the best ones,
# tightening the context the LLM sees. Here we use an LLM-based reranker as a
# node postprocessor.

# To make the rerank mechanic visible on its own, THIS step runs it in
# isolation: a broad top-6 retrieval from a single content-rich index (the
# 10-K), trimmed to the best 2, no routing or fusion to obscure what the reranker is doing.

# Reranking is the "precision" half of a recall-then-precision pipeline whose
# "recall" half is the Step 7 hybrid retriever. Step 9 composes the two; here we
# keep reranking in isolation so the mechanic is visible by itself. (For a local
# cross-encoder instead of an LLM reranker, swap in SentenceTransformerRerank.)

from llama_index.core.postprocessor import LLMRerank


def step_8_rerank(indices: dict[str, VectorStoreIndex]) -> None:
    # Retrieve a generous top-6, then let the reranker trim to the best 2; the
    # chunks shown in the report are the post-rerank survivors.
    reranked_engine = indices["ten_k"].as_query_engine(
        similarity_top_k=6,
        node_postprocessors=[LLMRerank(top_n=2)],
    )

    # A focused single-source question (10-K only): Reranking is a precision technique,
    # We keep it on one index rather than the cross-source question. 
    # That's why we don't run the comparison question in this step which needs to span over multiples sources.
    q = "What was Acme's gross margin in fiscal 2024 and why did it expand?"
    report_step(
        "Step_8_reranking.md",
        "Step 8 — Reranking",
        "Retrieve a broad top-6 from the 10-K, then an LLMRerank postprocessor "
        "re-scores and keeps the best 2 chunks.",
        [{"question": q, "response": reranked_engine.query(q)}],
    )


# -- Step 9 : hybrid search + reranking (recall-then-precision) -------
# Steps 7 and 8 are the two halves of one pipeline, and here we compose them.
# The Step 7 hybrid retriever fuses dense + sparse results for broad RECALL; the
# Step 8 LLMRerank postprocessor then re-scores that fused pool and keeps only the
# best nodes for PRECISION. The ONLY change from Step 7 is the node_postprocessors
# argument on the engine — exactly the recall-then-precision pipeline of Fig 10.12.


def step_9_hybrid_rerank() -> None:
    # Same fused-recall retriever as Step 7, now with the reranker attached so the
    # broad hybrid pool is trimmed to the most relevant nodes before generation.
    hybrid_rerank_engine = RetrieverQueryEngine.from_args(
        build_hybrid_retriever(),
        node_postprocessors=[LLMRerank(top_n=3)],
    )

    comparative = hybrid_rerank_engine.query(COMPARISON_QUESTION)
    _log_comparison("Step 9 — Hybrid + rerank (recall-then-precision)", comparative)

    report_step(
        "Step_9_hybrid_rerank.md",
        "Step 9 — Hybrid search + reranking",
        "The Step 7 hybrid retriever for recall, with a Step 8 LLMRerank "
        "postprocessor attached to trim the fused pool to the best nodes for precision.",
        [{"question": COMPARISON_QUESTION, "response": comparative}],
    )


# -- Step 10 : evaluating the RAG answer -----------------------------
# In regulated finance, an answer is only useful if it is both grounded in the
# retrieved context AND on point. We score it with the evaluation triad one evaluator per leg
# 1- FaithfulnessEvaluator (is the answer supported by the sources?)
# 2- AnswerRelevancyEvaluator (is the answer relevant to the question?)
# 3- ContextRelevancyEvaluator (were the retrieved chunks relevant to the question?).
# Faithfulness returns a pass/fail; the two relevancy evaluators return a graded 0-1 score.

from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    FaithfulnessEvaluator,
)


def step_10_evaluate(indices: dict[str, VectorStoreIndex]) -> None:
    engine = indices["ten_k"].as_query_engine(similarity_top_k=3)
    q = "What was Acme's net income per diluted share in fiscal 2024?"
    response = engine.query(q)

    # The evaluation triad: 1- Faithfulness grades the answer against
    # the sources, 2- Answer relevancy grades the answer against the question, and
    # 3- Context relevancy grades the retrieved chunks against the question.
    faithfulness = FaithfulnessEvaluator()
    answer_relevancy = AnswerRelevancyEvaluator()
    context_relevancy = ContextRelevancyEvaluator()

    f_result = faithfulness.evaluate_response(response=response)
    ar_result = answer_relevancy.evaluate_response(query=q, response=response)
    cr_result = context_relevancy.evaluate_response(query=q, response=response)

    report_step(
        "Step_10_evaluation.md",
        "Step 10 — RAG evaluation",
        "The evaluation triad: FaithfulnessEvaluator checks the answer is grounded "
        "in the retrieved context, AnswerRelevancyEvaluator scores the answer against "
        "the question, and ContextRelevancyEvaluator scores the retrieved context "
        "against the question.",
        [
            {
                "question": q,
                "response": response,
                "note": (
                    f"Faithful to sources? {f_result.passing} | "
                    f"Answer relevancy (0-1): {ar_result.score} | "
                    f"Context relevancy (0-1): {cr_result.score}"
                ),
            }
        ],
    )


def main() -> None:
    indices = load_indices()
    step_3_naive_rag()
    step_4_router(indices)
    step_5_subquestion(indices)
    step_6_agent(indices)
    step_7_hybrid()
    step_8_rerank(indices)
    step_9_hybrid_rerank()
    step_10_evaluate(indices)
    # Side-by-side of every step's answer to the shared comparison question.
    write_comparison_report()


if __name__ == "__main__":
    main()
