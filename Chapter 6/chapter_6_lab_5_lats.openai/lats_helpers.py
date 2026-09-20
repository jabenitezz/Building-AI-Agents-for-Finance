import copy
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# ---------- Pure environment ----------
## This is a mock example. In practice, you can integrate your own APIs to
## retrieve portfolio exposures and associated constraints.
INITIAL_MARKET = {
    "tech_exposure_pct": 40.0,
    "target_pct": 30.0,
    "slippage_bps": [],
}


def apply_trim(state, positions, pct):
    """Apply a trim to a market state and return the post-trim snapshot.

    Args:
        state: dict with `tech_exposure_pct`, `target_pct`, and `slippage_bps`
            (list of floats). Not mutated — a deep copy is returned.
        positions: list of ticker symbol strings being trimmed.
        pct: percentage trimmed from each named position.

    Returns:
        (new_state, observation_text):
            - new_state: deep copy of `state` with `tech_exposure_pct` reduced
              and one slippage entry appended to `slippage_bps`.
            - observation_text: human-readable one-line summary of the trim.
    """
    new = copy.deepcopy(state)
    impact = 5 + len(positions) * 2 + pct * 1.5
    new["slippage_bps"].append(impact)
    new["tech_exposure_pct"] -= pct * len(positions) * 0.4
    obs = (
        f"Trimmed {positions} by {pct}% each. Slippage {impact:.1f} bps. "
        f"New tech exposure: {new['tech_exposure_pct']:.1f}%."
    )
    return new, obs


# ---------- OpenAI querying ----------
def _load_project_skill(system_prompt):
    """Load the local equivalent of the Claude project skill requested by a prompt.

    The original lab invokes Claude's Skill tool. The OpenAI port keeps the same
    skill files and injects the requested skill instructions into the model's
    instructions for that call.
    """
    lowered = system_prompt.lower()
    skill_name = None
    if "trade-evaluator" in lowered:
        skill_name = "trade-evaluator"
    elif "trade-reflector" in lowered:
        skill_name = "trade-reflector"

    if skill_name is None:
        return ""

    skill_path = (
        Path(__file__).resolve().parent
        / ".openai"
        / "skills"
        / skill_name
        / "SKILL.md"
    )
    try:
        return skill_path.read_text(encoding="utf-8")
    except OSError:
        return ""


async def _run_query(
    prompt,
    *,
    system_prompt,
    allowed_tools,
    max_turns,
    use_skills=False,
):
    """OpenAI equivalent of the original Claude Agent SDK query wrapper.

    `allowed_tools` and `max_turns` are retained in the signature so the rest
    of the lab keeps the same structure. This port does not need a separate
    agent-tool turn for project skills: when `use_skills=True`, the matching
    local skill instructions are injected directly into the OpenAI request.
    """
    del allowed_tools, max_turns

    instructions = system_prompt
    if use_skills:
        skill_text = _load_project_skill(system_prompt)
        if skill_text:
            instructions += (
                "\n\nProject skill instructions to apply for this call:\n"
                + skill_text
            )

    response = await client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=prompt,
        max_output_tokens=2048,
        store=False,
    )
    return response.output_text or ""


def _trajectory_block(parent_state, proposal, child_state, observation):
    return (
        f"Trajectory:\n"
        f"- Starting tech exposure: {parent_state['tech_exposure_pct']:.1f}%\n"
        f"- Action: trim {proposal['positions']} by {proposal['percent_each']}% each\n"
        f"- Rationale: {proposal.get('rationale', '')}\n"
        f"- Observation: {observation}\n"
        f"- Resulting tech exposure: {child_state['tech_exposure_pct']:.1f}%\n"
        f"- Target: {child_state['target_pct']}%"
    )


# ---------- LATS Step 2: one-shot k-expand ----------
PROPOSE_SYSTEM_PROMPT = """You are a trade-construction agent reducing tech exposure.

You will be asked to propose K distinct candidate trims from a given market state.
The K candidates MUST differ materially from one another — different ticker
selection, different slice size, or different concentration profile. Mode
collapse to identical or near-identical trims is a failure.

Output exactly one fenced JSON code block and no other text. The block must
contain a JSON array of K objects, each with these fields:
- positions: list of ticker symbols (strings, non-empty)
- percent_each: number, the % to trim from each named position
- rationale: short string explaining the candidate's distinct angle

Example:
```json
[
  {"positions": ["NVDA","AMD"], "percent_each": 5.0, "rationale": "concentrated semis trim"},
  {"positions": ["AAPL","MSFT","GOOGL","META"], "percent_each": 2.5, "rationale": "diversified mega-cap shave"}
]
```
"""


async def propose_k_trims(state, k_expand, lessons):
    """One LM invocation that returns k distinct trim proposals.

    Because the model emits all k proposals in a single response, candidate i
    is generated with candidates 1..i-1 already in its context — which is the
    paper's mechanism for diversity within an expansion.

    Returns a list of dicts each with keys `positions`, `percent_each`,
    `rationale`. Empty list if the model's output could not be parsed.
    """
    lessons_block = ""
    if lessons:
        lessons_block = (
            "\n\nLessons from prior attempts:\n- "
            + "\n- ".join(lessons)
        )
    prompt = (
        f"Current tech exposure: {state['tech_exposure_pct']:.1f}%. "
        f"Target: {state['target_pct']}%. "
        f"K = {k_expand}. Propose {k_expand} materially distinct trim candidates."
        f"{lessons_block}"
    )
    text = await _run_query(
        prompt,
        system_prompt=PROPOSE_SYSTEM_PROMPT,
        allowed_tools=[],
        max_turns=2,
    )
    return _parse_proposals(text)


def _parse_proposals(text):
    """Pull a JSON array of proposals from the model's output. [] on failure."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    if not candidate.startswith("["):
        a, b = candidate.find("["), candidate.rfind("]")
        if a == -1 or b <= a:
            return []
        candidate = candidate[a:b + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    valid = []
    for d in data:
        if (
            isinstance(d, dict)
            and isinstance(d.get("positions"), list)
            and len(d["positions"]) > 0
            and all(isinstance(p, str) for p in d["positions"])
            and isinstance(d.get("percent_each"), (int, float))
        ):
            valid.append(d)
    return valid


# ---------- LATS Step 3: per-child evaluation ----------
EVAL_SYSTEM_PROMPT = """You are evaluating a single tech-trim trajectory.

Apply the project skill `trade-evaluator` to the trajectory described.
End your response with a line of the exact form: SCORE: <integer 1-10>
"""


async def evaluate_proposal(parent_state, proposal, child_state, observation):
    """Score a trajectory using the trade-evaluator project skill.

    Returns the score normalized to [0, 1], or None if the LM's output did
    not contain a usable SCORE line.
    """
    text = await _run_query(
        _trajectory_block(parent_state, proposal, child_state, observation)
        + "\n\nEvaluate this trajectory.",
        system_prompt=EVAL_SYSTEM_PROMPT,
        allowed_tools=["Skill"],
        max_turns=4,
        use_skills=True,
    )
    return _parse_score(text)


def _parse_score(text):
    """Pull the integer after 'SCORE:' and normalize to [0, 1]. None on failure.

    Prefers the LAST valid line in case the model quotes the template format
    before producing its own answer.
    """
    last = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("SCORE:"):
            continue
        match = re.fullmatch(r"SCORE:\s*(10|[1-9])", stripped, re.IGNORECASE)
        if match:
            last = int(match.group(1)) / 10.0
    return last


# ---------- LATS Step 6: per-iteration reflection on failure ----------
REFLECT_SYSTEM_PROMPT = """You are reflecting on a failed tech-trim trajectory.

Apply the project skill `trade-reflector` to the trajectory described, then
end your response with a line of the exact form:
LESSON: When <situation>, do <action>.
"""


async def reflect_on_failure(parent_state, proposal, child_state, observation):
    """Generate a `When …, do …` lesson using the trade-reflector project skill.

    Returns the lesson body (the text after `LESSON:`), or None if the LM
    did not emit a usable line.
    """
    text = await _run_query(
        _trajectory_block(parent_state, proposal, child_state, observation)
        + "\n\nThe target was not reached. Reflect on this trajectory.",
        system_prompt=REFLECT_SYSTEM_PROMPT,
        allowed_tools=["Skill"],
        max_turns=4,
        use_skills=True,
    )
    return _parse_lesson(text)


def _parse_lesson(text):
    """Pull the 'LESSON: When …, do …' line. None if absent or malformed.

    Skips literal template lines (containing `<situation>` or `<action>`
    placeholders) and prefers the LAST valid match.
    """
    last = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("LESSON:"):
            continue
        body = stripped.split(":", 1)[1].strip()
        if not body.lower().startswith("when "):
            continue
        if "<situation>" in body.lower() or "<action>" in body.lower():
            continue
        last = body
    return last


# ---------- Output utilities ----------
def print_tree(node, indent=0):
    pad = "  " * indent
    flag = " [TERMINAL]" if node.is_terminal else ""
    print(
        f"{pad}{node.action_hint}: N={node.visits} V={node.value:.2f} "
        f"exposure={node.state['tech_exposure_pct']:.1f}%{flag}"
    )
    for c in node.children:
        print_tree(c, indent + 1)


def best_leaf(root):
    """DFS — return the leaf node with the highest value."""
    best = root
    stack = [root]
    while stack:
        n = stack.pop()
        if not n.children and n.value > best.value:
            best = n
        stack.extend(n.children)
    return best
