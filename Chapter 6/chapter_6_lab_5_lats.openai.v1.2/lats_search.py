import copy
import math
import os

import anyio
from dotenv import load_dotenv

from lats_helpers import (
    INITIAL_MARKET,
    apply_trim,
    propose_k_trims,
    evaluate_proposal,
    reflect_on_failure,
    print_tree,
    best_leaf,
)


load_dotenv()

TRACE = os.getenv("LATS_TRACE", "1").lower() not in {"0", "false", "no"}


def trace(label, message):
    if TRACE:
        print(f"  [TRACE:{label}] {message}")


# ---------- LATS tree ----------
class Node:
    """A node in the LATS search tree."""

    def __init__(self, action_hint, state, parent=None):
        self.action_hint = action_hint
        self.state = state
        self.parent = parent
        self.children = []
        self.depth = 0 if parent is None else parent.depth + 1
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False
        self.proposal = None
        self.observation = None

    def update(self, score):
        self.visits += 1
        self.value = ((self.value * (self.visits - 1)) + score) / self.visits


# ---------- Step 1: Selection (UCT descent) ----------
def uct(child, parent_visits, w=1.41):
    """UCT score with V in [0, 1]. Unvisited children get +inf."""
    if child.visits == 0:
        return float("inf")
    n_p = max(parent_visits, 1)
    return child.value + w * math.sqrt(math.log(n_p) / child.visits)


def select(root, k_expand, max_depth):
    """Walk root → leaf via max-UCT until a node is expandable or terminal."""
    node = root

    trace(
        "SELECT",
        (
            f"start at {node.action_hint}: depth={node.depth}, "
            f"N={node.visits}, V={node.value:.3f}, "
            f"children={len(node.children)}"
        ),
    )

    while (
        not node.is_terminal
        and node.depth < max_depth
        and len(node.children) >= k_expand
    ):
        trace(
            "SELECT",
            (
                f"{node.action_hint} is fully expanded "
                f"({len(node.children)}/{k_expand}); computing UCT"
            ),
        )

        ranked = []
        for child in node.children:
            score = uct(child, node.visits)
            ranked.append((child, score))
            score_text = "inf" if math.isinf(score) else f"{score:.4f}"
            trace(
                "UCT",
                (
                    f"{child.action_hint}: V={child.value:.3f}, "
                    f"N={child.visits}, parent_N={node.visits}, "
                    f"UCT={score_text}"
                ),
            )

        node, best_uct = max(ranked, key=lambda pair: pair[1])
        best_text = "inf" if math.isinf(best_uct) else f"{best_uct:.4f}"
        trace(
            "SELECT",
            f"choose {node.action_hint} because it has max UCT={best_text}",
        )

    reason = "expandable"
    if node.is_terminal:
        reason = "terminal"
    elif node.depth >= max_depth:
        reason = "max_depth"
    elif len(node.children) < k_expand:
        reason = f"not fully expanded ({len(node.children)}/{k_expand})"

    trace(
        "SELECT",
        f"stop descent at {node.action_hint}: {reason}",
    )
    return node


# ---------- Steps 2 + 3: Expansion + Evaluation ----------
async def expand(node, k_expand, lessons):
    """One-shot k-expand: 1 LM call → k proposals → apply each → eval each."""
    trace(
        "EXPAND",
        (
            f"expanding {node.action_hint}: "
            f"depth={node.depth}, exposure={node.state['tech_exposure_pct']:.1f}%"
        ),
    )

    proposals = await propose_k_trims(node.state, k_expand, lessons)
    if not proposals:
        print("  [skip] expansion produced no parseable proposals")
        return []

    new_pairs = []
    for i, prop in enumerate(proposals[:k_expand], start=1):
        action_hint = f"{node.action_hint}/c{i}"
        trace(
            "ACTION",
            (
                f"{action_hint}: positions={prop['positions']} "
                f"pct={prop['percent_each']}"
            ),
        )

        try:
            child_state, observation = apply_trim(
                node.state,
                prop["positions"],
                float(prop["percent_each"]),
            )
        except (TypeError, ValueError) as e:
            print(f"  [skip] {action_hint}: invalid proposal ({e})")
            continue

        trace("OBSERVATION", f"{action_hint}: {observation}")

        score = await evaluate_proposal(
            node.state,
            prop,
            child_state,
            observation,
        )
        if score is None:
            print(f"  [skip] {action_hint}: missing or malformed SCORE line")
            continue

        child = Node(action_hint=action_hint, state=child_state, parent=node)
        child.proposal = prop
        child.observation = observation
        child.is_terminal = (
            child.state["tech_exposure_pct"] <= child.state["target_pct"]
        )
        node.children.append(child)
        new_pairs.append((child, score))

        flag = " [TERMINAL]" if child.is_terminal else ""
        print(
            f"  [add ] {action_hint}: positions={prop['positions']} "
            f"pct={prop['percent_each']}  "
            f"score={score * 10:.1f}/10  reward={score:.2f}  "
            f"exposure={child.state['tech_exposure_pct']:.1f}%{flag}"
        )

        trace(
            "NODE",
            (
                f"created {action_hint}: depth={child.depth}, "
                f"N={child.visits}, V={child.value:.3f}, "
                f"terminal={child.is_terminal}"
            ),
        )

    return new_pairs


# ---------- Step 5: Backpropagation ----------
def backprop(leaf, score):
    """Walk leaf → root, updating value/visits at every ancestor."""
    trace(
        "BACKPROP",
        f"start from {leaf.action_hint} with reward={score:.3f}",
    )

    node = leaf
    while node is not None:
        old_n = node.visits
        old_v = node.value
        node.update(score)

        trace(
            "BACKPROP",
            (
                f"{node.action_hint}: "
                f"N {old_n}->{node.visits}, "
                f"V {old_v:.3f}->{node.value:.3f}"
            ),
        )
        node = node.parent


# ---------- LATS Orchestrator ----------
async def lats_search(
    n_iterations=16,
    k_expand=2,
    max_depth=6,
    lessons_cap=5,
):
    root = Node("root", state=copy.deepcopy(INITIAL_MARKET), parent=None)
    lessons = []

    print("=== LATS deeper-search demo v1.2 (TRACE) ===")
    print(
        f"Initial exposure={root.state['tech_exposure_pct']:.1f}%  "
        f"target={root.state['target_pct']:.1f}%"
    )
    print(
        f"Search: iterations={n_iterations}, "
        f"k_expand={k_expand}, max_depth={max_depth}, trace={TRACE}"
    )

    for it in range(1, n_iterations + 1):
        print(f"\n{'=' * 18} ITERATION {it} {'=' * 18}")

        trace(
            "ITER",
            (
                f"begin iteration {it}; lessons={len(lessons)}; "
                f"root N={root.visits}, V={root.value:.3f}"
            ),
        )

        # SELECTION
        print("\n[1] SELECTION")
        leaf = select(root, k_expand, max_depth)

        if leaf.is_terminal:
            print(
                f"Selection landed on terminal node "
                f"{leaf.action_hint}; stopping."
            )
            break
        if leaf.depth >= max_depth:
            print(
                f"Selection reached max depth at "
                f"{leaf.action_hint}; stopping."
            )
            break

        print(
            f"Selected: {leaf.action_hint} "
            f"(depth={leaf.depth}, N={leaf.visits}, V={leaf.value:.2f})"
        )

        # EXPANSION + EVALUATION
        print("\n[2] EXPANSION + [3] EVALUATION")
        new_pairs = await expand(leaf, k_expand, lessons)
        if not new_pairs:
            print("Expansion produced no valid children; stopping.")
            break

        # TERMINAL CHECK
        print("\n[4] TERMINAL CHECK")
        terminal_pairs = [(c, s) for c, s in new_pairs if c.is_terminal]
        trace(
            "TERMINAL",
            (
                f"{len(terminal_pairs)} terminal child(ren) "
                f"out of {len(new_pairs)} new child(ren)"
            ),
        )

        if terminal_pairs:
            winner, winner_score = max(
                terminal_pairs,
                key=lambda p: p[1],
            )
            print(
                f"Reached target at {winner.action_hint} "
                f"(score={winner_score * 10:.1f}/10)."
            )
            print("\n=== Final tree ===")
            print_tree(root)
            return winner

        # BACKPROPAGATION
        print("\n[5] BACKPROPAGATION")
        for child, score in new_pairs:
            backprop(child, score)

        # REFLECTION
        print("\n[6] REFLECTION")
        best_child, best_score = max(new_pairs, key=lambda p: p[1])
        trace(
            "REFLECT",
            (
                f"best failed child={best_child.action_hint}, "
                f"reward={best_score:.3f}"
            ),
        )

        lesson = await reflect_on_failure(
            best_child.parent.state,
            best_child.proposal,
            best_child.state,
            best_child.observation,
        )
        if lesson and lesson not in lessons:
            lessons.append(lesson)
            del lessons[:-lessons_cap]
            print(f"  [lesson] {lesson}")
        elif lesson:
            trace("MEMORY", "lesson already existed; not adding duplicate")
        else:
            trace("MEMORY", "no valid lesson produced")

        if TRACE:
            print("\n[TRACE:TREE] tree after iteration")
            print_tree(root)

    print("\n=== Final tree ===")
    print_tree(root)
    best = best_leaf(root)
    print(
        f"\nBest leaf: {best.action_hint}  V={best.value:.2f}  "
        f"exposure={best.state['tech_exposure_pct']:.1f}%"
    )
    return best


if __name__ == "__main__":
    anyio.run(lats_search, 16, 2, 6)
