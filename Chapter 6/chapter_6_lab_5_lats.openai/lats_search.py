import copy
import math

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


load_dotenv()  # picks up OPENAI_API_KEY from .env


# ---------- LATS tree ----------
class Node:
    """A node in the LATS search tree.

    state is a snapshot of the market AFTER this node's action was applied
    (root holds the pre-trade snapshot). value is the running average of
    normalized evaluator scores in [0, 1].
    """

    def __init__(self, action_hint, state, parent=None):
        self.action_hint = action_hint
        self.state = state
        self.parent = parent
        self.children = []
        self.depth = 0 if parent is None else parent.depth + 1
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False
        self.proposal = None      # populated for non-root nodes
        self.observation = None   # populated for non-root nodes

    def update(self, score):
        # V_new(s) = (V_old(s) * (N(s) - 1) + r) / N(s) — paper's running average.
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
    """Walk root → leaf via max-UCT until a node is not-fully-expanded or terminal."""
    node = root
    while (
        not node.is_terminal
        and node.depth < max_depth
        and len(node.children) >= k_expand
    ):
        node = max(node.children, key=lambda c: uct(c, node.visits))
    return node


# ---------- Steps 2 + 3: Expansion + Evaluation ----------
async def expand(node, k_expand, lessons):
    """One-shot k-expand: 1 LM call → k proposals → apply each → eval each.

    Returns list of (child_node, normalized_score). Children whose proposal
    cannot be applied or whose evaluation lacks a SCORE line are dropped.
    """
    proposals = await propose_k_trims(node.state, k_expand, lessons)
    if not proposals:
        print("  [skip] expansion produced no parseable proposals")
        return []

    new_pairs = []
    for i, prop in enumerate(proposals[:k_expand], start=1):
        action_hint = f"{node.action_hint}/c{i}"
        try:
            child_state, observation = apply_trim(
                node.state, prop["positions"], float(prop["percent_each"])
            )
        except (TypeError, ValueError) as e:
            print(f"  [skip] {action_hint}: invalid proposal ({e})")
            continue

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
            f"score={score * 10:.1f}/10  V={score:.2f}  "
            f"exposure={child.state['tech_exposure_pct']:.1f}%{flag}"
        )
    return new_pairs


# ---------- Step 5: Backpropagation ----------
def backprop(leaf, score):
    """Walk leaf → root, updating value/visits at every ancestor (incl. root)."""
    node = leaf
    while node is not None:
        node.update(score)
        node = node.parent


# ---------- LATS Orchestrator: full LATS loop ----------
async def lats_search(
    n_iterations=3,
    k_expand=2,
    max_depth=3,
    lessons_cap=5,
):
    root = Node(
        "root",
        state=copy.deepcopy(INITIAL_MARKET),
        parent=None,
    )
    lessons = []

    for it in range(1, n_iterations + 1):
        print(f"\n=== Iteration {it} ===")

        # SELECTION
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
        new_pairs = await expand(leaf, k_expand, lessons)
        if not new_pairs:
            print("Expansion produced no valid children; stopping.")
            break

        # Early termination if any new child reached target.
        # Per the paper: on success, return; backprop and reflection only run on failure.
        terminal_pairs = [
            (c, s)
            for c, s in new_pairs
            if c.is_terminal
        ]
        if terminal_pairs:
            winner, winner_score = max(
                terminal_pairs,
                key=lambda p: p[1],
            )
            print(
                f"\nReached target at {winner.action_hint} "
                f"(score={winner_score * 10:.1f}/10)."
            )
            print("\n=== Final tree ===")
            print_tree(root)
            return winner

        # BACKPROPAGATION — failure path only
        for child, score in new_pairs:
            backprop(child, score)

        # REFLECTION — separate LM call on the best-scoring fallen-short child
        best_child, _ = max(new_pairs, key=lambda p: p[1])
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

    print("\n=== Final tree ===")
    print_tree(root)
    best = best_leaf(root)
    print(
        f"\nBest leaf: {best.action_hint}  V={best.value:.2f}  "
        f"exposure={best.state['tech_exposure_pct']:.1f}%"
    )
    return best


if __name__ == "__main__":
    anyio.run(lats_search, 3, 2, 3)
