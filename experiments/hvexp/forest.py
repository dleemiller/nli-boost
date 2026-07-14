"""Zero-shot LLM decision-tree *forest* — the llm-trees adaptation (arXiv 2409.18594).

An LLM writes K traversable decision trees over the classes (each internal node is an NLI
hypothesis about the text, each leaf a class). Two downstream uses, mirroring the paper:

  * EMBEDDING — flatten every internal node's condition into a hypothesis pool and feed the NLI
    features to a learned head (reuse `systems.HVHead`). This is `flatten_conditions(forest)`.
  * INDUCTION — route text through the trees via soft NLI entailment, label-free
    (`systems.LLMForestInduction`).

Generation uses the library `Proposer` (cost tracking, retry LM, `deepseek-v4-flash`); the forest
is cached to JSON so a generated forest is a reusable asset (like `scripts/generate_pool.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis_vectorizer.dedup import norm_statement
from hypothesis_vectorizer.train.proposer import TreeNode

# ------------------------------------------------------------------ traversal / flattening


def _iter_conditions(node: TreeNode | None):
    """Yield every internal-node condition in the tree (pre-order), skipping empties."""
    if node is None:
        return
    cond = (node.condition or "").strip()
    if cond:
        yield cond
    yield from _iter_conditions(node.yes)
    yield from _iter_conditions(node.no)


def flatten_conditions(forest: list[TreeNode]) -> list[str]:
    """The EMBEDDING pool: all internal-node conditions across the forest, de-duplicated on the
    library's normalized statement form (case/whitespace/trailing-period-insensitive), first
    occurrence kept so column order is stable."""
    seen: set[str] = set()
    out: list[str] = []
    for tree in forest:
        for cond in _iter_conditions(tree):
            key = norm_statement(cond)
            if key and key not in seen:
                seen.add(key)
                out.append(cond)
    return out


def leaf_classes(forest: list[TreeNode]) -> set[str]:
    """Every class name that appears at a leaf — used to sanity-check class coverage."""
    out: set[str] = set()

    def walk(node: TreeNode | None) -> None:
        if node is None:
            return
        if node.leaf_class:
            out.add(node.leaf_class.strip())
        walk(node.yes)
        walk(node.no)

    for tree in forest:
        walk(tree)
    return out


# ------------------------------------------------------------------ generation


def build_forest(
    proposer, task: str, class_definitions: list[str], examples: list[str], k_trees: int = 5
) -> list[TreeNode]:
    """K decision trees via K `generate_tree` calls. Each call's `avoid` is seeded with the
    conditions of all trees so far, so diversity comes from temperature (LMConfig default 1.0) AND
    an explicit no-repeat instruction. Failed generations (None) are dropped — a short forest is
    still valid. Requires LM/API calls: gate on approval before running for real."""
    forest: list[TreeNode] = []
    avoid: list[str] = []
    for _ in range(k_trees):
        tree = proposer.generate_tree(task, class_definitions, examples, avoid)
        if tree is None:
            continue
        forest.append(tree)
        avoid = flatten_conditions(forest)  # union of all conditions seen so far
    return forest


# ------------------------------------------------------------------ persistence


def save_forest(forest: list[TreeNode], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [t.model_dump(exclude_none=True) for t in forest]
    path.write_text(json.dumps(payload, indent=2))


def load_forest(path: str | Path) -> list[TreeNode]:
    payload = json.loads(Path(path).read_text())
    return [TreeNode.model_validate(t) for t in payload]
