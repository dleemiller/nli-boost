"""GPU/LM-free tests for the llm-trees forest adaptation: flattening, generation wiring,
soft/hard induction routing, and JSON round-trip. All logic runs on hand-built TreeNodes and a
tiny fake featurizer, so no encoder or API is touched."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))

from hypothesis_vectorizer.train.proposer import TreeNode  # noqa: E402

from hvexp import forest as F  # noqa: E402
from hvexp.systems import LLMForestInduction  # noqa: E402


# --------------------------------------------------------------------------- fakes
class FakeFeaturizer:
    """Mirrors hvexp.NLIFeaturizer.features(texts, pool, score_mode). Texts encode values as
    'v0|v1|...'; a condition whose first token is 'f<i>' reads column i as its P(entail)."""

    def features(self, texts, pool, score_mode="entail_contradict"):
        vals = np.array([[float(v) for v in t.split("|")] for t in texts])
        cols = [int(h.split()[0][1:]) for h in pool]  # "f3 ..." -> column 3
        e = vals[:, cols]
        if score_mode == "entail":
            return e
        c = 1.0 - e
        if score_mode == "contrast":
            return e - c
        return np.concatenate([e, c], axis=1)


class FakeTreeProposer:
    """Queues generate_tree responses (TreeNode or None) and records the avoid seed per call."""

    def __init__(self, trees):
        self.trees = list(trees)
        self.avoid_seen = []

    def generate_tree(self, task, class_definitions, examples, avoid):
        self.avoid_seen.append(list(avoid))
        return self.trees.pop(0) if self.trees else None


def _leaf(cls):
    return TreeNode(leaf_class=cls)


def _stump(cond, yes_cls, no_cls):
    return TreeNode(condition=cond, yes=_leaf(yes_cls), no=_leaf(no_cls))


def _encode(rows):
    return ["|".join(f"{v:.6f}" for v in row) for row in rows]


# --------------------------------------------------------------------------- flatten / dedup
def test_flatten_collects_and_dedups():
    # two trees; the second repeats the first's condition modulo case + trailing period
    forest = [
        _stump("f0 asks about a location", "A", "B"),
        _stump("F0 asks about a location.", "A", "B"),
        TreeNode(condition="f1 mentions money", yes=_leaf("A"), no=_stump("f2 is angry", "B", "A")),
    ]
    conds = F.flatten_conditions(forest)
    assert conds == ["f0 asks about a location", "f1 mentions money", "f2 is angry"]  # order stable, deduped


def test_flatten_tolerates_partial_and_leaf_only_trees():
    partial = TreeNode(condition="f0 x", yes=_leaf("A"), no=None)  # missing child
    leaf_only = _leaf("B")  # a degenerate one-leaf tree
    assert F.flatten_conditions([partial, leaf_only]) == ["f0 x"]
    assert F.leaf_classes([partial, leaf_only]) == {"A", "B"}


# --------------------------------------------------------------------------- generation wiring
def test_build_forest_drops_none_and_seeds_avoid():
    t1 = _stump("f0 a", "A", "B")
    t2 = _stump("f1 b", "A", "B")
    prop = FakeTreeProposer([t1, None, t2])  # middle generation fails
    forest = F.build_forest(prop, "task", ["A: x", "B: y"], ["ex"], k_trees=3)
    assert len(forest) == 2  # None dropped
    assert prop.avoid_seen[0] == []  # first call has nothing to avoid
    assert prop.avoid_seen[1] == ["f0 a"]  # second call avoids tree 1's condition
    assert prop.avoid_seen[2] == ["f0 a"]  # third: tree 2 (the None) added nothing


# --------------------------------------------------------------------------- induction routing
def test_induction_shapes_and_rows_sum_to_one():
    forest = [_stump("f0 a", "A", "B"), _stump("f1 b", "B", "A")]
    texts = _encode([[0.2, 0.9], [0.7, 0.1], [0.5, 0.5]])
    sysm = LLMForestInduction(2, forest, ["A", "B"], FakeFeaturizer(), routing="soft")
    proba = sysm.run([], np.array([]), texts)
    assert proba.shape == (3, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_induction_hard_traces_the_leaf():
    forest = [_stump("f0 marks A", "A", "B")]
    texts = _encode([[1.0], [0.0]])  # entail=1 -> yes(A); entail=0 -> no(B)
    sysm = LLMForestInduction(2, forest, ["A", "B"], FakeFeaturizer(), routing="hard")
    pred = sysm.run([], np.array([]), texts).argmax(1)
    assert list(pred) == [0, 1]


def test_soft_and_hard_agree_on_saturated_inputs():
    forest = [_stump("f0 marks A", "A", "B"), _stump("f1 marks A", "A", "B")]
    texts = _encode([[1.0, 0.0], [0.0, 1.0]])  # every P(entail) is 0 or 1
    fz = FakeFeaturizer()
    soft = LLMForestInduction(2, forest, ["A", "B"], fz, routing="soft").run([], [], texts)
    hard = LLMForestInduction(2, forest, ["A", "B"], fz, routing="hard").run([], [], texts)
    assert np.allclose(soft, hard)


def test_soft_splits_mass_where_hard_commits():
    forest = [_stump("f0 marks A", "A", "B")]
    texts = _encode([[0.5]])  # maximal uncertainty
    fz = FakeFeaturizer()
    soft = LLMForestInduction(2, forest, ["A", "B"], fz, routing="soft").run([], [], texts)
    hard = LLMForestInduction(2, forest, ["A", "B"], fz, routing="hard").run([], [], texts)
    assert np.allclose(soft[0], [0.5, 0.5])  # soft routing splits the mass
    assert np.allclose(hard[0], [1.0, 0.0])  # hard routing commits (0.5 >= 0.5 -> yes)


def test_induction_uniform_when_no_tree_can_route():
    # a tree whose only leaf names an UNKNOWN class -> no mass reaches a known leaf -> uniform
    forest = [TreeNode(condition="f0 x", yes=_leaf("ZZZ"), no=_leaf("ZZZ"))]
    texts = _encode([[0.7]])
    proba = LLMForestInduction(3, forest, ["A", "B", "C"], FakeFeaturizer()).run([], [], texts)
    assert np.allclose(proba[0], [1 / 3, 1 / 3, 1 / 3])


# --------------------------------------------------------------------------- persistence
def test_save_load_roundtrip(tmp_path):
    forest = [
        TreeNode(condition="f0 a", yes=_leaf("A"), no=_stump("f1 b", "B", "C")),
        _leaf("A"),
    ]
    path = tmp_path / "forest.json"
    F.save_forest(forest, path)
    loaded = F.load_forest(path)
    assert F.flatten_conditions(loaded) == F.flatten_conditions(forest)
    assert F.leaf_classes(loaded) == F.leaf_classes(forest)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
