import json

from conftest import FakeProposer, FakeScorer, TextOnlyDeduper, make_bundle

from nli_boost.config import DataConfig, PoolConfig, RunConfig
from nli_boost.data import _SPECS
from nli_boost.dedup import Deduper, norm_statement
from nli_boost.runner import run


def _cfg(tmp_path, **pool_kwargs) -> RunConfig:
    return RunConfig(
        run_name="t",
        data=DataConfig(name="trec"),
        pool=PoolConfig(size=8, rounds=2, patience=2, rank_sample=0, **pool_kwargs),
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )


def test_runner_end_to_end_with_fakes(tmp_path):
    cfg = _cfg(tmp_path)
    proposer = FakeProposer(
        generate_batches=[[f"f{i}" for i in range(8)]],
        refill_batches=[["f2 replacement"], ["f3 replacement"]],
    )
    results = run(
        cfg, scorer=FakeScorer(), proposer=proposer, deduper=TextOnlyDeduper(), bundle=make_bundle()
    )

    # honest protocol: pool_cv is the ONLY reported head
    assert set(results) == {"pool_cv", "cv_train_accuracy"}
    assert results["pool_cv"]["accuracy"] > 0.85

    out = tmp_path / "runs" / "t"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["results"]["pool_cv"] == results["pool_cv"]
    model = json.loads((out / "model.json").read_text())
    assert model["type"] == "nli_pool" and len(model["hypotheses"]) >= 4
    assert (out / "log.jsonl").read_text().strip()  # evolution audit trail exists
    assert (out / "costs.json").exists() and (out / "config.yaml").exists()


def test_runner_from_run_reuses_pool_without_llm(tmp_path):
    cfg = _cfg(tmp_path)
    proposer = FakeProposer(generate_batches=[[f"f{i}" for i in range(8)]], refill_batches=[[], []])
    run(cfg, scorer=FakeScorer(), proposer=proposer, deduper=TextOnlyDeduper(), bundle=make_bundle())

    cfg2 = _cfg(tmp_path, from_run="t")
    cfg2.run_name = "t_finalized"
    p2 = FakeProposer()
    run(cfg2, scorer=FakeScorer(), proposer=p2, deduper=TextOnlyDeduper(), bundle=make_bundle())
    assert not p2.generate_calls and not p2.refill_calls  # zero LM usage on finalization


def test_dedup_exact_and_covariance():
    import numpy as np
    from conftest import encode

    assert norm_statement("The text  is Brief. ") == norm_statement("the text is brief")
    ref = encode(np.random.default_rng(0).random((60, 4)))  # 60 ref texts, 4 feature columns
    d = Deduper(FakeScorer(), ref, corr_threshold=0.95)
    # exact-text: "f0 x." normalizes to "f0 x" -> duplicate dropped before any scoring
    kept, rejected = d.filter(["f0 x", "f0 x."], against=[], seen=set())
    assert kept == ["f0 x"] and rejected == ["f0 x."]
    # covariance: "f1 a" and "f1 b" read the SAME feature column -> collinear -> one dropped;
    # "f2 c" is a distinct feature -> kept
    kept2, rejected2 = d.filter(["f1 a", "f1 b", "f2 c"], against=[], seen=set())
    assert "f1 a" in kept2 and "f2 c" in kept2 and len(kept2) == 2
    assert any("corr" in s for s in rejected2)


def test_dataset_specs_are_complete():
    for name, spec in _SPECS.items():
        if spec["classes"] is not None:
            assert spec["descriptions"] is not None
            assert len(spec["classes"]) == len(spec["descriptions"])
            for c, d in zip(spec["classes"], spec["descriptions"]):
                assert d.startswith(f"{c}:")
