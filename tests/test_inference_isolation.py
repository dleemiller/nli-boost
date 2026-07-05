"""Guard the packaging promise: inference never needs the `train` extra.

Runs a subprocess whose import machinery BLOCKS the train-extra packages
(dspy, datasets, typer, rich, dotenv) and then exercises the whole
inference-side API. If anyone adds a top-level (non-lazy) train import to the
core modules — vectorizer, encoder, cache, dedup, config, costs — this fails.
A subprocess is required: this test process already has dspy in sys.modules.
"""

import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import sys

    class BlockTrainExtras:
        BLOCKED = {"dspy", "datasets", "typer", "rich", "dotenv", "litellm"}

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in self.BLOCKED:
                raise ImportError(f"train extra blocked in inference guard: {name}")

    sys.meta_path.insert(0, BlockTrainExtras())

    import pickle
    import tempfile
    from pathlib import Path

    # the public inference entry point, plus every core module directly
    from hypothesis_vectorizer import HypothesisVectorizer
    import hypothesis_vectorizer.cache
    import hypothesis_vectorizer.config
    import hypothesis_vectorizer.costs
    import hypothesis_vectorizer.dedup
    import hypothesis_vectorizer.encoder

    # fit with a fixed pool (the inference path), no encoder ever constructed
    vec = HypothesisVectorizer(
        hypotheses=["The text is about sports.", "The text asks for a number."],
        fixed_hypotheses=["The text is a question."],
    ).fit()
    assert vec.hypotheses_[0] == "The text is a question."
    assert len(vec.get_feature_names_out()) == 2 * len(vec.hypotheses_)

    # sklearn-native: clone params + pickle round-trip
    from sklearn.base import clone
    clone(vec)
    vec2 = pickle.loads(pickle.dumps(vec))
    assert vec2.hypotheses_ == vec.hypotheses_

    # save -> load round-trip of the inference artifact
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.json"
        vec.save(p)
        vec3 = HypothesisVectorizer.load(p)
        assert vec3.hypotheses_ == vec.hypotheses_

    print("INFERENCE_ISOLATION_OK")
    """
)


def test_inference_api_works_with_train_extras_blocked():
    proc = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "INFERENCE_ISOLATION_OK" in proc.stdout
