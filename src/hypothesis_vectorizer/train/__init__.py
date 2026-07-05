"""Training side: hypothesis generation, evolution, and the experiment CLI.

Everything here may import the `train` extras (dspy, datasets, typer, rich).
Inference (`hypothesis_vectorizer.HypothesisVectorizer` with a fixed pool) must
never need this subpackage.
"""
