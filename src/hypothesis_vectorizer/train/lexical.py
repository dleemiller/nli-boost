"""Optional TF-IDF lexical channel (training side), concatenated with hypothesis features.

Motivation (measured): TF-IDF standalone reaches 0.828 on TREC and 0.565 on
20NG — signal the NLI encoder may not carry. If complementary, concatenation
buys points; if subsumed, the CV head ignores the extra columns. It is just
TfidfVectorizer -> TruncatedSVD (fit on TRAIN ONLY — no leakage); at inference
the same channel is composed via sklearn FeatureUnion around HypothesisVectorizer.
"""

import numpy as np

from ..config import LexicalConfig


class LexicalFeaturizer:
    def __init__(self, cfg: LexicalConfig, seed: int):
        self.cfg = cfg
        self.seed = seed
        self._pipeline = None

    def fit(self, train_texts: list[str]) -> "LexicalFeaturizer":
        if self.cfg.kind == "tfidf_svd":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import make_pipeline

            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            tf = vec.fit_transform(train_texts)
            dims = min(self.cfg.dims, tf.shape[1] - 1)  # SVD needs dims < vocab size
            self._pipeline = make_pipeline(vec, TruncatedSVD(n_components=dims, random_state=self.seed))
            self._pipeline.fit(train_texts)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.cfg.kind == "tfidf_svd":
            return np.asarray(self._pipeline.transform(texts), dtype=np.float32)
        raise ValueError(f"no lexical features for kind={self.cfg.kind!r}")
