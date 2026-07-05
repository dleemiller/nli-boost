"""Dataset loading with seeded stratified subsampling.

Class DEFINITIONS (not just names) ship with every dataset: grounding the
proposer in one-line definitions was a measured win on label sets whose names
under-specify the boundary (TREC's ENTY vs DESC). val/test are drawn once,
deterministically per seed, and test is evaluated exactly once per run.
"""

from dataclasses import dataclass

import numpy as np

from .config import DataConfig


@dataclass
class Bundle:
    name: str
    task: str
    class_names: list[str]
    class_descriptions: list[str]  # "NAME: one-line definition", shown to the proposer LM
    train_texts: list[str]
    y_train: np.ndarray
    val_texts: list[str]
    y_val: np.ndarray
    test_texts: list[str]
    y_test: np.ndarray

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def stratified_indices(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """~n indices stratified by class (all of a class if smaller than its share)."""
    classes, counts = np.unique(y, return_counts=True)
    per_class = np.maximum(1, np.round(n * counts / counts.sum()).astype(int))
    picked = []
    for c, k in zip(classes, per_class):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        picked.append(idx[: min(k, len(idx))])
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


def labeled_examples(
    texts: list[str],
    y: np.ndarray,
    class_names: list[str],
    per_class: int,
    rng: np.random.Generator,
    max_chars: int = 400,
) -> list[str]:
    """Stratified '[class] text' samples shown to the proposer LM."""
    out = []
    for c, name in enumerate(class_names):
        idx = np.flatnonzero(y == c)
        for i in rng.choice(idx, size=min(per_class, len(idx)), replace=False):
            out.append(f"[{name}] {texts[i][:max_chars]}")
    return out


_NEWSGROUP_GLOSS = {
    "alt.atheism": "atheism, arguments about religion and god's existence",
    "comp.graphics": "computer graphics, image formats, rendering",
    "comp.os.ms-windows.misc": "Microsoft Windows OS issues and software",
    "comp.sys.ibm.pc.hardware": "IBM PC hardware: motherboards, drives, cards",
    "comp.sys.mac.hardware": "Apple Macintosh hardware",
    "comp.windows.x": "the X Window System on Unix",
    "misc.forsale": "items offered for sale, prices, shipping",
    "rec.autos": "cars, driving, and the auto industry",
    "rec.motorcycles": "motorcycles and riding",
    "rec.sport.baseball": "baseball teams, games, and players",
    "rec.sport.hockey": "ice hockey teams, games, and players",
    "sci.crypt": "cryptography, encryption, privacy and security policy",
    "sci.electronics": "electronic circuits and components",
    "sci.med": "medicine, health, diseases, and treatment",
    "sci.space": "spaceflight, astronomy, NASA",
    "soc.religion.christian": "Christian faith, theology, and practice",
    "talk.politics.guns": "gun ownership, control laws, and rights",
    "talk.politics.mideast": "Middle East politics and conflicts",
    "talk.politics.misc": "general political debate",
    "talk.religion.misc": "general religious debate outside specific denominations",
}

_SPECS = {
    "ag_news": dict(
        hf="fancyzhx/ag_news",
        text_field="text",
        label_field="label",
        test_split="test",
        classes=["World", "Sports", "Business", "Sci/Tech"],
        task="Classify short news articles by topic: World, Sports, Business, or Sci/Tech.",
        descriptions=[
            "World: international news, politics, conflicts, diplomacy, and events outside business/sports/tech",
            "Sports: games, matches, teams, athletes, scores, and sporting events",
            "Business: companies, markets, earnings, deals, economic policy, and finance",
            "Sci/Tech: science, technology, software, hardware, internet, and research",
        ],
    ),
    "sst2": dict(
        hf="stanfordnlp/sst2",
        text_field="sentence",
        label_field="label",
        test_split="validation",  # SST-2 test labels are withheld
        classes=["negative", "positive"],
        task="Classify movie-review sentences as expressing negative or positive sentiment.",
        descriptions=[
            "negative: the reviewer expresses dislike, criticism, or disappointment",
            "positive: the reviewer expresses praise, enjoyment, or admiration",
        ],
    ),
    "trec": dict(
        hf="CogComp/trec",
        revision="refs/convert/parquet",  # repo's loader script is unsupported
        text_field="text",
        label_field="coarse_label",
        test_split="test",
        classes=["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"],
        task=(
            "Classify questions by the type of answer they seek: abbreviation (ABBR), "
            "entity (ENTY), description/definition (DESC), human (HUM), location (LOC), "
            "or number (NUM)."
        ),
        descriptions=[
            "ABBR: asks what an abbreviation, acronym, or short form stands for or means",
            "ENTY: asks for a thing — an object, animal, color, product, event, term, or other named entity (not a person or place)",
            "DESC: asks for a definition, description, explanation, reason, or manner — 'what is/why/how' questions seeking prose answers",
            "HUM: asks for a person, group, or organization — who someone is or which people/org did something",
            "LOC: asks for a place — city, state, country, mountain, or other location",
            "NUM: asks for a number — count, date, amount, percentage, speed, age, or other numeric value",
        ],
    ),
    "20newsgroups": dict(
        hf="SetFit/20_newsgroups",
        text_field="text",
        label_field="label",
        test_split="test",
        classes=None,  # from the dataset's label_text
        task="Classify Usenet posts into one of 20 newsgroups by topic.",
        descriptions=None,  # templated from _NEWSGROUP_GLOSS
    ),
}


def load(cfg: DataConfig, seed: int) -> Bundle:
    from datasets import load_dataset  # train extra; keeps `nli_boost.data` importable without it

    spec = _SPECS[cfg.name]
    rng = np.random.default_rng(seed)

    ds = load_dataset(spec["hf"], revision=spec.get("revision"))
    train, test = ds["train"], ds[spec["test_split"]]
    tf, lf = spec["text_field"], spec["label_field"]
    train_texts, y_train = list(train[tf]), np.asarray(train[lf], dtype=np.int64)
    test_texts, y_test = list(test[tf]), np.asarray(test[lf], dtype=np.int64)

    classes = spec["classes"]
    if classes is None:
        classes = [name for _, name in sorted(set(zip(train["label"], train["label_text"])))]
    descriptions = spec["descriptions"]
    if descriptions is None:
        descriptions = [f"{c}: {_NEWSGROUP_GLOSS.get(c, c)}" for c in classes]

    idx = stratified_indices(y_train, cfg.train_size + cfg.val_size, rng)
    tr, va = idx[: cfg.train_size], idx[cfg.train_size : cfg.train_size + cfg.val_size]
    te = stratified_indices(y_test, min(cfg.test_size, len(y_test)), rng)

    return Bundle(
        name=cfg.name,
        task=spec["task"],
        class_names=classes,
        class_descriptions=descriptions,
        train_texts=[train_texts[i] for i in tr],
        y_train=y_train[tr],
        val_texts=[train_texts[i] for i in va],
        y_val=y_train[va],
        test_texts=[test_texts[i] for i in te],
        y_test=y_test[te],
    )
