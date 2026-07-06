"""Dataset loading with seeded stratified subsampling.

Class DEFINITIONS (not just names) ship with every dataset: grounding the
proposer in one-line definitions was a measured win on label sets whose names
under-specify the boundary (TREC's ENTY vs DESC). val/test are drawn once,
deterministically per seed, and test is evaluated exactly once per run.
"""

from dataclasses import dataclass

import numpy as np

from ..config import DataConfig


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


def per_class_indices(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Exactly k indices per class (all of a class if it has fewer) — the K-shot setup."""
    picked = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        picked.append(idx[: min(k, len(idx))])
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


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

# Banking77 intent labels, in the dataset's ClassLabel index order (label field is int).
_BANKING77_CLASSES = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

# CLINC150 intent labels, in the dataset's ClassLabel index order (label field `intent`, int).
# The out-of-scope class ('oos', index 42) is KEPT as its own class -> 151 classes total. Dropping
# it for a clean 150-class task would require filtering rows and remapping the remaining label ids,
# which the shared config-less loader here (and in experiments/hvexp/datasets.py) does not do. We
# load the `clinc_oos`/`plus` data via the parquet-convert branch because the script-based repo
# otherwise demands a config the shared loader cannot pass (same mechanism as `trec` above).
_CLINC150_CLASSES = [
    "restaurant_reviews",
    "nutrition_info",
    "account_blocked",
    "oil_change_how",
    "time",
    "weather",
    "redeem_rewards",
    "interest_rate",
    "gas_type",
    "accept_reservations",
    "smart_home",
    "user_name",
    "report_lost_card",
    "repeat",
    "whisper_mode",
    "what_are_your_hobbies",
    "order",
    "jump_start",
    "schedule_meeting",
    "meeting_schedule",
    "freeze_account",
    "what_song",
    "meaning_of_life",
    "restaurant_reservation",
    "traffic",
    "make_call",
    "text",
    "bill_balance",
    "improve_credit_score",
    "change_language",
    "no",
    "measurement_conversion",
    "timer",
    "flip_coin",
    "do_you_have_pets",
    "balance",
    "tell_joke",
    "last_maintenance",
    "exchange_rate",
    "uber",
    "car_rental",
    "credit_limit",
    "oos",
    "shopping_list",
    "expiration_date",
    "routing",
    "meal_suggestion",
    "tire_change",
    "todo_list",
    "card_declined",
    "rewards_balance",
    "change_accent",
    "vaccines",
    "reminder_update",
    "food_last",
    "change_ai_name",
    "bill_due",
    "who_do_you_work_for",
    "share_location",
    "international_visa",
    "calendar",
    "translate",
    "carry_on",
    "book_flight",
    "insurance_change",
    "todo_list_update",
    "timezone",
    "cancel_reservation",
    "transactions",
    "credit_score",
    "report_fraud",
    "spending_history",
    "directions",
    "spelling",
    "insurance",
    "what_is_your_name",
    "reminder",
    "where_are_you_from",
    "distance",
    "payday",
    "flight_status",
    "find_phone",
    "greeting",
    "alarm",
    "order_status",
    "confirm_reservation",
    "cook_time",
    "damaged_card",
    "reset_settings",
    "pin_change",
    "replacement_card_duration",
    "new_card",
    "roll_dice",
    "income",
    "taxes",
    "date",
    "who_made_you",
    "pto_request",
    "tire_pressure",
    "how_old_are_you",
    "rollover_401k",
    "pto_request_status",
    "how_busy",
    "application_status",
    "recipe",
    "calendar_update",
    "play_music",
    "yes",
    "direct_deposit",
    "credit_limit_change",
    "gas",
    "pay_bill",
    "ingredients_list",
    "lost_luggage",
    "goodbye",
    "what_can_i_ask_you",
    "book_hotel",
    "are_you_a_bot",
    "next_song",
    "change_speed",
    "plug_type",
    "maybe",
    "w2",
    "oil_change_when",
    "thank_you",
    "shopping_list_update",
    "pto_balance",
    "order_checks",
    "travel_alert",
    "fun_fact",
    "sync_device",
    "schedule_maintenance",
    "apr",
    "transfer",
    "ingredient_substitution",
    "calories",
    "current_location",
    "international_fees",
    "calculator",
    "definition",
    "next_holiday",
    "update_playlist",
    "mpg",
    "min_payment",
    "change_user_name",
    "restaurant_suggestion",
    "travel_notification",
    "cancel",
    "pto_used",
    "travel_suggestion",
    "change_volume",
]


def _intent_descriptions(classes: list[str], domain: str) -> list[str]:
    """Short 'NAME: one-line gloss' per intent for the proposer LM (77/150-way label sets are
    too large to hand-write). 'oos' is glossed as the catch-all out-of-scope bucket."""
    out = []
    for c in classes:
        if c == "oos":
            out.append("oos: an out-of-scope request matching none of the supported intents")
        else:
            out.append(f"{c}: a {domain} intent about {c.replace('_', ' ')}")
    return out


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
    "banking77": dict(
        hf="legacy-datasets/banking77",  # parquet ClassLabel mirror; PolyAI/banking77 is script-based
        text_field="text",
        label_field="label",
        test_split="test",
        classes=_BANKING77_CLASSES,  # 77 fine-grained banking customer-support intents
        task="Classify an online-banking customer query into one of 77 fine-grained support intents.",
        descriptions=_intent_descriptions(_BANKING77_CLASSES, "banking customer-support"),
    ),
    "clinc150": dict(
        hf="clinc/clinc_oos",
        revision="refs/convert/parquet",  # config-less load of the script repo (needs plus/small/imbalanced)
        text_field="text",
        label_field="intent",
        test_split="test",
        classes=_CLINC150_CLASSES,  # 150 assistant intents + 'oos' (index 42) => 151 classes; see note above
        task="Classify a virtual-assistant utterance into one of 150 intents, or out-of-scope (oos).",
        descriptions=_intent_descriptions(_CLINC150_CLASSES, "virtual-assistant user"),
    ),
}


def load(cfg: DataConfig, seed: int) -> Bundle:
    from datasets import load_dataset  # train extra; keeps `hypothesis_vectorizer.data` importable without it

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

    if cfg.shots_per_class is not None:  # K-shot: exactly N per class, no val
        tr = per_class_indices(y_train, cfg.shots_per_class, rng)
        va = np.array([], dtype=int)
    else:
        idx = stratified_indices(y_train, cfg.train_size + cfg.val_size, rng)
        tr, va = idx[: cfg.train_size], idx[cfg.train_size : cfg.train_size + cfg.val_size]
    # test split uses its OWN seeded RNG so the test set depends only on (dataset, seed, test_size)
    # — NOT on train_size/val_size. Two runs that differ only in train size are then comparable
    # (same held-out test) instead of silently drawing different test rows.
    te = stratified_indices(y_test, min(cfg.test_size, len(y_test)), np.random.default_rng(seed))

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
