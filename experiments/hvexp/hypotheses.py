"""Hand-written, class-tagged expert hypothesis pools + zero-shot class templates.

These are the LM-free ingredients of the study:
  * `EXPERT_POOLS[ds]`   -> list of (hypothesis_text, intended_class) — the HV-fixed-expert pool
                           and the class tags the prior-aggregation head aggregates over.
  * `ZEROSHOT_TEMPLATES[ds]` -> one entailment template per class, in class-index order —
                           the zero-shot-NLI baseline (score each, argmax entailment).

Hypotheses follow the paper's rules: short, atomic, a statement *about the text*, verifiable
from the text alone, not a bare label name, semantically diverse. `intended_class` and any
rationale are metadata for analysis and for the prior head — only the text is a feature.
"""

from __future__ import annotations

from hypothesis_vectorizer.train.data import _SPECS

# ---------------------------------------------------------------------------
# TREC-6 question classification. Classes (index order matches data.py):
#   0 ABBR  1 ENTY  2 DESC  3 HUM  4 LOC  5 NUM
# ---------------------------------------------------------------------------
_TREC_EXPERT: list[tuple[str, str]] = [
    # ABBR — abbreviation / expansion
    ("The text asks what an abbreviation or acronym stands for.", "ABBR"),
    ("The text asks for the full form of an initialism.", "ABBR"),
    ("The question is about what a set of letters means.", "ABBR"),
    # ENTY — entity (thing, animal, substance, product, term, ...)
    ("The text asks for the name of a thing or object.", "ENTY"),
    ("The text asks which animal, plant, or substance is being described.", "ENTY"),
    ("The text asks for the title of a book, film, or creative work.", "ENTY"),
    ("The text asks what something is called.", "ENTY"),
    ("The text asks for the name of a color, food, or material.", "ENTY"),
    # DESC — description, definition, manner, reason
    ("The text asks for the definition of a term.", "DESC"),
    ("The text asks for an explanation of why something happens.", "DESC"),
    ("The text asks how to do something or how something works.", "DESC"),
    ("The text asks for a description of something's meaning or purpose.", "DESC"),
    # HUM — human / person / group / organization
    ("The text asks for the name of a person.", "HUM"),
    ("The text asks who did something or who is responsible.", "HUM"),
    ("The text asks about a group, team, or organization of people.", "HUM"),
    ("The text asks for the identity of an individual.", "HUM"),
    # LOC — location
    ("The text asks where something is located.", "LOC"),
    ("The text asks for the name of a place, city, or country.", "LOC"),
    ("The text asks about a geographic location.", "LOC"),
    ("The text asks which region or area something is in.", "LOC"),
    # NUM — numeric value
    ("The text asks for a number or a count.", "NUM"),
    ("The text asks how many of something there are.", "NUM"),
    ("The text asks for a date, year, or period of time.", "NUM"),
    ("The text asks for a distance, size, or measurement.", "NUM"),
    ("The text can be answered with a numeric value.", "NUM"),
]

# One entailment template per class, in class-index order, for zero-shot NLI.
_TREC_ZEROSHOT = [
    "The text asks what an abbreviation stands for.",  # ABBR
    "The text asks for the name of an entity or thing.",  # ENTY
    "The text asks for a description, definition, or reason.",  # DESC
    "The text asks about a person or group of people.",  # HUM
    "The text asks about a location or place.",  # LOC
    "The text asks for a number, quantity, or date.",  # NUM
]

# ---------------------------------------------------------------------------
# AG News topic classification. Classes: 0 World  1 Sports  2 Business  3 Sci/Tech
# ---------------------------------------------------------------------------
_AGNEWS_EXPERT: list[tuple[str, str]] = [
    ("The article is about international news, politics, or world events.", "World"),
    ("The article describes conflict, diplomacy, or a government.", "World"),
    ("The article is about a sports team, match, or athlete.", "Sports"),
    ("The article reports a game result or a sporting competition.", "Sports"),
    ("The article is about a company, market, or the economy.", "Business"),
    ("The article discusses finance, stocks, or corporate earnings.", "Business"),
    ("The article is about science, technology, or computing.", "Sci/Tech"),
    ("The article describes a scientific discovery or a new technology.", "Sci/Tech"),
]
_AGNEWS_ZEROSHOT = [
    "This article is about world news and politics.",
    "This article is about sports.",
    "This article is about business and finance.",
    "This article is about science and technology.",
]

# ---------------------------------------------------------------------------
# SST-2 sentiment. Classes: 0 negative  1 positive
# ---------------------------------------------------------------------------
_SST2_EXPERT: list[tuple[str, str]] = [
    ("The text expresses a negative opinion.", "negative"),
    ("The reviewer dislikes the film.", "negative"),
    ("The text criticizes or complains about something.", "negative"),
    ("The text conveys disappointment or frustration.", "negative"),
    ("The text expresses a positive opinion.", "positive"),
    ("The reviewer praises or recommends the film.", "positive"),
    ("The text conveys enjoyment or admiration.", "positive"),
    ("The text is enthusiastic or approving.", "positive"),
]
_SST2_ZEROSHOT = [
    "The text expresses a negative sentiment.",
    "The text expresses a positive sentiment.",
]

# ---------------------------------------------------------------------------
# Banking77 intent classification (77 classes). Too many for per-class hypotheses, so these are
# general intent-FAMILY probes, each tagged with a representative class that exists in the label set.
# ---------------------------------------------------------------------------
_BANKING77_EXPERT: list[tuple[str, str]] = [
    ("The text asks to activate or set up a card.", "activate_my_card"),
    ("The text asks when a card will arrive or be delivered.", "card_arrival"),
    ("The text reports that a card is not working.", "card_not_working"),
    ("The text reports a problem with contactless or tap payments.", "contactless_not_working"),
    ("The text asks to change or reset a card PIN.", "change_pin"),
    ("The text says the PIN is blocked or the passcode was forgotten.", "pin_blocked"),
    ("The text reports a lost, stolen, or compromised card.", "lost_or_stolen_card"),
    ("The text says a card was retained or swallowed by a machine.", "card_swallowed"),
    ("The text requests a refund or reports a refund not showing up.", "request_refund"),
    ("The text reports a card payment that was declined.", "declined_card_payment"),
    ("The text asks about a fee or an extra charge on a payment or statement.", "card_payment_fee_charged"),
    ("The text reports being charged twice or an unexpected extra charge.", "transaction_charged_twice"),
    ("The text asks about the exchange rate applied to a transaction.", "exchange_rate"),
    ("The text asks how long a transfer takes or why it is delayed.", "transfer_timing"),
    ("The text asks to cancel a transfer or reports a failed transfer.", "cancel_transfer"),
    ("The text asks about top-up limits or how much can be added.", "top_up_limits"),
    ("The text asks to verify identity or provide proof of identity.", "verify_my_identity"),
    ("The text asks to close, terminate, or delete an account.", "terminate_account"),
    ("The text asks how to get or order a physical card.", "get_physical_card"),
    ("The text asks about ATM support or cash withdrawal charges.", "atm_support"),
    ("The text asks about Apple Pay, Google Pay, or adding a card to a phone.", "apple_pay_or_google_pay"),
    ("The text asks whether a card works in a particular country.", "country_support"),
    ("The text asks whether Visa or Mastercard is supported.", "visa_or_mastercard"),
    ("The text asks to edit or update personal details.", "edit_personal_details"),
]
_BANKING77_ZEROSHOT: list[str] = [
    f"The customer's intent is {name.replace('_', ' ').lower()}." for name in _SPECS["banking77"]["classes"]
]

# ---------------------------------------------------------------------------
# CLINC150 intent classification (150 assistant intents + out-of-scope). General intent-FAMILY
# probes tagged with a representative class that exists in the label set.
# ---------------------------------------------------------------------------
_CLINC150_EXPERT: list[tuple[str, str]] = [
    ("The text asks about an account balance.", "balance"),
    ("The text asks to transfer or move money.", "transfer"),
    ("The text asks to pay a bill or about a bill.", "pay_bill"),
    ("The text asks about a credit limit or credit score.", "credit_limit"),
    ("The text asks about a minimum payment owed.", "min_payment"),
    ("The text asks about recent spending or transactions.", "spending_history"),
    ("The text reports a lost card or asks to freeze an account.", "report_lost_card"),
    ("The text reports fraud or a card being declined.", "report_fraud"),
    ("The text asks to order a new or replacement card.", "new_card"),
    ("The text asks about a currency exchange rate.", "exchange_rate"),
    ("The text asks to book a flight or a hotel.", "book_flight"),
    ("The text asks to rent or reserve a car.", "car_rental"),
    ("The text asks about a flight status or travel plan.", "flight_status"),
    ("The text asks to make a restaurant reservation.", "restaurant_reservation"),
    ("The text asks for the weather forecast.", "weather"),
    ("The text asks to set an alarm, timer, or reminder.", "reminder"),
    ("The text asks to play music or change the volume.", "play_music"),
    ("The text asks to translate a phrase or define a word.", "translate"),
    ("The text asks for directions or how far away something is.", "directions"),
    ("The text asks the assistant to do a calculation or conversion.", "calculator"),
    ("The text asks the assistant to tell a joke or a fun fact.", "tell_joke"),
    ("The text is a greeting or small talk with the assistant.", "greeting"),
    ("The text asks the assistant about itself or what it can do.", "what_can_i_ask_you"),
    ("The text is an unrelated request that fits none of the supported intents.", "oos"),
]
_CLINC150_ZEROSHOT: list[str] = [
    f"The user's intent is {name.replace('_', ' ').lower()}." for name in _SPECS["clinc150"]["classes"]
]


EXPERT_POOLS: dict[str, list[tuple[str, str]]] = {
    "trec": _TREC_EXPERT,
    "ag_news": _AGNEWS_EXPERT,
    "sst2": _SST2_EXPERT,
    "banking77": _BANKING77_EXPERT,
    "clinc150": _CLINC150_EXPERT,
}

ZEROSHOT_TEMPLATES: dict[str, list[str]] = {
    "trec": _TREC_ZEROSHOT,
    "ag_news": _AGNEWS_ZEROSHOT,
    "sst2": _SST2_ZEROSHOT,
    "banking77": _BANKING77_ZEROSHOT,
    "clinc150": _CLINC150_ZEROSHOT,
}


def expert_pool(dataset: str) -> tuple[list[str], list[str]]:
    """(hypothesis texts, intended class name per hypothesis) for a dataset."""
    pairs = EXPERT_POOLS[dataset]
    return [h for h, _ in pairs], [c for _, c in pairs]
