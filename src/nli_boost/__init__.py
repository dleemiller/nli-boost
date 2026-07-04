"""nli-boost: text classification from LM-written NLI hypotheses.

A frozen NLI cross-encoder scores whether each text entails each of ~64
LM-written hypotheses; those scores are features for a CV-disciplined
classical head. See METHOD.md for the full process and the measurements
behind each design choice.
"""

__version__ = "0.2.0"
