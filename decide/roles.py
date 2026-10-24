"""
Learned side-issue opener detector.

The keyword rules in textproc.DECOY only know the phrasings they list. This tiny logistic model reads
the opening clause of a sentence ("Quick aside:", "Not a big deal, but") through the pretrained
sentence embedding and says whether it introduces a SIDE issue. It is fitted at training time on the
phrase lists below (known markers + generic de-prioritising openers vs primary markers, greetings and
ordinary request openings). The held-out test phrasings in tests/make_complex.py are NOT in these
lists. Used by textproc.segments only when a ticket has no primary marker.
"""
import json
import os
import re
from typing import Callable, Optional

import numpy as np

SIDE_OPENERS = [
    "less important, but while i am writing", "separately, and this is less urgent", "on a related note",
    "also worth mentioning", "some context first", "for background", "for context",
    "btw", "side note", "minor thing", "separately", "p.s.", "low priority", "unrelated", "by the way", "no rush",
    "not urgent", "not that important, but", "while i have you", "one more thing", "another small thing",
    "one minor point", "less pressing", "secondary question", "a smaller issue", "not a priority",
]
NOT_SIDE = [
    "priority for me is", "the main thing i need resolved is", "what i actually need help with is this",
    "my real question is this", "most importantly", "the part i need you to action is", "urgent", "important",
    "asap", "please help urgently", "this is critical", "main issue",
    "hi team", "hello", "hey", "hi there", "good morning", "dear support", "thanks", "regards", "please advise",
    "i was charged twice", "my order has not arrived", "the export job fails", "i cannot log in", "can you help",
    "could you send", "the amount in question is", "i am on the pro plan", "i am using chrome",
    "my colleague on the enterprise plan", "i need a refund", "please cancel my account", "the app crashes",
    "the invoice for march", "since yesterday", "every time i open it", "as discussed", "unfortunately",
]
_PREFIX = re.compile(r"^\s*([^,:.?!]{2,40}?)\s*[,:]\s+\S")


def opening_clause(sentence: str) -> Optional[str]:
    """The clause before the first comma/colon, if it is short (<= 7 words)."""
    m = _PREFIX.match(sentence)
    if not m or len(m.group(1).split()) > 7:
        return None
    return m.group(1)


class SideOpener:
    def __init__(self, w: np.ndarray, b: float, m2v, threshold: float = 0.5):
        self.w, self.b, self.m2v, self.th = w, b, m2v, threshold

    def prob(self, phrase: str) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.m2v.encode(phrase) @ self.w + self.b))))

    def __call__(self, phrase: str) -> bool:
        return self.prob(phrase) > self.th

    @classmethod
    def fit(cls, m2v):
        from sklearn.linear_model import LogisticRegression
        X = np.stack([m2v.encode(t) for t in SIDE_OPENERS + NOT_SIDE])
        y = np.array([1] * len(SIDE_OPENERS) + [0] * len(NOT_SIDE))
        lr = LogisticRegression(C=10, max_iter=5000).fit(X, y)
        return cls(lr.coef_[0].astype(np.float32), float(lr.intercept_[0]), m2v)

    def save(self, path: str) -> None:
        json.dump({"w": self.w.tolist(), "b": self.b, "threshold": self.th}, open(path, "w"))

    @classmethod
    def load(cls, path: str, m2v):
        if not os.path.exists(path):
            return None
        d = json.load(open(path))
        return cls(np.asarray(d["w"], dtype=np.float32), float(d["b"]), m2v, float(d.get("threshold", 0.5)))


SideFn = Callable[[str], bool]
