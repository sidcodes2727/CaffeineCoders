"""
Grounded cascade: the shared network answers; the cross-encoder is consulted only on hard cases.

Hard case = the draft uses content words that appear nowhere in the KB article (possible paraphrase
the network cannot read), or the network is not confident. On a hard case a small logistic combiner
mixes [network log-odds, cross-encoder logit, size of the lexical gap, number-mismatch flag]; it was
fitted on held-out tickets (clean, augmented and paraphrased) that neither model trained on.
"""
import re
from typing import Dict, List

import numpy as np

from .pairfeats import STOP, _isnum, toks
from . import textnorm

_ALPHA = re.compile(r"^[a-z]+$")


def gap_features(draft: str, kb: str) -> Dict[str, float]:
    d = textnorm.clean(draft)
    k = textnorm.clean(kb)
    ds = set(toks(d)) - STOP
    ks = set(toks(k)) - STOP
    d_only = ds - ks
    words = [t for t in d_only if _ALPHA.match(t)]
    nums_d = {t for t in ds if _isnum(t)}
    nums_k = {t for t in ks if _isnum(t)}
    return {"gap": float(len(words)), "num_mismatch": float(bool(nums_d - nums_k))}


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def design(p_nn: float, ce: float, g: Dict[str, float]) -> List[float]:
    lg = _logit(p_nn)
    gap = min(g["gap"], 6.0) / 6.0
    return [lg, ce, gap, g["num_mismatch"], lg * gap, ce * gap]


class Cascade:
    def __init__(self, spec: dict, encoder):
        self.enc = encoder
        self.coef = np.asarray(spec["coef"], dtype=np.float64)
        self.bias = float(spec["bias"])
        self.min_gap = float(spec["min_gap"])
        self.unsure = float(spec["unsure"])
        self.thr = float(spec["abstain"])

    def triggered(self, p_nn: float, g: Dict[str, float]) -> bool:
        return g["gap"] >= self.min_gap or max(p_nn, 1 - p_nn) < self.unsure

    def combine(self, p_nn: float, ce: float, g: Dict[str, float]) -> float:
        z = float(np.dot(self.coef, design(p_nn, ce, g)) + self.bias)
        return float(1.0 / (1.0 + np.exp(-z)))

    def __call__(self, state: dict, p_nn: float):
        """-> (p, used_cross_encoder)"""
        draft, kb = state.get("draft_reply") or "", state.get("kb_article") or ""
        if not draft or not kb:
            return p_nn, False
        g = gap_features(draft, kb)
        if not self.triggered(p_nn, g):
            return p_nn, False
        return self.combine(p_nn, self.enc.logit(kb, draft), g), True
