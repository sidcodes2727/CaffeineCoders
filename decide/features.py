"""
Task-agnostic featuriser. One state in -> sparse token ids + a small dense vector.

Every decision type goes through exactly this function; which head reads the
result is the only thing that differs. Token namespaces keep the three views
(request intent, history cues, draft-vs-article diff) from colliding in the
shared embedding table.
"""
import re
from typing import Dict, List, Tuple

import numpy as np

from .pairfeats import dense as pair_dense, pair_tokens
from .textproc import segments

TIERS = ["free", "pro", "enterprise"]
N_PAIR = 5
N_NUM = N_PAIR
_WORD = re.compile(r"[a-z0-9']+")
_SENT = re.compile(r"(?<=[.?!])\s+")
_NORM = [(re.compile(r"\$[\d.,]+\d"), "$x"), (re.compile(r"#\d+"), "#n"), (re.compile(r"\d+"), "0")]
_FIX = {"acount": "account", "wiht": "with", "adn": "and", "teh": "the", "yuo": "you",
        "pls": "please", "cant": "can't", "dont": "don't"}


def _words(text: str) -> List[str]:
    return [_FIX.get(w, w) for w in _WORD.findall(text.lower())]


def _ngrams(ws: List[str]) -> List[str]:
    return ws + [a + "_" + b for a, b in zip(ws, ws[1:])]


def _norm_sentence(s: str) -> str:
    s = s.lower().strip().rstrip(".!?")
    for rx, rep in _NORM:
        s = rx.sub(rep, s)
    return re.sub(r"\s+", " ", s)[:80]


def tokens(state: Dict) -> List[str]:
    subject = state.get("subject") or ""
    body = state.get("body") or ""
    prim, _subj, content = segments(subject, body)
    intent = prim or content
    toks = ["p:" + t for t in _ngrams(_words(intent))]
    toks.append("pflag:%d" % bool(prim))
    # whole-sentence tokens catch the history cues ("I raised something similar...")
    toks += ["s:" + _norm_sentence(s) for s in _SENT.split(body) if s.strip()]
    # decoy / secondary content, unigrams only
    toks += ["b:" + w for w in _words(content)]
    draft, kb = state.get("draft_reply") or "", state.get("kb_article") or ""
    if draft and kb:
        toks += ["g:" + t for t in pair_tokens(draft, kb)]
    return toks


def numeric(state: Dict) -> np.ndarray:
    """Dense trunk input: draft-vs-article scalars."""
    x = np.zeros(N_NUM, dtype=np.float32)
    draft, kb = state.get("draft_reply") or "", state.get("kb_article") or ""
    if draft and kb:
        x[:N_PAIR] = pair_dense(draft, kb)
    return x


def tier_amount(state: Dict) -> Tuple[int, float]:
    """(tier index, amount) used by the approve head's policy gate."""
    tier = str(state.get("customer_tier") or "free").lower()
    t = TIERS.index(tier) if tier in TIERS else 0
    try:
        a = float(state.get("amount") or 0.0)
    except (TypeError, ValueError):
        a = 0.0
    return t, max(a, 0.0)
