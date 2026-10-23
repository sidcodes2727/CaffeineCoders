"""Vocabulary + encoding of featuriser output into id/weight arrays."""
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .features import numeric, tokens


def ns_weights(toks: List[str]) -> List[float]:
    cnt = Counter(t.split(":", 1)[0] for t in toks)
    return [1.0 / np.sqrt(cnt[t.split(":", 1)[0]]) for t in toks]


def build_vocab(states: List[Dict], min_df: int = 2) -> Dict[str, int]:
    df = Counter()
    for s in states:
        df.update(set(tokens(s)))
    keep = sorted(t for t, c in df.items() if c >= min_df)
    return {t: i for i, t in enumerate(keep)}


def encode(state: Dict, vocab: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    toks = tokens(state)
    w = ns_weights(toks)
    ids, ws = [], []
    for t, wi in zip(toks, w):
        i = vocab.get(t)
        if i is not None:
            ids.append(i)
            ws.append(wi)
    if not ids:                               # keep EmbeddingBag happy
        ids, ws = [0], [0.0]
    return np.asarray(ids, dtype=np.int64), np.asarray(ws, dtype=np.float32), numeric(state)
