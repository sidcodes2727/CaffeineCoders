"""Vocabulary + encoding of featuriser output into id/weight arrays."""
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .features import featurize


# Per-namespace weight 1/sqrt(count) keeps long texts from drowning short ones - except for whole
# sentences (s:), where one history cue must count the same however many other sentences surround it.
FIXED_WEIGHT = {"s": 0.5}


def ns_weights(toks: List[str], n_hashed: int) -> Tuple[List[float], float]:
    cnt = Counter(t.split(":", 1)[0] for t in toks)
    w = []
    for t in toks:
        ns = t.split(":", 1)[0]
        w.append(FIXED_WEIGHT.get(ns, 1.0 / np.sqrt(cnt[ns])))
    return w, (1.0 / np.sqrt(n_hashed) if n_hashed else 0.0)


def build_vocab(states: List[Dict], min_df: int = 2) -> Dict[str, int]:
    df = Counter()
    for s in states:
        df.update(set(featurize(s)[0]))
    keep = sorted(t for t, c in df.items() if c >= min_df)
    return {t: i for i, t in enumerate(keep)}


def encode_feats(feats, vocab: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embedding-row ids, per-row weights, dense vector. Hashed n-grams live after the vocabulary."""
    toks, hashed, x = feats
    w, wh = ns_weights(toks, len(hashed))
    ids, ws = [], []
    for t, wi in zip(toks, w):
        i = vocab.get(t)
        if i is not None:
            ids.append(i)
            ws.append(wi)
    V = len(vocab)
    ids.extend(V + h for h in hashed)
    ws.extend([wh] * len(hashed))
    if not ids:                               # keep EmbeddingBag happy
        ids, ws = [0], [0.0]
    return np.asarray(ids, dtype=np.int64), np.asarray(ws, dtype=np.float32), x


def encode(state: Dict, vocab: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return encode_feats(featurize(state), vocab)
