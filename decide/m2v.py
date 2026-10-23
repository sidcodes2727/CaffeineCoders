"""
Pretrained static sentence embedding: Model2Vec `minishlab/potion-base-8M` (MIT licence).

Pure-python re-implementation of its BERT-uncased WordPiece tokenizer + mean pooling, so serving
needs only numpy. The embedding table (29,528 x 256) is exported once by tools/export_m2v.py into
artifacts/m2v_emb.npy (+ m2v_vocab.txt); any token mapping / per-token weights are folded into it.
encode() reproduces StaticModel.encode() (checked in export_m2v.py).
"""
import os
import re
import unicodedata
from typing import Dict, List, Optional

import numpy as np


_ASCII_TOK = re.compile(r"[a-z0-9]+|[!-/:-@\[-`{-~]")
_ASCII_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or
            0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or
            0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class M2V:
    def __init__(self, artifacts_dir: str):
        self.emb = np.load(os.path.join(artifacts_dir, "m2v_emb.npy")).astype(np.float32)
        with open(os.path.join(artifacts_dir, "m2v_vocab.txt"), encoding="utf-8") as fh:
            self.vocab: Dict[str, int] = {w.rstrip("\n"): i for i, w in enumerate(fh)}
        self.dim = self.emb.shape[1]
        self._cache: Dict[str, List[int]] = {}
        self._wcache: Dict[str, Optional[np.ndarray]] = {}

    # BertNormalizer(clean_text, handle_chinese_chars, lowercase, strip_accents=None -> follows lowercase)
    @staticmethod
    def _normalize(text: str) -> str:
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"):
                continue
            if ch in "\t\n\r" or unicodedata.category(ch) == "Zs":
                out.append(" ")
            elif _is_cjk(cp):
                out.append(" %s " % ch)
            else:
                out.append(ch)
        t = "".join(out).lower()
        t = unicodedata.normalize("NFD", t)
        return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")

    @staticmethod
    def _pretokenize(text: str) -> List[str]:
        words, cur = [], []
        for ch in text:
            if ch.isspace():
                if cur:
                    words.append("".join(cur)); cur = []
            elif _is_punct(ch):
                if cur:
                    words.append("".join(cur)); cur = []
                words.append(ch)
            else:
                cur.append(ch)
        if cur:
            words.append("".join(cur))
        return words

    def _wordpiece(self, word: str) -> List[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids, start = [], 0
        if len(word) > 100:
            ids = []
        else:
            while start < len(word):
                end, cur = len(word), None
                while start < end:
                    piece = word[start:end] if start == 0 else "##" + word[start:end]
                    if piece in self.vocab:
                        cur = self.vocab[piece]
                        break
                    end -= 1
                if cur is None:
                    ids = []                 # whole word -> [UNK], which Model2Vec drops
                    break
                ids.append(cur)
                start = end
        if len(self._cache) < 200000:
            self._cache[word] = ids
        return ids

    def ids(self, text: str) -> List[int]:
        out: List[int] = []
        if text.isascii():
            # fast path, identical result for ASCII: lowercase, drop control chars, split on
            # whitespace and every punctuation character
            words = _ASCII_TOK.findall(_ASCII_CTRL.sub("", text).lower())
        else:
            words = self._pretokenize(self._normalize(text))
        for w in words:
            out.extend(self._wordpiece(w))
        return out

    def encode(self, text: str, max_tokens: Optional[int] = 512) -> np.ndarray:
        ids = self.ids(text)[:max_tokens]
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        v = self.emb[ids].mean(0)
        return v / (np.linalg.norm(v) + 1e-32)

    def word(self, w: str) -> Optional[np.ndarray]:
        """Unit vector for a single word (mean of its word pieces), or None if unknown."""
        if w in self._wcache:
            return self._wcache[w]
        ids = self.ids(w)
        v = None
        if ids:
            v = self.emb[ids].mean(0)
            v = v / (np.linalg.norm(v) + 1e-32)
        if len(self._wcache) < 100000:
            self._wcache[w] = v
        return v
