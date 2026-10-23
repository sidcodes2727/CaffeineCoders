"""
Input hygiene applied to every text field before any feature is computed (training and serving).

  clean(text)  unicode-normalise, ascii quotes/dashes, drop emoji/symbols/control chars, lowercase,
               expand contractions, collapse whitespace, truncate, then spell-correct.

Spell correction is SymSpell-style at edit distance 1 (delete / insert / substitute / transpose)
against the training-text vocabulary. A word is only rewritten when it is unknown both to the
training vocabulary and to the pretrained WordPiece vocabulary, so real English words the data
never used ("parcel", "passphrase") are left alone.
"""
import json
import os
import re
import unicodedata
from typing import Dict, Iterable, Optional, Set

MAX_CHARS = 2000          # per field; the longest real ticket body is ~420 chars
_QUOTES = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'",
           "“": '"', "”": '"', "„": '"', "″": '"',
           "–": "-", "—": "-", "−": "-", "…": "...", " ": " "}
_CONTRACT = [(re.compile(r"\bcan't\b"), "cannot"), (re.compile(r"\bwon't\b"), "will not"),
             (re.compile(r"\bshan't\b"), "shall not"), (re.compile(r"n't\b"), " not"),
             (re.compile(r"'ll\b"), " will"), (re.compile(r"'ve\b"), " have"), (re.compile(r"'re\b"), " are"),
             (re.compile(r"\bi'm\b"), "i am"), (re.compile(r"'d\b"), " would"),
             (re.compile(r"\b(it|that|there|what|here|who|where|how|he|she)'s\b"), r"\1 is")]
_WORD = re.compile(r"[a-z]+")
_WS = re.compile(r"\s+")


def _strip_symbols(t: str) -> str:
    out = []
    for ch in t:
        cat = unicodedata.category(ch)
        if cat in ("So", "Sk", "Cs", "Co", "Cn") or (cat.startswith("C") and ch not in "\t\n\r"):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def basic(text) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    t = unicodedata.normalize("NFKC", text[:MAX_CHARS * 2])
    t = "".join(_QUOTES.get(ch, ch) for ch in t)
    t = _strip_symbols(t).lower()
    for rx, rep in _CONTRACT:
        t = rx.sub(rep, t)
    return _WS.sub(" ", t).strip()[:MAX_CHARS]


class Speller:
    def __init__(self, freq: Dict[str, int], protected: Optional[Set[str]] = None):
        self.freq = freq
        self.protected = protected or set()
        self.index: Dict[str, list] = {}
        for w in freq:
            for d in self._deletes(w) | {w}:
                self.index.setdefault(d, []).append(w)
        self._cache: Dict[str, str] = {}

    @staticmethod
    def _deletes(w: str):
        return {w[:i] + w[i + 1:] for i in range(len(w))}

    def correct(self, w: str) -> str:
        if len(w) < 4 or w in self.freq or w in self.protected:
            return w
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        cands = set(self.index.get(w, ()))                     # w is a deletion of a known word
        for d in self._deletes(w):
            if d in self.freq:
                cands.add(d)                                   # w has one extra character
            cands.update(self.index.get(d, ()))                # substitution / transposition
        best = max(cands, key=lambda c: (self.freq[c], c)) if cands else w
        if len(self._cache) < 100000:
            self._cache[w] = best
        return best

    def fix(self, t: str) -> str:
        return _WORD.sub(lambda m: self.correct(m.group(0)), t)

    @classmethod
    def build(cls, texts: Iterable[str], protected: Optional[Set[str]] = None, min_count: int = 3):
        cnt: Dict[str, int] = {}
        for t in texts:
            for w in _WORD.findall(basic(t)):
                cnt[w] = cnt.get(w, 0) + 1
        return cls({w: c for w, c in cnt.items() if c >= min_count and len(w) >= 3}, protected)

    def save(self, path: str) -> None:
        json.dump(self.freq, open(path, "w"))


SPELL: Optional[Speller] = None


def set_speller(sp: Optional[Speller]) -> None:
    global SPELL
    SPELL = sp


def load_speller(artifacts_dir: str, protected: Optional[Set[str]] = None) -> Optional[Speller]:
    p = os.path.join(artifacts_dir, "spell_vocab.json")
    return Speller(json.load(open(p)), protected) if os.path.exists(p) else None


def clean(text) -> str:
    t = basic(text)
    return SPELL.fix(t) if SPELL is not None else t
