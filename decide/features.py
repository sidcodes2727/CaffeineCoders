"""
Task-agnostic featuriser. One state in -> sparse tokens + hashed char n-grams + a dense vector.

Every decision type goes through exactly this function; only the question asked differs.

Sparse token namespaces (one shared embedding table):
  p:  words + word pairs of the primary request          d:  words of decoy clauses
  s:  whole normalised body sentences (history cues)     b:  words of the rest of the body
  g:  draft-vs-article comparison tokens                 (hashed) char 3-5-grams of request words
Dense vector:
  [0:5]    draft/article overlap ratios
  [5:12]   draft/article soft alignment through pretrained word vectors (synonym-tolerant)
  [12:268] pretrained sentence embedding (Model2Vec potion-base-8M) of the primary request
  [268:524] the same embedding of the rest of the body (history / context sentences)
  [524:526] cue similarity: max cosine of any body sentence to the mined hold-cue bank, and of the
            request to the mined always-hold-request bank (paraphrase-tolerant approve evidence)
"""
import re
import zlib
from typing import Dict, List, Tuple

import numpy as np

from . import textnorm
from .pairfeats import dense as pair_dense, pair_tokens, split_kb
from .textproc import segments

TIERS = ["free", "pro", "enterprise"]
N_PAIR = 5
N_SOFT = 7
M2V_DIM = 256
O_SOFT = N_PAIR
O_INTENT = N_PAIR + N_SOFT
O_REST = O_INTENT + M2V_DIM
O_CUE = O_REST + M2V_DIM
N_CUE = 2
N_NUM = O_CUE + N_CUE
H_BUCKETS = 8192                 # hashed char n-gram rows appended after the vocabulary

_WORD = re.compile(r"[a-z0-9']+")
_ALPHA = re.compile(r"[a-z]+")
_SENT = re.compile(r"(?<=[.?!])\s+")
_NORM = [(re.compile(r"\$\s?[\d.,]+\d"), "$x"), (re.compile(r"#\d+"), "#n"), (re.compile(r"\d+"), "0")]
_SOFT_STOP = set("""the a an of to in on at is are be by for and or with that this it as you your we our i so from
will can may use within up ahead before after each every any no new has have per then than hi hello hey there thanks
thank please sorry hope helps help just also me my us am was were been being do does did done would could should
shall let know here what which who how when if but""".split())

M2V = None                       # decide.m2v.M2V instance, set by init()
SIDE = None                      # decide.roles.SideOpener, learned side-issue opener detector
CUES = None                      # {"hold": (k,256), "flag": (k,256)} mined cue banks


def init(artifacts_dir: str, speller: bool = True) -> None:
    """Load the pretrained embedding, spell vocabulary, side-opener detector and cue banks."""
    global M2V, SIDE, CUES
    import os
    from .m2v import M2V as _M2V
    from .roles import SideOpener
    M2V = _M2V(artifacts_dir)
    if speller:
        protected = {w for w in M2V.vocab if w.isalpha()}
        textnorm.set_speller(textnorm.load_speller(artifacts_dir, protected))
    SIDE = SideOpener.load(os.path.join(artifacts_dir, "side_opener.json"), M2V)
    p = os.path.join(artifacts_dir, "cue_bank.npz")
    CUES = {k: v.astype(np.float32) for k, v in np.load(p).items()} if os.path.exists(p) else None


def cue_sims(intent: str, sentences: List[str]) -> List[float]:
    if CUES is None or M2V is None:
        return [0.0, 0.0]
    hold = 0.0
    if sentences and len(CUES["hold"]):
        S = np.stack([M2V.encode(x) for x in sentences])
        hold = float((S @ CUES["hold"].T).max())
    flag = float((CUES["flag"] @ M2V.encode(intent)).max()) if len(CUES["flag"]) and intent else 0.0
    return [hold, flag]


def parse_amount(v) -> float:
    """Lenient amount parsing: 12.5, '1,200', '$1.2k', 'USD 40', 'twelve hundred'. Unparseable -> 0."""
    if v is None or isinstance(v, bool):
        return 0.0
    if isinstance(v, (int, float)):
        f = float(v)
    else:
        s = str(v).strip().lower().replace(",", "").replace("$", "").replace("usd", "").replace("dollars", "").strip()
        mult = 1.0
        if s.endswith("k"):
            mult, s = 1000.0, s[:-1].strip()
        try:
            f = float(s) * mult
        except ValueError:
            from .pairfeats import _NUMRUN, _words_to_int
            m = _NUMRUN.search(s)
            f = float(_words_to_int(m.group(0))) if m and m.group(0) == s else 0.0
    if f != f or f in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(f, 1e9))


def tier_amount(state: Dict) -> Tuple[int, float]:
    """(tier index, amount) used by the approve head's policy gate. Unknown tier -> free."""
    tier = str(state.get("customer_tier") or "free").strip().lower()
    t = TIERS.index(tier) if tier in TIERS else 0
    return t, parse_amount(state.get("amount"))


def _words(text: str) -> List[str]:
    return _WORD.findall(text)


def _ngrams(ws: List[str]) -> List[str]:
    return ws + [a + "_" + b for a, b in zip(ws, ws[1:])]


def _norm_sentence(s: str) -> str:
    s = s.strip().rstrip(".!?")
    for rx, rep in _NORM:
        s = rx.sub(rep, s)
    return re.sub(r"\s+", " ", s)[:80]


def _char_hashes(words: List[str]) -> List[int]:
    out = set()
    for w in words:
        if len(w) < 3 or not w.isalpha():
            continue
        g = "<" + w + ">"
        for n in (3, 4, 5):
            for i in range(len(g) - n + 1):
                out.add(zlib.crc32(g[i:i + n].encode()) % H_BUCKETS)
    return sorted(out)


def _soft_align(draft: str, kb: str) -> List[float]:
    """How well each draft-only word is covered by some article word, and each word of the
    RELEVANT article sentences by some draft word, by cosine of pretrained word vectors."""
    dw = set(_ALPHA.findall(draft)) - _SOFT_STOP
    kw = set(_ALPHA.findall(kb)) - _SOFT_STOP
    kr = set(_ALPHA.findall(split_kb(draft, kb)[0])) - _SOFT_STOP
    d_only, k_only = sorted(dw - kw), sorted(kr - dw)

    def mat(ws):
        vs = [M2V.word(w) for w in ws]
        vs = [v for v in vs if v is not None]
        return np.stack(vs) if vs else None

    def cover(only, other_words):
        if not only:
            return 0.0, 1.0, 1.0
        A, B = mat(only), mat(sorted(other_words))
        if A is None or B is None:
            return 1.0, 0.0, 0.0
        best = (A @ B.T).max(1)
        return float((best < 0.5).mean()), float(best.mean()), float(best.min())

    f = list(cover(d_only, kw)) + list(cover(k_only, dw))
    a, b = M2V.encode(draft), M2V.encode(split_kb(draft, kb)[0])
    f.append(float(a @ b))
    return f


def featurize(state: Dict) -> Tuple[List[str], List[int], np.ndarray]:
    subject = textnorm.clean(state.get("subject"))
    body = textnorm.clean(state.get("body"))
    draft = textnorm.clean(state.get("draft_reply"))
    kb = textnorm.clean(state.get("kb_article"))
    prim, _, content, decoys = segments(subject, body, side_fn=SIDE)
    intent = prim or content
    iw = _words(intent)
    toks = ["p:" + t for t in _ngrams(iw)]
    toks.append("pflag:%d" % bool(prim))
    sents = [s for s in _SENT.split(body) if s.strip()]
    toks += ["s:" + _norm_sentence(s) for s in sents]
    toks += ["b:" + w for w in _words(content)]
    toks += ["d:" + w for w in _words(decoys)]
    if draft and kb:
        toks += ["g:" + t for t in pair_tokens(draft, kb)]
    hashed = _char_hashes(iw)

    x = np.zeros(N_NUM, dtype=np.float32)
    if draft and kb:
        x[:N_PAIR] = pair_dense(draft, kb)
        if M2V is not None:
            x[O_SOFT:O_SOFT + N_SOFT] = _soft_align(draft, kb)
    if M2V is not None:
        x[O_INTENT:O_INTENT + M2V_DIM] = M2V.encode(intent)
        rest = " ".join(s for s in sents if s not in intent)
        x[O_REST:O_REST + M2V_DIM] = M2V.encode(rest)
        x[O_CUE:O_CUE + N_CUE] = cue_sims(intent, [s for s in sents if s not in intent])
    return toks, hashed, x
