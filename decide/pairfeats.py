"""Draft-vs-KB comparison features for the `grounded` decision (pure python)."""
import re

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i >= 2}
_SCALES = {"hundred": 100, "thousand": 1000}
_NW = "|".join(sorted(list(_UNITS) + list(_TENS) + list(_SCALES), key=len, reverse=True))
# a run of number words: "three hundred", "sixty-five", "one thousand", "a hundred"
_NUMRUN = re.compile(r"\b(?:(?:a|an)[\s-]+(?=hundred|thousand))?(?:%s)(?:[\s-]+(?:and[\s-]+)?(?:%s))*\b" % (_NW, _NW))
_NUM = r"(\d+(?:[.,]\d+)?|an|a)"


def _words_to_int(run: str) -> int:
    total = cur = 0
    for w in re.split(r"[\s-]+", run):
        if w in ("a", "an", "and"):
            continue
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur = (cur or 1) * 100
        elif w == "thousand":
            total += (cur or 1) * 1000
            cur = 0
    return total + cur
_UNIT = {"minute": 1, "minutes": 1, "hour": 60, "hours": 60, "day": 1440, "days": 1440,
         "week": 10080, "weeks": 10080, "month": 43200, "months": 43200}
# "a quarter of an hour", "a fortnight"
_SPECIAL = [(re.compile(r"\ba quarter of an hour\b"), "dur_15"),
            (re.compile(r"\bhalf an hour\b"), "dur_30"),
            (re.compile(r"\b(?:a|one) year\b"), "dur_525600"),
            (re.compile(r"\ba fortnight\b"), "dur_20160")]
_DUR = re.compile(r"\b%s[\s-]+(business[\s-]+)?(minutes?|hours?|days?|weeks?|months?)\b" % _NUM)
_RATE = re.compile(r"\b%s\s+requests\s+per\s+minute\b" % _NUM)
_TOK = re.compile(r"dur_\d+|rate_\d+|n\d+|[a-z]+")


def _val(s):
    s = s.replace(",", "")
    return 1.0 if s in ("a", "an") else float(s)


def canon(text: str) -> str:
    t = text.lower()
    for rx, rep in _SPECIAL:
        t = rx.sub(rep, t)
    t = t.replace("first of the next month", "firstnextmonth")
    t = _NUMRUN.sub(lambda m: str(_words_to_int(m.group(0))), t)

    def dur(m):
        n = _val(m.group(1))
        unit = _UNIT[m.group(3)]
        minutes = int(round(n * unit))
        return " dur_%d " % minutes
    t = _RATE.sub(lambda m: " rate_%d " % int(_val(m.group(1))), t)
    t = _DUR.sub(dur, t)
    t = re.sub(r"\b(\d+)\b", r" n\1 ", t)        # bare counts ("five failed attempts" -> n5)
    return t


def _stem(t: str) -> str:
    if t.startswith(("dur_", "rate_")) or (t[0] == "n" and t[1:].isdigit()) or len(t) <= 4:
        return t
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[:-len(suf)] + rep
    return t


def toks(text: str):
    return [_stem(t) for t in _TOK.findall(canon(text))]


# function words and conversational filler: never facts, so never evidence of (un)support
STOP = set(_stem(w) for w in """the a an of to in on at is are be by for and or with that this it as you your we our i so
from will can may use within up ahead before after each every any no new has have per then than
hi hello hey there thanks thank please sorry hope helps help just also me my us am was were been being do does did
done would could should shall let know here what which who how when if but not all""".split()) - {"all", "not"}


_SENT = re.compile(r"(?<=[.?!])\s+")


def split_kb(draft: str, kb: str):
    """
    (relevant, other) article text. A help article can hold many facts the reply never mentions.
    A sentence is relevant when it shares at least two content tokens (numbers/durations count) with
    the draft, or 30% of its own tokens - so a sentence that contradicts the draft ("14 days" vs
    "30 days") shares its topic words and stays relevant. Nothing is discarded: tokens that occur only
    in the other sentences get their own namespace, and the model learns how much they matter.
    """
    sents = [x for x in _SENT.split(kb) if x.strip()]
    if len(sents) <= 1:
        return kb, ""
    dt = set(toks(draft)) - STOP
    rel, oth = [], []
    for x in sents:
        st = set(toks(x)) - STOP
        ov = len(st & dt)
        (rel if (ov >= 2 or (st and ov / len(st) >= 0.3)) else oth).append(x)
    if not rel:
        return kb, ""
    return " ".join(rel), " ".join(oth)


def _isnum(t: str) -> bool:
    return t.startswith(("dur_", "rate_")) or (t[0] == "n" and t[1:].isdigit())


def pair_tokens(draft: str, kb: str):
    """
    Feature names describing how the draft differs from the article.
      D: draft token found nowhere in the article          S: token in both
      K: relevant-article token missing from the draft    Q: token only in unrelated article sentences
    """
    rel, oth = split_kb(draft, kb)
    ds = set(toks(draft)) - STOP
    ka = set(toks(kb)) - STOP
    kr = set(toks(rel)) - STOP
    d_only, k_only, q_only = sorted(ds - ka), sorted(kr - ds), sorted((ka - kr) - ds)
    f = ["D:" + t for t in d_only] + ["K:" + t for t in k_only] + ["Q:" + t for t in q_only]
    f += ["S:" + t for t in sorted(ds & ka)]
    # numeric facts are the most decisive: durations / counts / rates
    dn = {t for t in ds if _isnum(t)}
    dn_un = dn - {t for t in ka if _isnum(t)}
    kn = {t for t in kr if _isnum(t)} - dn
    qn = {t for t in (ka - kr) if _isnum(t)} - dn
    f.append("numD_notinK=%d" % min(len(dn_un), 3))
    f.append("numK_notinD=%d" % min(len(kn), 3))
    f.append("numQ_notinD=%d" % min(len(qn), 3))
    for a in sorted(dn_un):
        for b in sorted(kn):
            if a.split("_")[0] == b.split("_")[0]:
                f.append("X:%s>%s" % (a, b))
        for b in sorted(qn):
            if a.split("_")[0] == b.split("_")[0]:
                f.append("XQ:%s>%s" % (a, b))
    # cross pairs of the (few) differing non-numeric tokens
    dw = [t for t in d_only if not _isnum(t)][:6]
    kw = [t for t in k_only if not _isnum(t)][:6]
    for a in dw:
        for b in kw:
            f.append("P:%s>%s" % (a, b))
    f.append("nD=%d" % min(len(d_only), 8))
    f.append("nK=%d" % min(len(k_only), 8))
    f.append("nQ=%d" % min(len(q_only), 8))
    return f


def dense(draft: str, kb: str):
    rel, _ = split_kb(draft, kb)
    dt, kt = toks(draft), toks(kb)
    ds, ks, kr = set(dt) - STOP, set(kt) - STOP, set(toks(rel)) - STOP
    inter = len(ds & ks)
    return [len(ds - ks) / max(1, len(ds)), len(kr - ds) / max(1, len(kr)),
            inter / max(1, len(ds | kr)), len(dt) / 50.0, len(toks(rel)) / 50.0]
