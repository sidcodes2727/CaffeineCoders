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


def toks(text: str):
    return _TOK.findall(canon(text))


STOP = set("""the a an of to in on at is are be by for and or with that this it as you your we our i so
from will can may use within up ahead before after each every any no new has have per then than""".split())


def pair_tokens(draft: str, kb: str):
    """Hashable feature names describing how the draft differs from the article."""
    dt, kt = toks(draft), toks(kb)
    ds, ks = set(dt) - STOP, set(kt) - STOP
    f = []
    d_only, k_only = sorted(ds - ks), sorted(ks - ds)
    f += ["D:" + t for t in d_only]
    f += ["K:" + t for t in k_only]
    f += ["S:" + t for t in sorted(ds & ks)]
    # numeric facts are the most decisive: durations / counts / rates
    dn = {t for t in ds if t.startswith(("dur_", "rate_")) or ((t[0] == "n" and t[1:].isdigit()))}
    kn = {t for t in ks if t.startswith(("dur_", "rate_")) or ((t[0] == "n" and t[1:].isdigit()))}
    f.append("numD_notinK=%d" % min(len(dn - kn), 3))
    f.append("numK_notinD=%d" % min(len(kn - dn), 3))
    for a in sorted(dn - kn):
        for b in sorted(kn - dn):
            if a.split("_")[0] == b.split("_")[0]:
                f.append("X:%s>%s" % (a, b))
    # cross pairs of the (few) differing non-numeric tokens
    dw = [t for t in d_only if not (t.startswith(("dur_", "rate_")) or (t[0] == "n" and t[1:].isdigit()))][:6]
    kw = [t for t in k_only if not (t.startswith(("dur_", "rate_")) or (t[0] == "n" and t[1:].isdigit()))][:6]
    for a in dw:
        for b in kw:
            f.append("P:%s>%s" % (a, b))
    f.append("nD=%d" % min(len(d_only), 8))
    f.append("nK=%d" % min(len(k_only), 8))
    return f


def dense(draft: str, kb: str):
    dt, kt = toks(draft), toks(kb)
    ds, ks = set(dt) - STOP, set(kt) - STOP
    inter = len(ds & ks)
    return [len(ds - ks) / max(1, len(ds)), len(ks - ds) / max(1, len(ks)),
            inter / max(1, len(ds | ks)), len(dt) / 50.0, len(kt) / 50.0]
