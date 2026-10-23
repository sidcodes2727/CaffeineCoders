"""Text normalisation shared by training and serving (pure python, no deps)."""
import re

PRIMARY = [
    r"priority for me is\s*:", r"the main thing i need resolved is\s*:",
    r"what i actually need help (?:with|wiht) is this\s*:", r"my real question is this\s*:",
    r"most importantly\s*:", r"the part i need (?:you|yuo) to action is\s*:",
]
DECOY = [
    r"less important, but while i am writing\s*:", r"separately, and this is less urgent,",
    r"on a related note,", r"also worth mentioning,", r"some context first\s*:",
    r"for background,", r"for context",
]
_P = re.compile("|".join(PRIMARY), re.I)
_D = re.compile("|".join(DECOY), re.I)
_ANY = re.compile("|".join(PRIMARY + DECOY), re.I)
_SENT = re.compile(r"(?<=[.?!])\s+")
BOILER = re.compile(
    r"^(the amount in question|i'?m on the|i am using|my colleague on|i have admin|"
    r"the payment method|i raised something|i only signed|i did (?:check|dispute)|"
    r"thanks|regards|let me know|appreciate|any help|please (?:advise|reply)|"
    r"-- sent|sent from|not impressed|this is (?:getting|the)|i've been waiting|i know this|"
    r"the invoice for|the charge on|my last order|delivery of my|i have already|i was on|i can only|i only have)", re.I)


def primary_text(body: str) -> str:
    """The sentence(s) that carry the actual request in a multi-issue ticket."""
    m = _P.search(body)
    if not m:
        return ""
    rest = body[m.end():]
    d = _D.search(rest)
    if d:
        rest = rest[:d.start()]
    # primary request is usually one sentence; keep up to the first sentence end
    # that is followed by a boilerplate sentence or a new marker
    parts = _SENT.split(rest.strip())
    keep = []
    for p in parts:
        if keep and (BOILER.match(p.strip()) or _ANY.search(p)):
            break
        keep.append(p)
        if len(keep) >= 2:
            break
    return " ".join(keep).strip()


def segments(subject: str, body: str):
    """(primary, subject, content) where content = body with decoy clauses and boilerplate removed."""
    prim = primary_text(body)
    sents = [s for s in _SENT.split(body) if s.strip()]
    content = [s for s in sents if not BOILER.match(s.strip())]
    return prim, subject, " ".join(content)
