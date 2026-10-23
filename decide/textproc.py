"""
Find the request that matters in a ticket (pure python, no deps).

Tickets often carry a main request plus decoy clauses ("Less important, but...", "On a related
note, ..."). The primary request follows a marker ("Priority for me is:"). When that marker is
missing, the decoy clauses are still recognisable, so the primary request is whatever remains
after removing them. Works on raw or textnorm-cleaned text (case-insensitive, contractions
in either form).
"""
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
    # general "this part is secondary" cues, so unseen phrasings of a side issue are still recognised
    r"(?<![a-z])(?:unrelated|side note|by the way|btw)\b[^.?!:]{0,30}?[:,]",
    r"(?<![a-z.])p\.s\.?(?:\s|:)",
    r"(?<![a-z])(?:one|another|a) (?:more|minor|small|last) (?:thing|point|question|issue)\b[^.?!:]{0,20}?[:,]",
    r"(?<![a-z])(?:minor|separate) (?:thing|point|question|issue)\b[^.?!:]{0,20}?[:,]",
    r"(?<![a-z])(?:no rush|low[- ]priority|not (?:as |that |very |so )?(?:important|urgent)|"
    r"less (?:important|urgent)|while i have you)\b[^.?!:]{0,30}?(?:[:,]|\bbut\b)",
]
_P = re.compile("|".join(PRIMARY), re.I)
_D = re.compile("|".join(DECOY), re.I)
_ANY = re.compile("|".join(PRIMARY + DECOY), re.I)
_SENT = re.compile(r"(?<=[.?!])\s+")
_END = re.compile(r"[.?!](?:\s|$)")
BOILER = re.compile(
    r"^(the amount in question|i(?:'?m| am) on the|i am using|my colleague on|i have admin|"
    r"the payment method|i raised something|i only signed|i did (?:check|dispute)|"
    r"thanks|regards|let me know|appreciate|any help|please (?:advise|reply)|"
    r"-- sent|sent from|not impressed|this is (?:getting|the)|i(?:'ve| have) been waiting|i know this|"
    r"the invoice for|the charge on|my last order|delivery of my|i have already|i was on|i can only|i only have)", re.I)


def primary_text(body: str) -> str:
    """The sentence(s) after a primary marker, or '' if there is no marker."""
    m = _P.search(body)
    if not m:
        return ""
    rest = body[m.end():]
    d = _D.search(rest)
    if d:
        rest = rest[:d.start()]
    parts = _SENT.split(rest.strip())
    keep = []
    for p in parts:
        if keep and (BOILER.match(p.strip()) or _ANY.search(p)):
            break
        keep.append(p)
        if len(keep) >= 2:
            break
    return " ".join(keep).strip()


def split_decoys(body: str):
    """(body with decoy clauses removed, decoy clause text). A decoy clause runs from its marker to
    the end of its sentence, or to the next primary marker if that comes first."""
    kept, decoys, pos = [], [], 0
    for m in _D.finditer(body):
        if m.start() < pos:
            continue
        kept.append(body[pos:m.start()])
        tail = body[m.end():]
        stop = len(tail)
        e = _END.search(tail)
        if e:
            stop = e.end()
        p = _P.search(tail)
        if p and p.start() < stop:
            stop = p.start()
        decoys.append(tail[:stop])
        pos = m.end() + stop
    kept.append(body[pos:])
    return re.sub(r"\s+", " ", " ".join(kept)).strip(), " ".join(decoys).strip()


def segments(subject: str, body: str):
    """(primary, subject, content, decoys). primary falls back to the decoy-free content when no
    marker is present; content = non-boilerplate sentences (decoys included)."""
    prim = primary_text(body)
    no_decoy, decoys = split_decoys(body)
    sents = [s for s in _SENT.split(body) if s.strip()]
    content = " ".join(s for s in sents if not BOILER.match(s.strip()))
    if not prim and decoys:
        prim = " ".join(s for s in _SENT.split(no_decoy) if s.strip() and not BOILER.match(s.strip()))
    return prim, subject, content, decoys
