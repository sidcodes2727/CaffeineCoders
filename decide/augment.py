"""
Training-time augmentation (TRAINING ONLY). Each training ticket gets one perturbed copy with the
same labels, so the network stops relying on exact template wording and on the regex finding
the marker.

Deliberately disjoint from tests/perturb.py: different synonym table, different filler and
distractor sentences, and no upper-casing / emoji / unicode-quote / amount re-wording at all,
so the hard test set still measures robustness to things the model never saw.
"""
import random
import re

_PRIMARY = re.compile(r"(priority for me is|the main thing i need resolved is|what i actually need help with is this|"
                      r"my real question is this|most importantly|the part i need you to action is)\s*:", re.I)
_SENT = re.compile(r"(?<=[.?!])\s+")
_PROTECT = re.compile(r"\d|^(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fourteen|fifteen|"
                      r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|days?|weeks?|months?|"
                      r"hours?|minutes?|years?|business)$", re.I)

TRAIN_SYNONYMS = [
    ("refund", "reimbursement"), ("charged", "debited"), ("order", "purchase"), ("error", "fault"),
    ("export", "download"), ("payment", "transaction"), ("receipt", "proof of payment"), ("app", "application"),
    ("email", "e-mail"), ("tracking", "shipment status"), ("stuck", "frozen"), ("locked out", "shut out"),
    ("subscription", "membership"), ("damaged", "broken in transit"), ("reset", "renew"), ("fails", "errors out"),
]
DRAFT_SYNONYMS = [("within", "inside"), ("immediately", "right away"), ("covers", "applies to"),
                  ("available", "open"), ("approved", "accepted")]
TRAIN_FILLERS = ["Thank you in advance for looking at this.", "Have a great day.", "Apologies if this is a duplicate.",
                 "Writing from my work laptop.", "Quick one if you have a moment."]
TRAIN_DISTRACTORS = ["it would be nice to have a keyboard shortcut for search.", "do you have a status page?",
                     "the onboarding emails were helpful.", "is there an Android widget planned?"]
TRAIN_SIDE = ["Btw, {}", "Side note: {}", "Minor thing, {}", "Separately, {}", "P.S. {}", "Low priority: {}"]
DRAFT_OPENERS =["Thanks for reaching out.", "Good question.", "Happy to help with this.", "Sorry for the wait."]
DRAFT_CLOSERS = ["Let us know if anything else comes up.", "Have a good day.", "Hope this clears it up."]


def _typos(text, rng, rate=0.5):
    words = text.split(" ")
    for i, w in enumerate(words):
        core = re.sub(r"\W", "", w)
        if len(core) >= 4 and not _PROTECT.search(core) and rng.random() < rate * 0.15:
            letters = [j for j, ch in enumerate(w) if ch.isalpha()]
            j = rng.choice(letters[:-1])
            words[i] = (w[:j] + w[j + 1] + w[j] + w[j + 2:]) if rng.random() < 0.5 else (w[:j] + w[j + 1:])
    return " ".join(words)


def _syn(text, table, rng, p=0.6):
    for a, b in table:
        if rng.random() < p:
            text = re.sub(r"\b%s\b" % re.escape(a), b, text, flags=re.I)
    return text


def _sents(text):
    return [s for s in _SENT.split(text) if s.strip()]


def build_pools(rows):
    """(side issues: [(route, sentence)], kb sentences: [sentence]) from TRAINING rows only."""
    from .textproc import primary_text, segments
    hold = re.compile(r"authenticator|two-factor|2fa|delete|erase|gdpr|close (?:the )?account|partnership|"
                      r"business development|reseller|raised something|signed up|dispute|exception", re.I)
    sides, kbs = [], []
    for r in rows:
        b = r["state"].get("body") or ""
        req = primary_text(b) or segments(r["state"].get("subject") or "", b)[2]
        first = (_sents(req.strip()) or [""])[0].strip()
        if first and len(first) < 200 and not hold.search(b):
            sides.append((r["labels"]["route"], first[0].lower() + first[1:]))
        kbs.extend(_sents(r["state"].get("kb_article") or ""))
    return sides, kbs


def augment_state(state, rng: random.Random, route=None, pools=None):
    """route: this ticket's queue label (side issues are drawn from other queues); pools: build_pools()."""
    s = dict(state)
    body = s.get("body") or ""
    ops = ["typos", "drop_marker", "shuffle", "filler", "distractor", "synonyms", "lower"]
    if pools:
        ops += ["side_issues", "side_issues", "long_kb"]
    ops = rng.sample(ops, rng.randint(1, 3))
    if "side_issues" in ops:
        cands = [t for q, t in pools[0] if q != route]
        ss = _sents(body)
        for t in rng.sample(cands, min(len(cands), rng.randint(1, 2))):
            ss.insert(rng.randint(0, len(ss)), rng.choice(TRAIN_SIDE).format(t))
        body = " ".join(ss)
    if "long_kb" in ops and s.get("kb_article"):
        kw = set(re.findall(r"[a-z]+", s["kb_article"].lower()))
        for _ in range(20):
            cand = rng.choice(pools[1])
            if len(kw & set(re.findall(r"[a-z]+", cand.lower()))) <= 2:
                s["kb_article"] = s["kb_article"] + " " + cand
                break
    if "drop_marker" in ops:
        body = re.sub(r"\s+", " ", _PRIMARY.sub("", body)).strip()
    if "shuffle" in ops:
        ss = _sents(body); rng.shuffle(ss); body = " ".join(ss)
    if "filler" in ops:
        ss = _sents(body); ss.insert(rng.randint(0, len(ss)), rng.choice(TRAIN_FILLERS)); body = " ".join(ss)
    if "distractor" in ops:
        body = "Also worth mentioning, %s %s" % (rng.choice(TRAIN_DISTRACTORS), body)
    if "synonyms" in ops:
        body = _syn(body, TRAIN_SYNONYMS, rng)
        s["subject"] = _syn(s.get("subject") or "", TRAIN_SYNONYMS, rng)
    if "typos" in ops:
        body = _typos(body, rng)
        s["subject"] = _typos(s.get("subject") or "", rng)
    if "lower" in ops:
        body = body.lower()
    s["body"] = body

    d = s.get("draft_reply") or ""
    if d:
        dops = rng.sample(["typos", "open", "close", "synonyms", "lower", "none"], rng.randint(1, 2))
        if "synonyms" in dops:
            d = _syn(d, DRAFT_SYNONYMS, rng)
        if "typos" in dops:
            d = _typos(d, rng)
        if "open" in dops:
            d = rng.choice(DRAFT_OPENERS) + " " + d
        if "close" in dops:
            d = d + " " + rng.choice(DRAFT_CLOSERS)
        if "lower" in dops:
            d = d.lower()
        s["draft_reply"] = d
    return s
