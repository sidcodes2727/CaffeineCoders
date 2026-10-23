"""
Complex-query stress sets, built from a labelled split with the labels kept.

Each variant makes the tickets harder to read without changing what a careful human would answer:

  multi_issue        two extra issues from OTHER tickets (other queues), introduced with the usual
                     "less urgent" phrasing, at random positions
  multi_issue_new    same, but with side-issue phrasings never seen in training ("While I have you,")
  multi_issue_fresh  same, with phrasings that share no cue word with the model's side-issue rules or
                     its training templates ("If you get a chance,") - the clean held-out version
  buried_request     side issues and chatter first, the real request (with its marker) last
  long_ticket        6-10 extra neutral sentences (device info, greetings, apologies)
  long_kb            1-2 sentences from an unrelated help article appended to the KB article
  everything         multi_issue_new + long_ticket + long_kb + 2-3 surface perturbations (typos, case,
                     shuffles, emoji, synonyms ...)

Side issues are taken only from tickets whose request is not an always-hold intent and carries no
history cue, so the approve label cannot be changed by them. Appended KB sentences come from an
article on a different topic, so they cannot support a claim the draft makes.

    python tests/make_complex.py --src data/dev.jsonl --outdir data/complex
    python selfcheck.py --url ... --eval data/complex/everything.jsonl --json /tmp/x.json
"""
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from perturb import hard_state  # noqa: E402
from decide.textproc import primary_text, segments  # noqa: E402

KNOWN_SIDE = ["Separately, and this is less urgent, {}", "Less important, but while I am writing: {}",
              "On a related note, {}", "Also worth mentioning, {}"]
NEW_SIDE = ["While I have you, {}", "Unrelated and low priority: {}", "One minor thing, no rush: {}",
            "Not important right now, but {}"]
# written BEFORE the side-issue keyword rules in decide/textproc.py and the training templates in
# decide/augment.py, and deliberately sharing no cue words with either: a clean held-out phrasing test
FRESH_SIDE = ["If you get a chance, {}", "For what it is worth, {}", "Small extra thing, {}",
              "This is not why I am writing, but {}"]
NEUTRAL = ["I am using Chrome on Windows 11 if that matters.", "Thanks in advance.", "Regards.",
           "Please advise.", "Sent from my phone.", "Hope you are well.", "Sorry if this is the wrong inbox.",
           "I tried again this morning.", "My manager asked me to follow up.", "We are based in Lisbon.",
           "Our team is about twenty people.", "I have attached nothing, just describing it here.",
           "Apologies for the long email.", "Happy to jump on a call.", "Let me know what you need from me."]
HOLD = re.compile(r"authenticator|two-factor|2fa|delete|erase|gdpr|close (?:the )?account|partnership|"
                  r"business development|reseller|raised something|signed up|dispute|exception", re.I)
_SENT = re.compile(r"(?<=[.?!])\s+")
_WORDS = re.compile(r"[a-z]+")


def side_pool(rows):
    pool = []
    for r in rows:
        b = r["state"]["body"]
        req = primary_text(b) or segments(r["state"]["subject"], b)[2]
        req = (sents(req.strip()) or [""])[0].strip()          # one sentence: a self-contained side issue
        req = re.sub(r"^(hi|hello|hey)( there| team)?,?\s*", "", req, flags=re.I)
        if req and len(req) < 200 and not HOLD.search(b):
            pool.append((r["labels"]["route"], req[0].lower() + req[1:].rstrip(".?!") + "."))
    return pool


def pick_sides(pool, route, rng, k=2):
    cands = [t for q, t in pool if q != route]
    return rng.sample(cands, k)


def sents(text):
    return [s for s in _SENT.split(text) if s.strip()]


def add_sides(body, sides, templates, rng):
    ss = sents(body)
    for t in sides:
        ss.insert(rng.randint(0, len(ss)), rng.choice(templates).format(t))
    return " ".join(ss)


def bury(body, sides, rng):
    p = primary_text(body)
    m = re.search(r"(priority for me is|the main thing i need resolved is|what i actually need help with is this|"
                  r"my real question is this|most importantly|the part i need you to action is)\s*:", body, re.I)
    if not p or not m:
        req, rest = body, ""
    else:
        req = m.group(0) + " " + p
        rest = (body[:m.start()] + " " + body[m.end():].replace(p, "", 1)).strip()
    chatter = rng.sample(NEUTRAL, 3)
    parts = [rng.choice(KNOWN_SIDE).format(s) for s in sides] + chatter + ([rest] if rest else [])
    return " ".join(parts + [req if m else "The main thing I need resolved is: " + req])


def far_kb_sentences(rows, kb, rng, k):
    kw = set(_WORDS.findall(kb.lower()))
    for _ in range(50):
        other = rng.choice(rows)["state"]["kb_article"]
        ow = set(_WORDS.findall(other.lower()))
        if len(kw & ow) / max(1, len(kw | ow)) < 0.12:
            ss = sents(other)
            return " ".join(rng.sample(ss, min(k, len(ss))))
    return ""


def variant(name, r, rows, pool, rng):
    s = dict(r["state"])
    route = r["labels"]["route"]
    if name in ("multi_issue", "multi_issue_new", "multi_issue_fresh", "everything"):
        tpl = {"multi_issue": KNOWN_SIDE, "multi_issue_fresh": FRESH_SIDE}.get(name, NEW_SIDE)
        s["body"] = add_sides(s["body"], pick_sides(pool, route, rng), tpl, rng)
    if name == "buried_request":
        s["body"] = bury(s["body"], pick_sides(pool, route, rng), rng)
    if name in ("long_ticket", "everything"):
        ss = sents(s["body"])
        for t in rng.sample(NEUTRAL, rng.randint(6, 10)):
            ss.insert(rng.randint(0, len(ss)), t)
        s["body"] = " ".join(ss)
    if name in ("long_kb", "everything"):
        extra = far_kb_sentences(rows, s["kb_article"], rng, rng.randint(1, 2))
        if extra:
            s["kb_article"] = s["kb_article"] + " " + extra
    if name == "everything":
        s, _ = hard_state(s, rng)
    return s


VARIANTS = ["multi_issue", "multi_issue_new", "multi_issue_fresh", "buried_request", "long_ticket", "long_kb",
            "everything"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/dev.jsonl")
    ap.add_argument("--outdir", default="data/complex")
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.src) if l.strip()]
    pool = side_pool(rows)
    os.makedirs(args.outdir, exist_ok=True)
    for name in VARIANTS:
        rng = random.Random("%s-%d" % (name, args.seed))
        path = os.path.join(args.outdir, name + ".jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps({"request_id": r["request_id"] + "-" + name,
                                     "state": variant(name, r, rows, pool, rng), "labels": r["labels"]}) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
