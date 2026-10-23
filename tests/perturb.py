"""
Label-preserving perturbations used ONLY for testing (invariance tests and the hard test set).

Kept deliberately separate from decide/augment.py (training augmentation): the synonym
table here is disjoint from the training one, and several perturbation types (upper case,
emoji, unicode quotes, amount re-wording) never appear in training, so the hard set measures
robustness to things the model was not shown.
"""
import random
import re

PRIMARY_MARKERS = [
    "Priority for me is:", "The main thing I need resolved is:", "What I actually need help with is this:",
    "My real question is this:", "Most importantly:", "The part I need you to action is:",
]
_PRIMARY_RX = re.compile("|".join(re.escape(m) for m in PRIMARY_MARKERS), re.I)
_SENT = re.compile(r"(?<=[.?!])\s+")
_NUMWORDS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand
day days week weeks month months hour hours minute minutes year years business""".split())

# test-only synonyms (training augmentation uses a different table)
TEST_SYNONYMS = [
    ("refund", "money back"), ("log in", "sign in"), ("login", "sign-in"), ("charged twice", "billed two times"),
    ("invoice", "bill"), ("package", "parcel"), ("cancel", "terminate"), ("password", "passphrase"),
    ("crashes", "keeps closing"), ("not working", "broken"), ("arrived", "turned up"), ("delivered", "dropped off"),
]
FILLERS = ["Hope you are having a good week.", "Sorry for the long message.", "Thanks for your patience with me.",
           "I appreciate any pointers you can share.", "Not sure if this is the right place to ask."]
DISTRACTORS = ["would be great if the dashboard had a dark mode.", "does your mobile app support widgets yet?",
               "your new logo looks nice by the way.", "is there a public roadmap I can follow?"]
EMOJIS = ["🙏", "😡", "😕", "👍", "🔥"]


def _sentences(text):
    return [s for s in _SENT.split(text) if s.strip()]


def typo(text, rng, rate=1.0, protect_numbers=True):
    """Swap or drop one character in 1-2 words per sentence."""
    out = []
    for s in _sentences(text) or [text]:
        words = s.split(" ")
        cand = [i for i, w in enumerate(words) if len(re.sub(r"\W", "", w)) >= 4 and
                not (protect_numbers and (re.sub(r"\W", "", w).lower() in _NUMWORDS or any(ch.isdigit() for ch in w)))]
        rng.shuffle(cand)
        for i in cand[:rng.randint(1, 2)] if rng.random() < rate else []:
            w = words[i]
            letters = [j for j, ch in enumerate(w) if ch.isalpha()]
            if len(letters) < 3:
                continue
            j = rng.choice(letters[:-1])
            if rng.random() < 0.5 and j + 1 < len(w):
                w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            else:
                w = w[:j] + w[j + 1:]
            words[i] = w
        out.append(" ".join(words))
    return " ".join(out)


def remove_marker(text, rng=None):
    return re.sub(r"\s+", " ", _PRIMARY_RX.sub("", text)).strip()


def shuffle_sentences(text, rng):
    s = _sentences(text)
    rng.shuffle(s)
    return " ".join(s)


def add_filler(text, rng):
    s = _sentences(text)
    s.insert(rng.randint(0, len(s)), rng.choice(FILLERS))
    return " ".join(s)


def add_distractor(text, rng):
    return "Less important, but while I am writing: %s %s" % (rng.choice(DISTRACTORS), text)


def synonyms(text, rng=None):
    for a, b in TEST_SYNONYMS:
        text = re.sub(r"\b%s\b" % re.escape(a), b, text, flags=re.I)
    return text


def _fmt_amount(m, rng):
    v = float(m.group(1).replace(",", ""))
    style = rng.randint(0, 3)
    if style == 0:
        return "${:,.2f}".format(v)
    if style == 1:
        return "USD {:.2f}".format(v)
    if style == 2:
        return "{:.2f} dollars".format(v)
    return "${:.1f}k".format(v / 1000) if v >= 1000 else "$ {:.2f}".format(v)


def reformat_amount(text, rng):
    return re.sub(r"\$([\d,]+(?:\.\d+)?)", lambda m: _fmt_amount(m, rng), text)


def noise_chars(text, rng):
    text = text.replace("'", "’").replace('"', "“")
    text = re.sub(r" ", lambda m: "  " if rng.random() < 0.1 else " ", text)
    return text + " " + rng.choice(EMOJIS) + "\n"


def lower(text, rng=None):
    return text.lower()


def upper(text, rng=None):
    return text.upper()


# name -> (function, fields it applies to per decision)
#   "req" = subject + body (route / approve), "draft" = draft_reply, "all" = every text field
PERTURBATIONS = {
    "typos": typo,
    "lowercase": lower,
    "uppercase": upper,
    "marker_removed": remove_marker,
    "sentences_shuffled": shuffle_sentences,
    "filler_added": add_filler,
    "distractor_added": add_distractor,
    "synonyms": synonyms,
    "amount_reformatted": reformat_amount,
    "whitespace_emoji_quotes": noise_chars,
}


def apply(state, name, rng, decision=None):
    """Return a perturbed copy of state. Grounded perturbations touch the draft (and case/noise the KB too)."""
    s = dict(state)
    fn = PERTURBATIONS[name]
    req_fields = ["subject", "body"] if name not in ("marker_removed", "sentences_shuffled", "filler_added",
                                                     "distractor_added", "amount_reformatted") else ["body"]
    if decision == "grounded":
        if name in ("typos", "synonyms", "filler_added", "lowercase", "uppercase", "whitespace_emoji_quotes"):
            fields = ["draft_reply"] + (["kb_article"] if name in ("lowercase", "uppercase", "whitespace_emoji_quotes") else [])
        else:
            fields = []
    else:
        fields = req_fields
    for f in fields:
        if s.get(f):
            s[f] = fn(s[f], rng)
    return s


def hard_state(state, rng, k=(2, 3)):
    """Combine several perturbations on every text field (used for the hard hidden-style set)."""
    names = rng.sample([n for n in PERTURBATIONS if n != "uppercase"], rng.randint(*k))
    if rng.random() < 0.15:
        names.append("uppercase")
    s = dict(state)
    for n in names:
        fn = PERTURBATIONS[n]
        if n in ("marker_removed", "sentences_shuffled", "distractor_added", "amount_reformatted"):
            s["body"] = fn(s["body"], rng)
        elif n == "filler_added":
            s["body"] = fn(s["body"], rng)
            if s.get("draft_reply"):
                s["draft_reply"] = fn(s["draft_reply"], rng)
        else:
            for f in ("subject", "body", "draft_reply") + (("kb_article",) if n in ("lowercase", "uppercase", "whitespace_emoji_quotes") else ()):
                if s.get(f):
                    s[f] = fn(s[f], rng)
    return s, names
