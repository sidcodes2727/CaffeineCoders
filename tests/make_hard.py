"""
Build a HARD, hidden-style evaluation set from a labelled split.

Every record keeps its original labels and gets 2-3 label-preserving perturbations combined
(typos, marker removed, shuffled sentences, distractor/filler clauses, unseen synonyms, case,
amount re-wording, emoji/unicode/whitespace noise). Score it with the official harness:

    python tests/make_hard.py --src data/dev.jsonl --out data/hard_dev.jsonl
    python selfcheck.py --url http://localhost:8000/v1/decide --eval data/hard_dev.jsonl --json hard_results.json
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturb import hard_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/dev.jsonl")
    ap.add_argument("--out", default="data/hard_dev.jsonl")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.src) if l.strip()]
    used = collections.Counter()
    with open(args.out, "w") as fh:
        for r in rows:
            s, names = hard_state(r["state"], rng)
            used.update(names)
            fh.write(json.dumps({"request_id": r["request_id"] + "-hard", "state": s, "labels": r["labels"],
                                 "perturbations": names}) + "\n")
    print("wrote %d rows to %s" % (len(rows), args.out))
    print("perturbation counts:", dict(used))


if __name__ == "__main__":
    main()
