"""
Fit the grounded cascade (network + fine-tuned NLI cross-encoder) and compare it with the network alone.

    python tools/fit_cascade.py --nn-holdout <dir of `train.py --holdout 0.15` artifacts> \
                                --nli artifacts/nli --ship artifacts

1. Fitting data = the 15% holdout of train.jsonl, which neither the holdout network nor the
   cross-encoder (tools/train_nli.py --holdout 0.15) was trained on, in four forms: clean, augmented,
   paraphrased with the TRAINING fact-synonym table, and with unrelated KB sentences appended.
2. Trigger rule (min lexical gap, network-confidence floor) chosen by cross-fitted log-loss.
3. Combiner = logistic regression on decide/cascade.design(); abstain threshold from the same rows.
4. Comparison on dev-derived sets with the SHIP network: clean dev, test-synonym drafts, hard dev,
   long_kb, everything. Writes <nli>/cascade.json.
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from decide.augment import TRAIN_FACT_SYNONYMS, augment_state, build_pools  # noqa: E402
from decide.cascade import design, gap_features  # noqa: E402
from decide.nli import CrossEncoder  # noqa: E402
from evalkit import ece, macro_f1  # noqa: E402

GROUNDED_COVERAGE = 0.88


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def split_holdout(rows, frac=0.15, seed=1234):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    hold = set(idx[:int(len(rows) * frac)])
    return [r for i, r in enumerate(rows) if i not in hold], [r for i, r in enumerate(rows) if i in hold]


def fact_syn_all(text):
    for a, b in TRAIN_FACT_SYNONYMS:
        text = re.sub(r"\b%s\b" % re.escape(a), b, text, flags=re.I)
    return text


def nn_probs(model, states):
    return np.array([model.grounded(s)[1] for s in states])


def score_rows(states, y, p_nn, ce_fn, cas):
    """Final probabilities under the cascade spec `cas` (None = network only)."""
    out, used = [], 0
    for s, p in zip(states, p_nn):
        if cas is None or not s.get("draft_reply") or not s.get("kb_article"):
            out.append(p)
            continue
        g = gap_features(s["draft_reply"], s["kb_article"])
        if g["gap"] >= cas["min_gap"] or max(p, 1 - p) < cas["unsure"]:
            used += 1
            z = float(np.dot(cas["coef"], design(p, ce_fn(s), g)) + cas["bias"])
            out.append(1 / (1 + np.exp(-z)))
        else:
            out.append(p)
    return np.array(out), used / max(1, len(states))


def grounded_report(p, y, thr):
    y = np.asarray(y, dtype=bool)
    ab = np.maximum(p, 1 - p) < thr
    ans = ~ab
    v = p > 0.5
    f1 = macro_f1(list(y[ans]), list(v[ans]), [True, False])
    triv = max(macro_f1(list(y[ans]), [True] * ans.sum(), [True, False]),
               macro_f1(list(y[ans]), [False] * ans.sum(), [True, False]))
    base = y.mean()
    return {"acc": float((v == y).mean()), "f1_answered": f1, "ece": ece(p, y.astype(float)),
            "brier_skill": float(1 - ((p - y) ** 2).mean() / (base * (1 - base))), "coverage": float(ans.mean()),
            "q": max(0, (f1 - triv) / (1 - triv)) * min(1, ans.mean() / 0.8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-holdout", required=True)
    ap.add_argument("--nli", default="artifacts/nli")
    ap.add_argument("--ship", default="artifacts")
    args = ap.parse_args()
    t0 = time.time()
    from model import DecisionModel
    from perturb import synonyms as test_synonyms
    enc = CrossEncoder(args.nli)
    cache = {}

    def ce(s):
        k = (s["kb_article"], s["draft_reply"])
        if k not in cache:
            cache[k] = enc.logit(*k)
        return cache[k]

    # ---- fitting rows ---------------------------------------------------------------------------
    tr = load(os.path.join(ROOT, "data", "train.jsonl"))
    rest, hold = split_holdout(tr)
    pools = build_pools(rest)
    rng = random.Random(11)
    fit_states, fit_y = [], []
    for r in hold:
        s, y = r["state"], r["labels"]["grounded"]
        variants = [s, augment_state(s, rng, r["labels"]["route"], pools),
                    dict(s, draft_reply=fact_syn_all(s["draft_reply"]))]
        a = augment_state(s, rng, r["labels"]["route"], (pools[0], pools[1]))
        variants.append(dict(s, kb_article=a["kb_article"]))
        for v in variants:
            fit_states.append(v); fit_y.append(y)
    fit_y = np.array(fit_y, dtype=int)
    m_hold = DecisionModel(args.nn_holdout)
    p_fit = nn_probs(m_hold, fit_states)
    G = [gap_features(s["draft_reply"], s["kb_article"]) for s in fit_states]
    CE = np.array([ce(s) for s in fit_states])
    print("fitting rows %d (holdout x4) | network acc %.4f | cross-encoder acc %.4f | %.0fs" % (
        len(fit_y), ((p_fit > .5) == fit_y).mean(), ((CE > 0) == fit_y).mean(), time.time() - t0), flush=True)

    X = np.array([design(p, c, g) for p, c, g in zip(p_fit, CE, G)])
    gap = np.array([g["gap"] for g in G])
    conf = np.maximum(p_fit, 1 - p_fit)
    best = None
    for min_gap in (1, 2, 3):
        for unsure in (0.9, 0.97, 0.995):
            trig = (gap >= min_gap) | (conf < unsure)
            final = p_fit.copy()
            for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, fit_y):
                ta, tb = a[trig[a]], b[trig[b]]
                if len(set(fit_y[ta])) < 2:
                    continue
                lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[ta], fit_y[ta])
                final[tb] = lr.predict_proba(X[tb])[:, 1]
            ll = -np.mean(fit_y * np.log(np.clip(final, 1e-6, 1)) + (1 - fit_y) * np.log(np.clip(1 - final, 1e-6, 1)))
            acc = ((final > .5) == fit_y).mean()
            print("  trigger gap>=%d or conf<%.3f: fires %.1f%% | cross-fitted logloss %.4f acc %.4f" % (
                min_gap, unsure, 100 * trig.mean(), ll, acc))
            if best is None or ll < best[0]:
                best = (ll, min_gap, unsure, trig)
    ll0 = -np.mean(fit_y * np.log(np.clip(p_fit, 1e-6, 1)) + (1 - fit_y) * np.log(np.clip(1 - p_fit, 1e-6, 1)))
    print("  network alone: logloss %.4f acc %.4f" % (ll0, ((p_fit > .5) == fit_y).mean()))
    _, min_gap, unsure, trig = best
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[trig], fit_y[trig])
    cas = {"coef": lr.coef_[0].tolist(), "bias": float(lr.intercept_[0]), "min_gap": min_gap, "unsure": unsure}
    final, _ = score_rows(fit_states, fit_y, p_fit, ce, cas)
    cas["abstain"] = float(np.quantile(np.maximum(final, 1 - final), 1 - GROUNDED_COVERAGE))
    cas["fit"] = {"rows": int(len(fit_y)), "trigger_rate": float(trig.mean()), "logloss": float(best[0]),
                  "logloss_network_only": float(ll0)}
    json.dump(cas, open(os.path.join(args.nli, "cascade.json"), "w"), indent=1)
    print("chosen: gap>=%d or conf<%.3f, coef %s, abstain %.4f" % (min_gap, unsure, np.round(cas["coef"], 3), cas["abstain"]))

    # ---- comparison with the ship network ---------------------------------------------------------
    m = DecisionModel(args.ship)
    thr_nn = m.thr["grounded"]
    dv = load(os.path.join(ROOT, "data", "dev.jsonl"))
    sets = {"dev": dv,
            "dev_test_synonyms": [dict(r, state=dict(r["state"], draft_reply=test_synonyms(r["state"]["draft_reply"]))) for r in dv]}
    for name, path in (("hard_dev", "data/hard_dev.jsonl"), ("long_kb", "data/complex/long_kb.jsonl"),
                       ("everything", "data/complex/everything.jsonl")):
        if os.path.exists(os.path.join(ROOT, path)):
            sets[name] = load(os.path.join(ROOT, path))
    print("\n%-18s %-9s %7s %7s %7s %7s %6s %7s %7s" % ("set", "model", "acc", "F1ans", "ECE", "Bskill", "cov", "q", "CE%"))
    res = {}
    for name, rows in sets.items():
        st = [r["state"] for r in rows]
        y = np.array([r["labels"]["grounded"] for r in rows])
        p_nn = nn_probs(m, st)
        p_c, rate = score_rows(st, y, p_nn, ce, cas)
        for label, p, thr, r_ in (("network", p_nn, thr_nn, 0.0), ("cascade", p_c, cas["abstain"], rate)):
            rep = grounded_report(p, y, thr)
            res["%s/%s" % (name, label)] = dict(rep, ce_rate=r_)
            print("%-18s %-9s %7.4f %7.4f %7.4f %7.4f %6.3f %7.4f %6.1f%%" % (
                name, label, rep["acc"], rep["f1_answered"], rep["ece"], rep["brier_skill"], rep["coverage"], rep["q"], 100 * r_))
    json.dump(res, open(os.path.join(args.nli, "cascade_comparison.json"), "w"), indent=1)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
