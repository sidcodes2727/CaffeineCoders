"""Offline re-implementation of the selfcheck scoring pieces (for fast iteration)."""
import numpy as np
from collections import Counter

FP = 3.0


def macro_f1(y, p, classes):
    out = []
    for c in classes:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(out))


def ece(conf, y, bins=15):
    conf = np.asarray(conf); y = np.asarray(y, dtype=float); n = len(conf); tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = ((conf > lo) & (conf <= hi)) | ((b == 0) & (conf <= 0))
        if m.any():
            tot += m.sum() / n * abs(conf[m].mean() - y[m].mean())
    return tot


def sel_risk(conf, correct, cov=0.8):
    order = np.argsort(-np.asarray(conf), kind="stable")
    k = max(1, int(round(len(order) * cov)))
    return 1 - np.mean(np.asarray(correct)[order[:k]])


def report(name, R=None, A=None, G=None, truth_r=None, truth_a=None, truth_g=None, abst=None, QUEUES=None):
    """R: (n,5) probs, A/G: p_true arrays. abst: dict of boolean arrays. Returns dict of scores."""
    out = {}
    if R is not None:
        pred = R.argmax(1); acc = (pred == truth_r).mean(); conf = R.max(1)
        ab = abst['route']
        ans = ~ab
        f1 = macro_f1(list(truth_r[ans]), list(pred[ans]), range(5))
        maj = Counter(truth_r[ans]).most_common(1)[0][0]
        triv = macro_f1(list(truth_r[ans]), [maj] * ans.sum(), range(5))
        onehot = np.eye(5)[truth_r]
        brier = ((R - onehot) ** 2).sum(1).mean()
        prior = np.bincount(truth_r, minlength=5) / len(truth_r)
        ref = ((prior[None] - onehot) ** 2).sum(1).mean()
        out['route'] = dict(acc=acc, f1=f1, triv=triv, ece=ece(conf, pred == truth_r), bskill=1 - brier / ref,
                            risk=sel_risk(conf, pred == truth_r), err=1 - acc, cov=ans.mean(),
                            q=max(0, (f1 - triv) / (1 - triv)) * min(1, ans.mean() / .8))
    for nm, P, T, key in (('approve', A, truth_a, 'approve'), ('grounded', G, truth_g, 'grounded')):
        if P is None:
            continue
        ab = abst[key]; ans = ~ab
        T = np.asarray(T).astype(bool)
        brier = ((P - T) ** 2).mean(); base = T.mean(); ref = base * (1 - base)
        conf = np.maximum(P, 1 - P); corr = (P > .5) == T
        d = dict(ece=ece(P, T), bskill=max(0, 1 - brier / ref), risk=sel_risk(conf, corr), err=1 - corr.mean(),
                 cov=ans.mean())
        if nm == 'approve':
            v = P > 0.75
            Ta, va = T[ans], v[ans]
            cost = (((va & ~Ta) * FP) + (~va & Ta)).mean()
            rate = Ta.mean(); hold = rate; appr = (1 - rate) * FP; 
            rel = (hold - cost) / hold
            d.update(cost=cost, hold=hold, q=max(0, min(1, .25 + .75 * rel)) * min(1, ans.mean() / .8))
        else:
            v = P > 0.5
            f1 = macro_f1(list(T[ans]), list(v[ans]), [True, False])
            triv = max(macro_f1(list(T[ans]), [True] * ans.sum(), [True, False]),
                       macro_f1(list(T[ans]), [False] * ans.sum(), [True, False]))
            d.update(f1=f1, triv=triv, q=max(0, (f1 - triv) / (1 - triv)) * min(1, ans.mean() / .8))
        out[nm] = d
    q = np.mean([out[k]['q'] for k in out])
    me = np.mean([out[k]['ece'] for k in out]); bs = np.mean([out[k]['bskill'] for k in out])
    cal = .5 * max(0, 1 - me / .2) + .5 * bs
    ab_ = np.mean([max(0, 1 - out[k]['risk'] / out[k]['err']) for k in out])
    out['score_no_latency'] = dict(quality=q, calibration=cal, abstention=ab_,
                                   total_with_full_latency=100 * (.4 * q + .25 * cal + .2 + .15 * ab_))
    return out
