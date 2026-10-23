"""
Train the Decide model.

    python train.py --repeats 2                      # shipped config: train.jsonl only, dev held out
    python train.py --repeats 3 --holdout 0.15       # evaluation protocol: also hold out 15% of train,
                                                     # report repeated-CV mean +/- std and the holdout once

Writes everything the service needs into --out (default artifacts/).
"""
import argparse, json, os, random, shutil, sys, time
import numpy as np
import torch
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from decide import features, textnorm
from decide.augment import augment_state, build_pools
from decide.encode import build_vocab, encode_feats
from decide.features import H_BUCKETS, N_NUM, featurize, tier_amount
from decide.gate import Gate, export_booster
from decide.net import np_forward, np_forward_typed
from decide.torchnet import export
from train_lib import QUEUES, load, predict, train_net
from evalkit import report

torch.set_num_threads(4)
sig = lambda z: 1 / (1 + np.exp(-z))
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
softmax = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
COVERAGE = 0.835          # declared coverage; target is 0.80 so keep margin for test-set noise
GROUNDED_COVERAGE = 0.88  # grounded is ~99.5% accurate on clean text; answer more so that coverage
                          # stays above 80% when wording drifts (perturbed sets)
APPROVE_TH = 0.75         # cost 3:1 -> approve iff p > 3/(3+1)
CFG = dict(epochs=15, dim=128, p=0.5, wd=0.1, wide_wd=0.02, ls=0.1, tok_drop=0.2, modality_drop=0.25)
# rows hidden by modality dropout: request wording only. History sentences (s:) stay visible - hiding
# them would teach approve to deny without the cue that caused the denial.
TEXT_NS = ("p:", "b:", "d:", "pflag:")


def gbm_feats(states, te):
    ta = [tier_amount(s) for s in states]
    return np.array([[t, a, e] for (t, a), e in zip(ta, te)], dtype=np.float64)


def fit_gbm(X, y, seed=0):
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=1.0, reg_lambda=5.0,
                              monotone_constraints=[0, -1, 1], random_state=seed, verbose=-1).fit(X, y)


class Calib:
    """Everything that turns raw head outputs into served probabilities and abstain flags."""

    def __init__(self, T, g_ab, gbm, approve_cal, thr):
        self.T, self.g_ab, self.gbm, self.ac, self.thr = T, g_ab, gbm, approve_cal, thr

    def approve_p(self, gate_p):
        ac = self.ac
        pi = np.interp(gate_p, ac["iso_x"], ac["iso_y"])
        pp = sig(ac["platt"][0] * logit(gate_p) + ac["platt"][1])
        return {"iso": pi, "platt": pp, "blend": 0.5 * (pi + pp)}[ac["kind"]]

    def approve_rank(self, gate_p):
        """Tie-free version of p (Platt on the gate) used only to RANK cases for the abstain flag:
        isotonic steps put hundreds of cases on one value, which a threshold cannot split."""
        return approve_cost(sig(self.ac["platt"][0] * logit(gate_p) + self.ac["platt"][1]))

    def probs(self, R, A, G, states):
        Pr = softmax(R / self.T)
        Pg = sig(self.g_ab[0] * G + self.g_ab[1])
        Ga = self.gbm.predict_proba(gbm_feats(states, A))[:, 1]
        return Pr, self.approve_p(Ga), Pg, Ga

    def abstain(self, Pr, Ga, Pg):
        return {"route": Pr.max(1) < self.thr["route"],
                "grounded": np.maximum(Pg, 1 - Pg) < self.thr["grounded"],
                "approve": self.approve_rank(Ga) > self.thr["approve"]}


def approve_cost(p):
    """Expected cost of the action taken at the 0.75 threshold (hold costs p, approve costs 3(1-p))."""
    return np.minimum(3 * (1 - p), p)


def labels(rows):
    return (np.array([QUEUES.index(r["labels"]["route"]) for r in rows]),
            np.array([r["labels"]["approve"] for r in rows]).astype(bool),
            np.array([r["labels"]["grounded"] for r in rows]).astype(bool))


def score(cal, R, A, G, rows):
    Pr, Pa, Pg, Ga = cal.probs(R, A, G, [r["state"] for r in rows])
    yr, ya, yg = labels(rows)
    return report("x", Pr, Pa, Pg, yr, ya, yg, cal.abstain(Pr, Ga, Pg))


def ens_logits(nets, arch, items):
    Rl, Al, Gl = [], [], []
    for ids, w, x in items:
        if arch == "typed":
            Rl.append(np.mean([np_forward_typed(sd, ids, w, x, 0) for sd in nets], 0))
            Al.append(np.mean([np_forward_typed(sd, ids, w, x, 1) for sd in nets]))
            Gl.append(np.mean([np_forward_typed(sd, ids, w, x, 2) for sd in nets]))
        else:
            rr = [np_forward(sd, ids, w, x) for sd in nets]
            Rl.append(np.mean([q[0] for q in rr], 0)); Al.append(np.mean([q[1] for q in rr]))
            Gl.append(np.mean([q[2] for q in rr]))
    return np.array(Rl), np.array(Al), np.array(Gl)


def show(name, rep):
    s = rep["score_no_latency"]
    print("%-14s total %.2f | q %.4f cal %.4f abst %.4f | route acc %.4f f1 %.4f ece %.4f | approve cost %.4f "
          "ece %.4f | grounded f1 %.4f ece %.4f | cov %.3f/%.3f/%.3f" % (
              name, s["total_with_full_latency"], s["quality"], s["calibration"], s["abstention"],
              rep["route"]["acc"], rep["route"]["f1"], rep["route"]["ece"], rep["approve"]["cost"],
              rep["approve"]["ece"], rep["grounded"]["f1"], rep["grounded"]["ece"],
              rep["route"]["cov"], rep["approve"]["cov"], rep["grounded"]["cov"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2, help="repeated k-fold; ensemble = repeats x folds nets")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--arch", default="typed", choices=["typed", "shared"])
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--hard", default="data/hard_dev.jsonl", help="perturbed dev (tests/make_hard.py)")
    ap.add_argument("--holdout", type=float, default=0.0, help="fraction of train never used for anything")
    ap.add_argument("--aug", type=int, default=1, help="augmented copies per training row (0 = off)")
    ap.add_argument("--no-m2v", action="store_true", help="ablation: drop the pretrained embedding input")
    ap.add_argument("--no-spell", action="store_true", help="ablation: no spell correction")
    ap.add_argument("--epochs", type=int, default=None, help="override CFG epochs (smoke tests)")
    args = ap.parse_args()
    if args.epochs:
        CFG["epochs"] = args.epochs
    t0 = time.time()
    src_art = "artifacts"
    os.makedirs(args.out, exist_ok=True)
    for f in ("m2v_emb.npy", "m2v_vocab.txt"):
        if os.path.abspath(os.path.join(src_art, f)) != os.path.abspath(os.path.join(args.out, f)):
            shutil.copy(os.path.join(src_art, f), os.path.join(args.out, f))

    tr, dv = load(args.train), load(args.dev)
    hold = []
    if args.holdout > 0:
        rng = random.Random(1234)
        idx = list(range(len(tr))); rng.shuffle(idx)
        k = int(len(tr) * args.holdout)
        hold = [tr[i] for i in sorted(idx[:k])]
        tr = [tr[i] for i in sorted(idx[k:])]
        print("holdout: %d rows set aside, %d used for training" % (len(hold), len(tr)))
    fit_rows = tr
    states = [r["state"] for r in fit_rows]

    # pretrained embedding + spell vocabulary (built from the training text only)
    features.init(args.out, speller=False)
    if args.no_m2v:
        features.M2V = None
    if not args.no_spell:
        protected = {w for w in features.M2V.vocab if w.isalpha()} if features.M2V else None
        sp = textnorm.Speller.build((s.get(f) for s in states for f in ("subject", "body", "draft_reply", "kb_article")),
                                    protected)
        sp.save(os.path.join(args.out, "spell_vocab.json"))
        textnorm.set_speller(sp)
        print("spell vocab %d words" % len(sp.freq))
    elif os.path.exists(os.path.join(args.out, "spell_vocab.json")):
        os.remove(os.path.join(args.out, "spell_vocab.json"))

    arng = random.Random(99)
    pools = build_pools(fit_rows)                 # training rows only (holdout / dev never contribute)
    aug_states = [augment_state(r["state"], arng, r["labels"]["route"], pools) for r in fit_rows] if args.aug else []
    feats = [featurize(s) for s in states]
    aug_feats = [featurize(s) for s in aug_states]
    print("featurised %d (+%d augmented) in %.0fs" % (len(feats), len(aug_feats), time.time() - t0), flush=True)

    from collections import Counter
    df = Counter()
    for f in feats + aug_feats:
        df.update(set(f[0]))
    vocab = {t: i for i, t in enumerate(sorted(t for t, c in df.items() if c >= 2))}
    inv = sorted(vocab, key=vocab.get)
    n_rows = len(vocab) + H_BUCKETS
    wide_mask = np.array([1.0 if t.startswith(("p:", "s:")) else 0.0 for t in inv] + [0.0] * H_BUCKETS, dtype=np.float32)
    text_rows = np.array([t.startswith(TEXT_NS) for t in inv] + [True] * H_BUCKETS)
    items = [encode_feats(f, vocab) for f in feats]
    aug_items = [encode_feats(f, vocab) for f in aug_feats]
    yr, ya, yg = labels(fit_rows)
    ya, yg = ya.astype(int), yg.astype(int)
    print("vocab %d + %d hashed rows" % (len(vocab), H_BUCKETS))

    caps = []
    for t in range(3):
        am = [tier_amount(s)[1] for s, a in zip(states, ya) if a and tier_amount(s)[0] == t]
        caps.append(float(np.percentile(am, 90)))
    amask = np.array([1.0 if tier_amount(s)[1] < 0.6 * caps[tier_amount(s)[0]] else 0.0 for s in states])
    print("caps(p90 of approved amount per tier):", [round(c, 1) for c in caps], "safe frac %.2f" % amask.mean())

    # ---- repeated k-fold ensemble; out-of-fold logits drive every calibration step ----------
    n = len(items)
    oof = [(np.zeros((n, 5)), np.zeros(n), np.zeros(n)) for _ in range(args.repeats)]
    oofG_aug = np.zeros(n)          # out-of-fold grounded logits on the augmented copies
    nets = []
    for rep in range(args.repeats):
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=rep)
        for k, (a, b) in enumerate(skf.split(np.zeros(n), yr * 2 + ya)):
            ti = [items[i] for i in a] + ([aug_items[i] for i in a] if args.aug else [])
            dup = lambda v: np.concatenate([v[a], v[a]]) if args.aug else v[a]
            m = train_net(ti, dup(yr), dup(ya), dup(yg), dup(amask), n_rows, seed=rep * 100 + k,
                          wide_mask=wide_mask, arch=args.arch, n_num=N_NUM, text_rows=text_rows, **CFG)
            r, aa, g = predict(m, [items[i] for i in b])
            oof[rep][0][b], oof[rep][1][b], oof[rep][2][b] = r, aa, g
            if args.aug:
                oofG_aug[b] += predict(m, [aug_items[i] for i in b])[2] / args.repeats
            nets.append(export(m))
            print("repeat %d fold %d done %.0fs" % (rep, k, time.time() - t0), flush=True)
    oofR = np.mean([o[0] for o in oof], 0); oofA = np.mean([o[1] for o in oof], 0); oofG = np.mean([o[2] for o in oof], 0)

    # numpy serving path must reproduce torch
    sd = nets[-1]
    chk = [items[i] for i in range(32)]
    tr_r, tr_a, tr_g = predict(m, chk)
    f = (lambda it, q: np_forward_typed(sd, *it, q)) if args.arch == "typed" else (lambda it, q: np_forward(sd, *it)[q])
    assert np.allclose(tr_r, np.array([f(it, 0) for it in chk]), atol=1e-3), "numpy != torch (route)"
    assert np.allclose(tr_a, np.array([f(it, 1) for it in chk]), atol=1e-3), "numpy != torch (approve)"
    assert np.allclose(tr_g, np.array([f(it, 2) for it in chk]), atol=1e-3), "numpy != torch (grounded)"

    # ---- calibration ---------------------------------------------------------------------------
    nll = lambda T: -np.log(softmax(oofR / T[0])[np.arange(n), yr] + 1e-12).mean()
    T_route = float(minimize(nll, [1.0], bounds=[(0.3, 5.0)]).x[0])
    pg = LogisticRegression(C=1e4).fit(oofG[:, None], yg)
    g_ab = (float(pg.coef_[0, 0]), float(pg.intercept_[0]))
    Xa = gbm_feats(states, oofA)
    inner = np.zeros(n)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(Xa, ya):
        inner[b] = fit_gbm(Xa[a], ya[a]).predict_proba(Xa[b])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(inner, ya)
    pl = LogisticRegression(C=1e4).fit(logit(inner)[:, None], ya)
    gbm = fit_gbm(Xa, ya)
    ac = {"iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist(),
          "platt": (float(pl.coef_[0, 0]), float(pl.intercept_[0])), "kind": "iso"}

    def thresholds(cal, R, A_gate, G):
        oR = softmax(R / cal.T); oG = sig(cal.g_ab[0] * G + cal.g_ab[1]); oGa = sig(cal.g_ab[0] * oofG_aug + cal.g_ab[1])
        return {"route": float(np.quantile(oR.max(1), 1 - COVERAGE)),
                # grounded: clean + perturbed OOF, so coverage does not collapse when wording drifts
                "grounded": float(np.quantile(np.maximum(oG, 1 - oG) if not args.aug else np.concatenate(
                    [np.maximum(oG, 1 - oG), np.maximum(oGa, 1 - oGa)]), 1 - GROUNDED_COVERAGE)),
                "approve": float(np.quantile(cal.approve_rank(A_gate), COVERAGE))}

    # choose the approve calibrator on cross-fitted predictions (each calibrator scored on rows it
    # was not fitted on - otherwise isotonic always wins by memorising)
    cross = {k: np.zeros(n) for k in ("iso", "platt", "blend")}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=2).split(inner[:, None], ya):
        iso_k = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(inner[a], ya[a])
        pl_k = LogisticRegression(C=1e4).fit(logit(inner[a])[:, None], ya[a])
        cross["iso"][b] = iso_k.predict(inner[b])
        cross["platt"][b] = pl_k.predict_proba(logit(inner[b])[:, None])[:, 1]
        cross["blend"][b] = 0.5 * (cross["iso"][b] + cross["platt"][b])
    best = None
    for kind in ("iso", "platt", "blend"):
        Pa = cross[kind]
        rank = approve_cost(cross["platt"])
        r = report("oof", None, Pa, None, None, ya.astype(bool), None,
                   {"approve": rank > np.quantile(rank, COVERAGE)})["approve"]
        obj = 0.5 * max(0, 1 - r["ece"] / 0.2) + 0.5 * r["bskill"] + max(0, 1 - r["risk"] / r["err"])
        print("approve calibrator %-5s ece %.4f bskill %.4f risk@80 %.4f -> objective %.4f" % (kind, r["ece"], r["bskill"], r["risk"], obj))
        if best is None or obj > best[0]:
            best = (obj, kind)
    ac["kind"] = best[1]
    cal = Calib(T_route, g_ab, gbm, ac, None)
    cal.thr = thresholds(cal, oofR, inner, oofG)
    print("T_route %.3f  grounded platt %s  approve calibrator %s  abstain %s" % (T_route, g_ab, ac["kind"], cal.thr))

    # ---- repeated-CV estimate: score each repeat's out-of-fold predictions separately --------
    cv_rows = fit_rows
    per = []
    for o in oof:
        rep_ = score(cal, o[0], o[1], o[2], cv_rows)
        per.append(rep_["score_no_latency"]["total_with_full_latency"])
    print("repeated-CV score (OOF, per repeat): %s  mean %.2f  std %.2f" % (
        [round(x, 2) for x in per], np.mean(per), np.std(per, ddof=1) if len(per) > 1 else float("nan")))

    # ---- save -----------------------------------------------------------------------------------
    for fn in os.listdir(args.out):
        if fn.startswith("net_") or fn == "nets.npz":
            os.remove(os.path.join(args.out, fn))
    for i, sd_ in enumerate(nets):
        np.savez_compressed(os.path.join(args.out, "net_%02d.npz" % i), **{k: v.astype(np.float16) for k, v in sd_.items()})
    json.dump(vocab, open(os.path.join(args.out, "vocab.json"), "w"))
    json.dump(export_booster(gbm.booster_), open(os.path.join(args.out, "approve_gate.json"), "w"))
    _g = Gate(export_booster(gbm.booster_))
    _chk = max(abs(_g.prob(list(x)) - p) for x, p in zip(Xa[:200], gbm.predict_proba(Xa[:200])[:, 1]))
    assert _chk < 1e-6, "exported gate disagrees with LightGBM (%g)" % _chk
    json.dump({"T_route": T_route, "grounded_platt": g_ab, "abstain": cal.thr, "approve_threshold": APPROVE_TH,
               "n_nets": len(nets), "caps": caps, "arch": args.arch, "approve_cal": ac,
               "h_buckets": H_BUCKETS, "m2v": not args.no_m2v, "spell": not args.no_spell,
               "approve_iso": {"x": ac["iso_x"], "y": ac["iso_y"]}},
              open(os.path.join(args.out, "calibration.json"), "w"))
    print("saved artifacts %.0fs" % (time.time() - t0))

    # ---- evaluation on everything that was not trained on --------------------------------------
    nets16 = [{k: v.astype(np.float16).astype(np.float32) for k, v in sd_.items()} for sd_ in nets]  # as served
    sets = [("dev", dv)]
    if args.hard and os.path.exists(args.hard):
        sets.append(("hard_dev", load(args.hard)))
    if hold:
        from perturb import hard_state
        hr = random.Random(77)
        sets += [("holdout", hold),
                 ("hard_holdout", [dict(r, state=hard_state(r["state"], hr)[0]) for r in hold])]
    for name, rows in sets:
        it = [encode_feats(featurize(r["state"]), vocab) for r in rows]
        R, A, G = ens_logits(nets16, args.arch, it)
        show(name, score(cal, R, A, G, rows))


if __name__ == "__main__":
    main()
