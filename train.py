"""
Train the shared Decide model.

    python train.py                       # train on data/train.jsonl, evaluate honestly on data/dev.jsonl
    python train.py --final               # also fold dev into training (artifacts for deployment)

Writes everything the service needs into artifacts/.
"""
import argparse, json, os, pickle, sys, time
import numpy as np
import torch
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide.encode import build_vocab, encode
from decide.features import tier_amount
from decide.torchnet import export
from decide.gate import Gate, export_booster
from train_lib import QUEUES, load, predict, train_net
from evalkit import report

torch.set_num_threads(4)
sig = lambda z: 1 / (1 + np.exp(-z))
softmax = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
COVERAGE = 0.835          # declared coverage; target is 0.80 so keep margin for test-set noise
APPROVE_TH = 0.75         # cost 3:1 -> approve iff p > 3/(3+1)
CFG = dict(epochs=20, dim=128, p=0.5, wd=0.1, wide_wd=0.02, ls=0.1, tok_drop=0.2)


def gbm_feats(states, te):
    ta = [tier_amount(s) for s in states]
    return np.array([[t, a, e] for (t, a), e in zip(ta, te)], dtype=np.float64)


def fit_gbm(X, y, seed=0):
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=1.0, reg_lambda=5.0,
                              monotone_constraints=[0, -1, 1], random_state=seed, verbose=-1).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--arch", default="typed", choices=["typed", "shared"])
    ap.add_argument("--repeats", type=int, default=1, help="repeated k-fold; ensemble = repeats x folds nets")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    args = ap.parse_args()
    t0 = time.time()
    tr, dv = load(args.train), load(args.dev)
    fit_rows = tr + dv if args.final else tr
    states = [r["state"] for r in fit_rows]
    vocab = build_vocab(states)
    inv = sorted(vocab, key=vocab.get)
    wide_mask = np.array([1.0 if t.startswith(("p:", "s:")) else 0.0 for t in inv], dtype=np.float32)
    items = [encode(s, vocab) for s in states]
    yr = np.array([QUEUES.index(r["labels"]["route"]) for r in fit_rows])
    ya = np.array([r["labels"]["approve"] for r in fit_rows]).astype(int)
    yg = np.array([r["labels"]["grounded"] for r in fit_rows]).astype(int)

    # Region where the amount cap is not what decides approval, learnt from the data:
    # per tier, the 90th percentile of approved amounts is the cap's upper edge.
    caps = []
    for t in range(3):
        am = [tier_amount(s)[1] for s, a in zip(states, ya) if a and tier_amount(s)[0] == t]
        caps.append(float(np.percentile(am, 90)))
    amask = np.array([1.0 if tier_amount(s)[1] < 0.6 * caps[tier_amount(s)[0]] else 0.0 for s in states])
    print("caps(p90 of approved amount per tier):", [round(c, 1) for c in caps], "safe frac %.2f" % amask.mean())

    # ---- k-fold ensemble; out-of-fold logits drive every calibration step ----------------
    # With repeats > 1 each sample's OOF logit is the mean over `repeats` nets that never
    # saw it, which is closer to the averaged ensemble that serves.
    oofR, oofA, oofG = np.zeros((len(items), 5)), np.zeros(len(items)), np.zeros(len(items))
    nets = []
    for rep in range(args.repeats):
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=rep)
        for k, (a, b) in enumerate(skf.split(np.zeros(len(yr)), yr * 2 + ya)):
            m = train_net([items[i] for i in a], yr[a], ya[a], yg[a], amask[a], len(vocab),
                          seed=rep * 100 + k, wide_mask=wide_mask, arch=args.arch, **CFG)
            r, aa, g = predict(m, [items[i] for i in b])
            oofR[b] += r / args.repeats; oofA[b] += aa / args.repeats; oofG[b] += g / args.repeats
            nets.append(export(m))
            print("repeat", rep, "fold", k, "done %.0fs" % (time.time() - t0), flush=True)

    # ---- calibration -----------------------------------------------------------------------
    nll = lambda T: -np.log(softmax(oofR / T[0])[np.arange(len(yr)), yr] + 1e-12).mean()
    T_route = float(minimize(nll, [1.0], bounds=[(0.3, 5.0)]).x[0])
    pg = LogisticRegression(C=1e4).fit(oofG[:, None], yg)
    g_ab = (float(pg.coef_[0, 0]), float(pg.intercept_[0]))
    Xa = gbm_feats(states, oofA)
    inner = np.zeros(len(yr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(Xa, ya):
        inner[b] = fit_gbm(Xa[a], ya[a]).predict_proba(Xa[b])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(inner, ya)
    gbm = fit_gbm(Xa, ya)
    print("T_route %.3f  grounded platt %s" % (T_route, g_ab))

    # ---- abstention thresholds from OOF confidence ---------------------------------------
    oR = softmax(oofR / T_route)
    oG = sig(g_ab[0] * oofG + g_ab[1])
    oA = iso.predict(inner)
    thr = {
        "route": float(np.quantile(oR.max(1), 1 - COVERAGE)),
        "grounded": float(np.quantile(np.maximum(oG, 1 - oG), 1 - COVERAGE)),
        "approve": float(np.quantile(np.minimum(3 * (1 - oA), oA), COVERAGE)),   # expected cost of the action taken
    }
    print("abstain thresholds", thr)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "nets.npz"),
                        **{"n%d/%s" % (i, k): v for i, sd in enumerate(nets) for k, v in sd.items()})
    json.dump(vocab, open(os.path.join(args.out, "vocab.json"), "w"))
    json.dump(export_booster(gbm.booster_), open(os.path.join(args.out, "approve_gate.json"), "w"))
    _g = Gate(export_booster(gbm.booster_))
    _chk = max(abs(_g.prob(list(x)) - p) for x, p in zip(Xa[:200], gbm.predict_proba(Xa[:200])[:, 1]))
    assert _chk < 1e-6, "exported gate disagrees with LightGBM (%g)" % _chk
    json.dump({"T_route": T_route, "grounded_platt": g_ab, "abstain": thr, "approve_threshold": APPROVE_TH,
               "n_nets": len(nets), "caps": caps, "arch": args.arch,
               "approve_iso": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}},
              open(os.path.join(args.out, "calibration.json"), "w"))
    print("saved artifacts %.0fs" % (time.time() - t0))

    # ---- honest evaluation on dev (only meaningful without --final) ------------------------
    items_d = [encode(r["state"], vocab) for r in dv]
    from decide.net import np_forward, np_forward_typed  # numpy path == serving path
    Rl, Al, Gl = [], [], []
    for it in items_d:
        if args.arch == "typed":
            Rl.append(np.mean([np_forward_typed(sd, *it, 0) for sd in nets], 0))
            Al.append(np.mean([np_forward_typed(sd, *it, 1) for sd in nets]))
            Gl.append(np.mean([np_forward_typed(sd, *it, 2) for sd in nets]))
        else:
            rr = [np_forward(sd, it[0], it[1], it[2]) for sd in nets]
            Rl.append(np.mean([x[0] for x in rr], 0)); Al.append(np.mean([x[1] for x in rr])); Gl.append(np.mean([x[2] for x in rr]))
    Rl, Al, Gl = np.array(Rl), np.array(Al), np.array(Gl)
    # the numpy path must reproduce torch exactly
    _r, _a, _g = predict(m, items_d[:64])
    if args.repeats * args.folds == 1:
        assert np.allclose(_r, Rl[:64], atol=1e-3)
    else:
        sd = nets[-1]
        f = (lambda it, q: np_forward_typed(sd, *it, q)) if args.arch == "typed" else None
        if f:
            assert np.allclose(_r, np.array([f(it, 0) for it in items_d[:64]]), atol=1e-3), "numpy != torch (route)"
            assert np.allclose(_a, np.array([f(it, 1) for it in items_d[:64]]), atol=1e-3), "numpy != torch (approve)"
            assert np.allclose(_g, np.array([f(it, 2) for it in items_d[:64]]), atol=1e-3), "numpy != torch (grounded)"
    Pr = softmax(Rl / T_route)
    Pg = sig(g_ab[0] * Gl + g_ab[1])
    Pa = iso.predict(gbm.predict_proba(gbm_feats([r["state"] for r in dv], Al))[:, 1])
    dr = np.array([QUEUES.index(r["labels"]["route"]) for r in dv])
    da = np.array([r["labels"]["approve"] for r in dv]); dg = np.array([r["labels"]["grounded"] for r in dv])
    ab = {"route": Pr.max(1) < thr["route"],
          "grounded": np.maximum(Pg, 1 - Pg) < thr["grounded"],
          "approve": np.minimum(3 * (1 - Pa), Pa) > thr["approve"]}
    rep = report("dev", Pr, Pa, Pg, dr, da, dg, ab)
    for k, v in rep.items():
        print(k, {a: round(float(b), 4) for a, b in v.items()})


if __name__ == "__main__":
    main()
