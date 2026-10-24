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
from decide.net import np_forward, np_forward_typed, np_forward_typed_ens, stack
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
# rows hidden by modality dropout. History sentences (s:) are included again: the cue-similarity
# features carry the same information through the pretrained embedding, so hiding the exact sentence
# teaches the network to read the paraphrase-tolerant version instead of creating label noise.
TEXT_NS = ("p:", "b:", "d:", "pflag:", "s:")


def gbm_feats(states, te, dis=None):
    """Gate inputs: tier, amount, text evidence [, ensemble disagreement on the evidence]."""
    ta = [tier_amount(s) for s in states]
    X = np.array([[t, a, e] for (t, a), e in zip(ta, te)], dtype=np.float64)
    return X if dis is None else np.hstack([X, np.asarray(dis, dtype=np.float64)[:, None]])


def fit_gbm(X, y, seed=0):
    mono = [0, -1, 1] + [0] * (X.shape[1] - 3)
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=1.0, reg_lambda=5.0,
                              monotone_constraints=mono, random_state=seed, verbose=-1).fit(X, y)


def route_conf_feats(P, P0, P1, pflag):
    """Inputs of the route p(correct) model: top prob, margin, entropy, ensemble disagreement, marker."""
    srt = np.sort(P, 1)[:, ::-1]
    ent = -(P * np.log(np.clip(P, 1e-9, 1))).sum(1)
    dis = 0.5 * np.abs(P0 - P1).sum(1)
    return np.column_stack([logit(srt[:, 0]), np.log(srt[:, 0] / np.clip(srt[:, 1], 1e-9, 1)), ent, dis,
                            np.asarray(pflag, dtype=float)])


def rescale_top(P, pc):
    """Replace each row's top probability by pc, scaling the others so the row still sums to 1;
    pc is floored so the top class stays on top."""
    P = P.copy()
    k = P.argmax(1)
    top = P[np.arange(len(P)), k]
    p2 = np.sort(P, 1)[:, -2]
    pc = np.maximum(pc, p2 / np.clip(1 - top + p2, 1e-9, None) + 1e-6)
    scale = (1 - pc) / np.clip(1 - top, 1e-9, None)
    P = P * scale[:, None]
    P[np.arange(len(P)), k] = pc
    return P


class Calib:
    """Everything that turns raw head outputs into served probabilities and abstain flags."""

    def __init__(self, T, g_ab, gbm, approve_cal, thr, bias=None, conf=None, gate_dis=False):
        self.T, self.g_ab, self.gbm, self.ac, self.thr = T, g_ab, gbm, approve_cal, thr
        self.bias = np.zeros(5) if bias is None else np.asarray(bias)
        self.conf, self.gate_dis = conf, gate_dis          # conf: (coef, intercept) of route p(correct)

    def route_p(self, R, R0, R1, pflag):
        P = softmax(R / self.T + self.bias)
        if self.conf is None:
            return P
        P0, P1 = softmax(R0 / self.T + self.bias), softmax(R1 / self.T + self.bias)
        z = route_conf_feats(P, P0, P1, pflag) @ np.asarray(self.conf[0]) + self.conf[1]
        return rescale_top(P, sig(z))

    def approve_p(self, gate_p):
        ac = self.ac
        pi = np.interp(gate_p, ac["iso_x"], ac["iso_y"])
        pp = sig(ac["platt"][0] * logit(gate_p) + ac["platt"][1])
        return {"iso": pi, "platt": pp, "blend": 0.5 * (pi + pp)}[ac["kind"]]

    def approve_rank(self, gate_p):
        """Tie-free version of p (Platt on the gate) used only to RANK cases for the abstain flag:
        isotonic steps put hundreds of cases on one value, which a threshold cannot split."""
        return approve_cost(sig(self.ac["platt"][0] * logit(gate_p) + self.ac["platt"][1]))

    def probs(self, O, states):
        """O: dict of ensemble outputs (R, A, G means; R0/R1/A0/A1 two independent members; pflag)."""
        Pr = self.route_p(O["R"], O["R0"], O["R1"], O["pflag"])
        Pg = sig(self.g_ab[0] * O["G"] + self.g_ab[1])
        Ga = self.gbm.predict_proba(gbm_feats(states, O["A"], np.abs(O["A0"] - O["A1"]) if self.gate_dis else None))[:, 1]
        return Pr, self.approve_p(Ga), Pg, Ga

    def abstain(self, Pr, Ga, Pg):
        return {"route": Pr.max(1) < self.thr["route"],
                "grounded": np.maximum(Pg, 1 - Pg) < self.thr["grounded"],
                "approve": self.approve_rank(Ga) > self.thr["approve"]}


def approve_cost(p):
    """Expected cost of the action taken at the 0.75 threshold (hold costs p, approve costs 3(1-p))."""
    return np.minimum(3 * (1 - p), p)


def mine_cues(rows, safe, min_n=15, max_rate=0.05):
    """Sentences / requests that (almost) always mean HOLD where the amount cap cannot be the reason.
    Returns their pretrained embeddings: {"hold": body sentences, "flag": primary requests}."""
    from collections import defaultdict
    from decide.features import _norm_sentence
    from decide.textproc import segments
    sent_stats, req_stats, text_of = defaultdict(list), defaultdict(list), {}
    for r, ok in zip(rows, safe):
        if not ok:
            continue
        body = textnorm.clean(r["state"].get("body"))
        prim, _, content, _ = segments("", body, side_fn=features.SIDE)
        intent = prim or content
        for x in [x for x in features._SENT.split(body) if x.strip() and x not in intent]:
            k = _norm_sentence(x); sent_stats[k].append(r["labels"]["approve"]); text_of[k] = x
        k = "req:" + _norm_sentence(intent)
        req_stats[k].append(r["labels"]["approve"]); text_of[k] = intent
    pick = lambda st: [text_of[k] for k, v in st.items() if len(v) >= min_n and np.mean(v) <= max_rate]
    hold, flag = pick(sent_stats), pick({k: v for k, v in req_stats.items() if len(v) >= 8})
    enc = lambda xs: (np.stack([features.M2V.encode(x) for x in xs]) if xs else np.zeros((0, 256))).astype(np.float32)
    return {"hold": enc(hold), "flag": enc(flag)}


def labels(rows):
    return (np.array([QUEUES.index(r["labels"]["route"]) for r in rows]),
            np.array([r["labels"]["approve"] for r in rows]).astype(bool),
            np.array([r["labels"]["grounded"] for r in rows]).astype(bool))


def score(cal, O, rows):
    Pr, Pa, Pg, Ga = cal.probs(O, [r["state"] for r in rows])
    yr, ya, yg = labels(rows)
    return report("x", Pr, Pa, Pg, yr, ya, yg, cal.abstain(Pr, Ga, Pg))


def ens_outputs(nets, items, pflag_id, i0, i1):
    """Served-style ensemble outputs, plus two independent members (i0, i1) for disagreement."""
    S = stack(nets)
    O = {k: [] for k in ("R", "A", "G", "R0", "R1", "A0", "A1", "pflag")}
    for ids, w, x in items:
        r = np_forward_typed_ens(S, ids, w, x, 0, members=True)
        a = np_forward_typed_ens(S, ids, w, x, 1, members=True)
        O["R"].append(r.mean(0)); O["R0"].append(r[i0]); O["R1"].append(r[i1])
        O["A"].append(a.mean()); O["A0"].append(a[i0]); O["A1"].append(a[i1])
        O["G"].append(np_forward_typed_ens(S, ids, w, x, 2))
        O["pflag"].append(float(pflag_id in set(ids.tolist())))
    return {k: np.array(v) for k, v in O.items()}


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
    ap.add_argument("--no-v6", action="store_true", help="ablation: no side-opener / cue banks / s: modality drop")
    ap.add_argument("--final", action="store_true", help="also train on dev (dev scores are then in-sample)")
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
    fit_rows = tr + dv if args.final else tr
    if args.final:
        print("FINAL: training on train + dev (%d rows); dev-based scores below are in-sample" % len(fit_rows))
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

    yr, ya, yg = labels(fit_rows)
    ya, yg = ya.astype(int), yg.astype(int)
    caps = []
    for t in range(3):
        am = [tier_amount(s)[1] for s, a in zip(states, ya) if a and tier_amount(s)[0] == t]
        caps.append(float(np.percentile(am, 90)))
    amask = np.array([1.0 if tier_amount(s)[1] < 0.6 * caps[tier_amount(s)[0]] else 0.0 for s in states])
    print("caps(p90 of approved amount per tier):", [round(c, 1) for c in caps], "safe frac %.2f" % amask.mean())

    # learned side-issue opener detector + mined cue banks (training rows only)
    for fn in ("side_opener.json", "cue_bank.npz"):
        if os.path.exists(os.path.join(args.out, fn)):
            os.remove(os.path.join(args.out, fn))
    features.SIDE = features.CUES = None
    if features.M2V is not None and not args.no_v6:
        from decide.roles import SideOpener
        features.SIDE = SideOpener.fit(features.M2V)
        features.SIDE.save(os.path.join(args.out, "side_opener.json"))
        bank = mine_cues(fit_rows, amask)
        np.savez_compressed(os.path.join(args.out, "cue_bank.npz"), **bank)
        features.CUES = bank
        print("cue banks: %d hold sentences, %d always-hold requests" % (len(bank["hold"]), len(bank["flag"])))

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
    text_ns = TEXT_NS if not args.no_v6 else ("p:", "b:", "d:", "pflag:")
    text_rows = np.array([t.startswith(text_ns) for t in inv] + [True] * H_BUCKETS)
    items = [encode_feats(f, vocab) for f in feats]
    aug_items = [encode_feats(f, vocab) for f in aug_feats]
    pflag_id = vocab.get("pflag:1", -1)
    print("vocab %d + %d hashed rows" % (len(vocab), H_BUCKETS))

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
    j1 = 1 if args.repeats > 1 else 0
    pflag = np.array([float(pflag_id in set(it[0].tolist())) for it in items])
    O = {"R": np.mean([o[0] for o in oof], 0), "A": np.mean([o[1] for o in oof], 0), "G": np.mean([o[2] for o in oof], 0),
         "R0": oof[0][0], "R1": oof[j1][0], "A0": oof[0][1], "A1": oof[j1][1], "pflag": pflag}

    # numpy serving path must reproduce torch
    sd = nets[-1]
    chk = [items[i] for i in range(32)]
    tr_r, tr_a, tr_g = predict(m, chk)
    f = (lambda it, q: np_forward_typed(sd, *it, q)) if args.arch == "typed" else (lambda it, q: np_forward(sd, *it)[q])
    assert np.allclose(tr_r, np.array([f(it, 0) for it in chk]), atol=1e-3), "numpy != torch (route)"
    assert np.allclose(tr_a, np.array([f(it, 1) for it in chk]), atol=1e-3), "numpy != torch (approve)"
    assert np.allclose(tr_g, np.array([f(it, 2) for it in chk]), atol=1e-3), "numpy != torch (grounded)"

    # ---- calibration ---------------------------------------------------------------------------
    cv5 = list(StratifiedKFold(5, shuffle=True, random_state=3).split(np.zeros(n), yr))

    def route_only_score(P, idx=None):
        idx = np.arange(n) if idx is None else idx
        Pi = P[idx]
        thr = np.quantile(Pi.max(1), 1 - COVERAGE)
        r = report("r", Pi, None, None, yr[idx], None, None, {"route": Pi.max(1) < thr})
        return r["score_no_latency"]["total_with_full_latency"], r["route"]

    def fit_temp_bias(R, y, with_bias):
        def nll(v):
            b = np.r_[v[1:], 0.0] if with_bias else np.zeros(5)
            return -np.log(softmax(R / v[0] + b)[np.arange(len(y)), y] + 1e-12).mean()
        x0 = [1.0] + ([0.0] * 4 if with_bias else [])
        v = minimize(nll, x0, bounds=[(0.3, 5.0)] + [(-3, 3)] * (len(x0) - 1)).x
        return float(v[0]), (np.r_[v[1:], 0.0] if with_bias else np.zeros(5))

    def fit_conf(P, P0, P1, pf, y):
        X = route_conf_feats(P, P0, P1, pf)
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(X, (P.argmax(1) == y).astype(int))
        return lr.coef_[0], float(lr.intercept_[0])

    # route variants, each scored on cross-fitted predictions
    variants = {}
    for name, with_bias, with_conf in (("temperature", False, False), ("temp+class-bias", True, False),
                                       ("temp+p(correct)", False, True), ("temp+bias+p(correct)", True, True)):
        if with_conf and args.repeats < 2:
            continue
        P_cv = np.zeros((n, 5))
        for a, b in cv5:
            T_, b_ = fit_temp_bias(O["R"][a], yr[a], with_bias)
            P = softmax(O["R"] / T_ + b_)
            if with_conf:
                P0, P1 = softmax(O["R0"] / T_ + b_), softmax(O["R1"] / T_ + b_)
                c = fit_conf(P[a], P0[a], P1[a], pflag[a], yr[a])
                P = rescale_top(P, sig(route_conf_feats(P, P0, P1, pflag) @ c[0] + c[1]))
            P_cv[b] = P[b]
        sc, rr = route_only_score(P_cv)
        variants[name] = (sc, with_bias, with_conf)
        print("route calibration %-22s score %.3f | ece %.4f bskill %.4f risk@80 %.4f err %.4f" % (
            name, sc, rr["ece"], rr["bskill"], rr["risk"], rr["err"]))
    best_r = max(variants, key=lambda k: variants[k][0])
    _, wb, wc = variants[best_r]
    T_route, bias = fit_temp_bias(O["R"], yr, wb)
    conf = None
    if wc:
        P = softmax(O["R"] / T_route + bias)
        c = fit_conf(P, softmax(O["R0"] / T_route + bias), softmax(O["R1"] / T_route + bias), pflag, yr)
        conf = (c[0].tolist(), c[1])
    print("route calibration chosen: %s" % best_r)

    pg = LogisticRegression(C=1e4).fit(O["G"][:, None], yg)
    g_ab = (float(pg.coef_[0, 0]), float(pg.intercept_[0]))

    # approve: gate with / without ensemble disagreement, then the calibrator kind
    dis_a = np.abs(O["A0"] - O["A1"])
    best = None
    for gate_dis in ((False, True) if args.repeats > 1 else (False,)):
        Xa = gbm_feats(states, O["A"], dis_a if gate_dis else None)
        inner = np.zeros(n)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(Xa, ya):
            inner[b] = fit_gbm(Xa[a], ya[a]).predict_proba(Xa[b])[:, 1]
        cross = {k: np.zeros(n) for k in ("iso", "platt", "blend")}
        for a, b in StratifiedKFold(5, shuffle=True, random_state=2).split(inner[:, None], ya):
            iso_k = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(inner[a], ya[a])
            pl_k = LogisticRegression(C=1e4).fit(logit(inner[a])[:, None], ya[a])
            cross["iso"][b] = iso_k.predict(inner[b])
            cross["platt"][b] = pl_k.predict_proba(logit(inner[b])[:, None])[:, 1]
            cross["blend"][b] = 0.5 * (cross["iso"][b] + cross["platt"][b])
        rank = approve_cost(cross["platt"])
        for kind in ("iso", "platt", "blend"):
            r = report("oof", None, cross[kind], None, None, ya.astype(bool), None,
                       {"approve": rank > np.quantile(rank, COVERAGE)})
            obj = r["score_no_latency"]["total_with_full_latency"]
            ra = r["approve"]
            print("approve gate%s calibrator %-5s score %.3f | ece %.4f bskill %.4f risk@80 %.4f cost %.4f" % (
                "+disagreement" if gate_dis else "             ", kind, obj, ra["ece"], ra["bskill"], ra["risk"], ra["cost"]))
            if best is None or obj > best[0]:
                best = (obj, kind, gate_dis, inner, Xa)
    _, kind, gate_dis, inner, Xa = best
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(inner, ya)
    pl = LogisticRegression(C=1e4).fit(logit(inner)[:, None], ya)
    gbm = fit_gbm(Xa, ya)
    ac = {"iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist(),
          "platt": (float(pl.coef_[0, 0]), float(pl.intercept_[0])), "kind": kind}
    print("approve chosen: calibrator %s, disagreement input %s" % (kind, gate_dis))

    cal = Calib(T_route, g_ab, gbm, ac, None, bias=bias, conf=conf, gate_dis=gate_dis)
    oR = cal.route_p(O["R"], O["R0"], O["R1"], pflag)
    oG = sig(g_ab[0] * O["G"] + g_ab[1]); oGa = sig(g_ab[0] * oofG_aug + g_ab[1])
    cal.thr = {"route": float(np.quantile(oR.max(1), 1 - COVERAGE)),
               # grounded: clean + perturbed OOF, so coverage does not collapse when wording drifts
               "grounded": float(np.quantile(np.maximum(oG, 1 - oG) if not args.aug else np.concatenate(
                   [np.maximum(oG, 1 - oG), np.maximum(oGa, 1 - oGa)]), 1 - GROUNDED_COVERAGE)),
               "approve": float(np.quantile(cal.approve_rank(inner), COVERAGE))}
    print("T_route %.3f  bias %s  grounded platt %s  abstain %s" % (T_route, np.round(bias, 3), g_ab, cal.thr))

    # ---- repeated-CV estimate: score each repeat's out-of-fold predictions separately --------
    per = []
    for o in oof:
        Oi = dict(O, R=o[0], A=o[1], G=o[2])
        per.append(score(cal, Oi, fit_rows)["score_no_latency"]["total_with_full_latency"])
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
    i0, i1 = 0, (args.folds if args.repeats > 1 else 0)
    json.dump({"T_route": T_route, "route_bias": bias.tolist(), "route_conf": conf, "grounded_platt": g_ab,
               "abstain": cal.thr, "approve_threshold": APPROVE_TH, "approve_gate_disagreement": gate_dis,
               "disagree_members": [i0, i1], "n_nets": len(nets), "caps": caps, "arch": args.arch, "approve_cal": ac,
               "h_buckets": H_BUCKETS, "m2v": not args.no_m2v, "spell": not args.no_spell,
               "approve_iso": {"x": ac["iso_x"], "y": ac["iso_y"]}},
              open(os.path.join(args.out, "calibration.json"), "w"))
    np.savez_compressed(os.path.join(args.out, "oof.npz"),                 # analysis only, not served
                        **{k: v for k, v in O.items()}, G_aug=oofG_aug, yr=yr, ya=ya, yg=yg)
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
        show(name, score(cal, ens_outputs(nets16, it, pflag_id, i0, i1), rows))


if __name__ == "__main__":
    main()
