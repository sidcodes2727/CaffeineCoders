"""
Decide: one typed-question model, three decision types.

Architecture (typed-question, "System One" style)
-------------------------------------------------
Every request is answered by ONE forward pass of the same network,
    f(state, question) -> typed answer
where the question is an input, not a router:

  clean text (unicode, case, contractions, spell-correction)
  featurise(state) -> sparse tokens + hashed char 3-5-grams  -> embedding bag
                   -> pair scalars + pretrained sentence embeddings (Model2Vec) -> linear
        + question embedding  ──► shared residual trunk, every block FiLM-modulated by the
                                  question (per-question scale/shift)  ──► h (128-d)
  typed output heads, chosen by ANSWER TYPE (not by decision):
    choice  (route)             h . option_embeddings -> 5 logits -> temperature softmax
    binary  (grounded)          shared binary head    -> 1 logit  -> Platt scaling
    binary  (approve)           shared binary head + sparse intent/history term -> evidence logit
                                -> monotone policy gate (tier, amount, evidence) -> calibrated p

Where the sharing happens: `_answer()` - the embedding table, the trunk and the binary head
are the same weights for every question. A fourth yes/no question needs only a new question
embedding (+ calibrator); a new multiple-choice question needs a new set of option embeddings.

Served model = mean of a bagged ensemble of that architecture. Numpy-only, no network calls.
Every public method is wrapped: an unexpected failure returns a valid, abstaining answer.
"""
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from decide import features
from decide.encode import encode
from decide.features import tier_amount
from decide.gate import Gate
from decide.net import Q_APPROVE, Q_GROUNDED, Q_ROUTE, np_forward, np_forward_typed_ens, stack

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _sig(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


class DecisionModel:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = artifacts_dir
        j = lambda n: os.path.join(artifacts_dir, n)
        cal = json.load(open(j("calibration.json")))
        if os.path.exists(j("m2v_emb.npy")) and cal.get("m2v", False):
            features.init(artifacts_dir, speller=cal.get("spell", False))
        self.vocab: Dict[str, int] = json.load(open(j("vocab.json")))
        files = sorted(glob.glob(j("net_*.npz")))
        if files:
            self.nets = [{k: z[k].astype(np.float32) for k in z.files} for z in map(np.load, files)]
        else:                                                   # legacy single-file format
            z = np.load(j("nets.npz"))
            self.nets = [{k.split("/", 1)[1]: z[k].astype(np.float32) for k in z.files if k.startswith("n%d/" % i)}
                         for i in range(cal["n_nets"])]
        self.arch = cal.get("arch", "shared")
        self.stacked = stack(self.nets) if self.arch == "typed" else None
        self.gate = Gate(json.load(open(j("approve_gate.json"))))
        self.T_route = cal["T_route"]
        self.g_a, self.g_b = cal["grounded_platt"]
        ac = cal.get("approve_cal") or {"iso_x": cal["approve_iso"]["x"], "iso_y": cal["approve_iso"]["y"],
                                        "platt": (1.0, 0.0), "kind": "iso"}
        self.ac_kind = ac["kind"]
        self.iso_x, self.iso_y = np.asarray(ac["iso_x"]), np.asarray(ac["iso_y"])
        self.pl_a, self.pl_b = ac["platt"]
        self.thr = cal["abstain"]
        self.approve_th = cal["approve_threshold"]
        self._warm()

    def _warm(self) -> None:
        s = {"subject": "warm up", "body": "The part I need you to action is: refund please. $1.",
             "customer_tier": "pro", "amount": 12.0, "draft_reply": "ok within 3 days",
             "kb_article": "within three days"}
        for _ in range(5):
            self.route(s); self.approve(s); self.grounded(s)

    # -- the one forward pass ------------------------------------------------
    def _answer(self, state: Dict[str, Any], q: int):
        """Ensemble-averaged raw answer for question q: route logits[5] or one binary logit."""
        ids, w, x = encode(state, self.vocab)
        if self.arch == "typed":
            return np_forward_typed_ens(self.stacked, ids, w, x, q)
        outs = [np_forward(sd, ids, w, x)[q] for sd in self.nets]       # legacy multi-head net
        return sum(outs) / len(outs)

    # -- routing ------------------------------------------------------------
    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        try:
            p = _softmax(np.asarray(self._answer(state, Q_ROUTE), dtype=np.float64) / self.T_route)
            if not np.all(np.isfinite(p)):
                raise ValueError("non-finite route probabilities")
            k = int(p.argmax())
            probs = {q: float(p[i]) for i, q in enumerate(QUEUES)}
            return QUEUES[k], probs, float(p[k]), bool(p[k] < self.thr["route"])
        except Exception:
            probs = {q: 1.0 / len(QUEUES) for q in QUEUES}
            return QUEUES[0], probs, probs[QUEUES[0]], True

    # -- approval -----------------------------------------------------------
    def _approve_p(self, gate_p: float) -> float:
        pi = float(np.interp(gate_p, self.iso_x, self.iso_y))
        g = min(max(gate_p, 1e-4), 1 - 1e-4)
        pp = _sig(self.pl_a * np.log(g / (1 - g)) + self.pl_b)
        return {"iso": pi, "platt": pp, "blend": 0.5 * (pi + pp)}[self.ac_kind]

    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        try:
            a = float(self._answer(state, Q_APPROVE))
            t, amt = tier_amount(state)
            gp = self.gate.prob([float(t), amt, a])
            p = self._approve_p(gp)
            if not np.isfinite(p):
                raise ValueError("non-finite approve probability")
            # A wrong approval costs 3x a wrong hold, so approve only when p > 3/(3+1).
            # Abstain on the highest expected cost of the action taken, ranked on the tie-free
            # Platt view of the gate (isotonic steps would tie hundreds of cases).
            g = min(max(gp, 1e-4), 1 - 1e-4)
            pr = _sig(self.pl_a * np.log(g / (1 - g)) + self.pl_b)
            exp_cost = min(3.0 * (1.0 - pr), pr)
            return p > self.approve_th, p, bool(exp_cost > self.thr["approve"])
        except Exception:
            return False, 0.4, True

    # -- grounding ----------------------------------------------------------
    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        try:
            g = float(self._answer(state, Q_GROUNDED))
            p = _sig(self.g_a * g + self.g_b)
            if not np.isfinite(p):
                raise ValueError("non-finite grounded probability")
            return p > 0.5, p, bool(max(p, 1.0 - p) < self.thr["grounded"])
        except Exception:
            return True, 0.6, True
