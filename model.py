"""
Decide: one typed-question model, three decision types.

Architecture (typed-question, "System One" style)
-------------------------------------------------
Every request is answered by ONE forward pass of the same network,
    f(state, question) -> typed answer
where the question is an input, not a router:

  featurise(state) -> embedding bag + pair scalars
        + question embedding  ──► shared residual trunk, every block FiLM-modulated by the
                                  question (per-question scale/shift)  ──► h (128-d)
  typed output heads, chosen by ANSWER TYPE (not by decision):
    choice  (route)             h . option_embeddings -> 5 logits -> temperature softmax
    binary  (grounded)          shared binary head    -> 1 logit  -> Platt scaling
    binary  (approve)           shared binary head + sparse intent/history term -> evidence logit
                                -> monotone policy gate (tier, amount, evidence) -> isotonic

Where the sharing happens: `_answer()` - the embedding table, the trunk and the binary head
are the same weights for every question. A fourth yes/no question needs only a new question
embedding (+ calibrator); a new multiple-choice question needs a new set of option embeddings.

The served model is the average of a 10-network bagged ensemble of that architecture.
Serving is numpy-only: no torch, no network calls.
"""
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from decide.encode import encode
from decide.features import tier_amount
from decide.gate import Gate
from decide.net import Q_APPROVE, Q_GROUNDED, Q_ROUTE, np_forward, np_forward_typed

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class DecisionModel:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = artifacts_dir
        j = lambda n: os.path.join(artifacts_dir, n)
        self.vocab: Dict[str, int] = json.load(open(j("vocab.json")))
        cal = json.load(open(j("calibration.json")))
        z = np.load(j("nets.npz"))
        self.nets = [{k.split("/", 1)[1]: z[k] for k in z.files if k.startswith("n%d/" % i)}
                     for i in range(cal["n_nets"])]
        self.arch = cal.get("arch", "shared")
        self.gate = Gate(json.load(open(j("approve_gate.json"))))
        self.T_route = cal["T_route"]
        self.g_a, self.g_b = cal["grounded_platt"]
        self.iso_x = np.asarray(cal["approve_iso"]["x"])
        self.iso_y = np.asarray(cal["approve_iso"]["y"])
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
            return sum(np_forward_typed(sd, ids, w, x, q) for sd in self.nets) / len(self.nets)
        outs = [np_forward(sd, ids, w, x)[q] for sd in self.nets]       # legacy multi-head net
        return sum(outs) / len(outs)

    # -- routing ------------------------------------------------------------
    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        p = _softmax(self._answer(state, Q_ROUTE) / self.T_route)
        k = int(p.argmax())
        probs = {q: float(p[i]) for i, q in enumerate(QUEUES)}
        return QUEUES[k], probs, float(p[k]), bool(p[k] < self.thr["route"])

    # -- approval -----------------------------------------------------------
    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        a = float(self._answer(state, Q_APPROVE))
        t, amt = tier_amount(state)
        p = float(np.interp(self.gate.prob([float(t), amt, a]), self.iso_x, self.iso_y))
        # A wrong approval costs 3x a wrong hold, so approve only when p > 3/(3+1).
        exp_cost = min(3.0 * (1.0 - p), p)
        return p > self.approve_th, p, bool(exp_cost > self.thr["approve"])

    # -- grounding ----------------------------------------------------------
    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        g = float(self._answer(state, Q_GROUNDED))
        p = float(1.0 / (1.0 + np.exp(-(self.g_a * g + self.g_b))))
        return p > 0.5, p, bool(max(p, 1.0 - p) < self.thr["grounded"])
