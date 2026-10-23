"""
Decide: one shared model, three decision types.

Where the sharing happens
-------------------------
`DecisionModel._trunk()` is the single forward pass every request goes through:
  featurise(state) -> embedding bag + pair scalars -> shared residual trunk -> h (128-d)
The three decisions are only different *heads* read off h:
  route     linear(h) -> 5 logits                        -> temperature-scaled softmax
  grounded  linear(h) -> 1 logit                         -> Platt-scaled probability
  approve   linear(h) + sparse intent/history term       -> "text evidence" logit, then a small
            monotone gate over (tier, amount, evidence)  -> isotonic-calibrated probability
Adding a fourth decision means adding one more head (and its calibrator) on h.

Serving is numpy-only: no torch, no network, no per-request allocation beyond the arrays.
"""
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from decide.encode import encode
from decide.features import tier_amount
from decide.gate import Gate
from decide.net import np_forward

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

    # -- shared trunk -------------------------------------------------------
    def _trunk(self, state: Dict[str, Any]) -> Tuple[np.ndarray, float, float]:
        """Ensemble-averaged head logits (route[5], approve-evidence, grounded)."""
        ids, w, x = encode(state, self.vocab)
        r = a = g = 0.0
        for sd in self.nets:
            ri, ai, gi = np_forward(sd, ids, w, x)
            r = r + ri; a += ai; g += gi
        n = len(self.nets)
        return r / n, a / n, g / n

    # -- routing ------------------------------------------------------------
    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        r, _, _ = self._trunk(state)
        p = _softmax(r / self.T_route)
        k = int(p.argmax())
        probs = {q: float(p[i]) for i, q in enumerate(QUEUES)}
        return QUEUES[k], probs, float(p[k]), bool(p[k] < self.thr["route"])

    # -- approval -----------------------------------------------------------
    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        _, a, _ = self._trunk(state)
        t, amt = tier_amount(state)
        p = float(np.interp(self.gate.prob([float(t), amt, a]), self.iso_x, self.iso_y))
        # A wrong approval costs 3x a wrong hold, so approve only when p > 3/(3+1).
        exp_cost = min(3.0 * (1.0 - p), p)
        return p > self.approve_th, p, bool(exp_cost > self.thr["approve"])

    # -- grounding ----------------------------------------------------------
    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        _, _, g = self._trunk(state)
        p = float(1.0 / (1.0 + np.exp(-(self.g_a * g + self.g_b))))
        return p > 0.5, p, bool(max(p, 1.0 - p) < self.thr["grounded"])
