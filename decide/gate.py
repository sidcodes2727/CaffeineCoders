"""Pure-numpy evaluator for the approve policy gate (a small monotone GBM exported from LightGBM)."""
import json
from typing import List

import numpy as np


def _flatten(node, feat, thr, left, right, val):
    """Append a subtree in preorder; returns its root index."""
    i = len(feat)
    if "leaf_value" in node:
        feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); val.append(node["leaf_value"])
        return i
    feat.append(node["split_feature"]); thr.append(node["threshold"]); left.append(-1); right.append(-1)
    val.append(0.0)
    left[i] = _flatten(node["left_child"], feat, thr, left, right, val)
    right[i] = _flatten(node["right_child"], feat, thr, left, right, val)
    return i


def export_booster(booster) -> dict:
    trees = []
    for t in booster.dump_model()["tree_info"]:
        feat, thr, left, right, val = [], [], [], [], []
        _flatten(t["tree_structure"], feat, thr, left, right, val)
        trees.append({"f": feat, "t": thr, "l": left, "r": right, "v": val})
    return {"trees": trees}


class Gate:
    def __init__(self, spec: dict):
        self.trees = [(t["f"], t["t"], t["l"], t["r"], t["v"]) for t in spec["trees"]]

    def raw(self, x: List[float]) -> float:
        s = 0.0
        for f, t, l, r, v in self.trees:
            i = 0
            while f[i] >= 0:
                i = l[i] if x[f[i]] <= t[i] else r[i]
            s += v[i]
        return s

    def prob(self, x: List[float]) -> float:
        return 1.0 / (1.0 + np.exp(-self.raw(x)))
