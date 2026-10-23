"""Numpy inference for the shared network (the only net code the service imports)."""
import numpy as np

N_PAIR = 5


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def np_forward(sd: dict, ids: np.ndarray, w: np.ndarray, x: np.ndarray):
    """Single-example inference. Returns (route_logits[5], approve_text_logit, grounded_logit)."""
    h = (sd["emb.weight"][ids] * w[:, None]).sum(0) + x @ sd["num.weight"].T + sd["num.bias"]
    h = _gelu(_ln(h, sd["ln0.weight"], sd["ln0.bias"]))
    for i in range(2):
        z = _ln(h, sd["lns.%d.weight" % i], sd["lns.%d.bias" % i])
        z = _gelu(z @ sd["blocks.%d.0.weight" % i].T + sd["blocks.%d.0.bias" % i])
        z = z @ sd["blocks.%d.3.weight" % i].T + sd["blocks.%d.3.bias" % i]
        h = h + z
    r = h @ sd["h_route.weight"].T + sd["h_route.bias"]
    wide = float((sd["wide.weight"][ids, 0] * sd["wide_mask"][ids, 0] * w).sum())
    a = float(h @ sd["h_approve.weight"].T + sd["h_approve.bias"]) + wide
    g = float(h @ sd["h_grounded.weight"].T + sd["h_grounded.bias"])
    return r, a, g
