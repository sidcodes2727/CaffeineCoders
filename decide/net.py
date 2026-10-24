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


Q_ROUTE, Q_APPROVE, Q_GROUNDED = 0, 1, 2


def np_forward_typed(sd: dict, ids: np.ndarray, w: np.ndarray, x: np.ndarray, q: int):
    """
    TypedNet inference for ONE question: one forward pass of the trunk conditioned on q.
    Returns route logits[5] for Q_ROUTE, else a single binary logit.
    """
    qv = sd["q.weight"][q]
    h = (sd["emb.weight"][ids] * w[:, None]).sum(0) + x @ sd["num.weight"].T + sd["num.bias"] + qv
    h = _gelu(_ln(h, sd["ln0.weight"], sd["ln0.bias"]))
    for i in range(2):
        gb = qv @ sd["film.%d.weight" % i].T + sd["film.%d.bias" % i]
        g, b = gb[:len(qv)], gb[len(qv):]
        z = _ln(h, sd["lns.%d.weight" % i], sd["lns.%d.bias" % i]) * (1 + g) + b
        z = _gelu(z @ sd["blocks.%d.0.weight" % i].T + sd["blocks.%d.0.bias" % i])
        z = z @ sd["blocks.%d.3.weight" % i].T + sd["blocks.%d.3.bias" % i]
        h = h + z
    if q == Q_ROUTE:
        return h @ sd["options"].T + sd["opt_bias"]
    logit = float(h @ sd["binary.weight"].T + sd["binary.bias"])
    if q == Q_APPROVE:
        logit += float((sd["wide.weight"][ids, 0] * sd["wide_mask"][ids, 0] * w).sum())
    return logit


def stack(nets: list) -> dict:
    """Stack N TypedNet state dicts along a new leading axis (for np_forward_typed_ens)."""
    return {k: np.stack([sd[k] for sd in nets]) for k in nets[0]}


def np_forward_typed_ens(S: dict, ids: np.ndarray, w: np.ndarray, x: np.ndarray, q: int, members: bool = False):
    """All N ensemble members in one pass; returns the member-averaged answer (same as looping),
    or with members=True the per-member answers (N,5) / (N,)."""
    qv = S["q.weight"][:, q]                                               # (N, d)
    h = np.einsum("nkd,k->nd", S["emb.weight"][:, ids], w)                 # (N, d)
    h = h + np.einsum("nod,d->no", S["num.weight"], x) + S["num.bias"] + qv
    h = _gelu(_ln(h, S["ln0.weight"], S["ln0.bias"]))
    d = qv.shape[1]
    for i in range(2):
        gb = np.einsum("nod,nd->no", S["film.%d.weight" % i], qv) + S["film.%d.bias" % i]
        g, b = gb[:, :d], gb[:, d:]
        z = _ln(h, S["lns.%d.weight" % i], S["lns.%d.bias" % i]) * (1 + g) + b
        z = _gelu(np.einsum("nod,nd->no", S["blocks.%d.0.weight" % i], z) + S["blocks.%d.0.bias" % i])
        z = np.einsum("nod,nd->no", S["blocks.%d.3.weight" % i], z) + S["blocks.%d.3.bias" % i]
        h = h + z
    if q == Q_ROUTE:
        r = np.einsum("nd,nkd->nk", h, S["options"]) + S["opt_bias"]
        return r if members else r.mean(0)
    logit = np.einsum("nd,nod->no", h, S["binary.weight"])[:, 0] + S["binary.bias"][:, 0]
    if q == Q_APPROVE:
        logit = logit + (S["wide.weight"][:, ids, 0] * S["wide_mask"][:, ids, 0] * w).sum(1)
    return logit if members else float(logit.mean())
