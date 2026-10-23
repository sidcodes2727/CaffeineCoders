import json, math, random
import numpy as np
import torch
import torch.nn.functional as F
from decide.torchnet import SharedNet, TypedNet

ARCHS = {"shared": SharedNet, "typed": TypedNet}

QUEUES = ["billing", "technical", "account", "shipping", "general"]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def collate(items, idx):
    ids, offs, ws, xs = [], [], [], []
    o = 0
    for i in idx:
        a, w, x = items[i]
        offs.append(o); o += len(a); ids.append(a); ws.append(w); xs.append(x)
    return (torch.from_numpy(np.concatenate(ids)), torch.tensor(offs), torch.from_numpy(np.concatenate(ws)),
            torch.from_numpy(np.stack(xs)))


def train_net(items, yr, ya, yg, amask, n_vocab, epochs=30, dim=128, lr=3e-3, wd=1e-2, bs=64,
              seed=0, p=0.3, w_task=(1.0, 1.0, 1.0), wide_mask=None, wide_wd=0.5, ls=0.0, tok_drop=0.0,
              arch="shared", n_num=None, text_rows=None, modality_drop=0.0):
    """
    modality_drop: per example, with this probability zero ALL sparse request-text rows
    (text_rows mask over embedding rows) so the trunk must also learn to answer from the
    pretrained sentence embedding alone - the fallback for wording it has never seen.
    """
    torch.manual_seed(seed); random.seed(seed)
    kw = {} if n_num is None else {"n_num": n_num}
    m = ARCHS[arch](n_vocab, dim=dim, p=p, wide_mask=wide_mask, **kw)
    text_rows_t = None if text_rows is None else torch.as_tensor(text_rows, dtype=torch.bool)
    wide_p = [m.wide.weight]
    rest = [q for q in m.parameters() if q is not m.wide.weight]
    opt = torch.optim.AdamW([{"params": rest, "weight_decay": wd}, {"params": wide_p, "weight_decay": wide_wd}], lr=lr)
    n = len(items)
    steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    yr_t, ya_t, yg_t = torch.tensor(yr), torch.tensor(ya, dtype=torch.float32), torch.tensor(yg, dtype=torch.float32)
    am_t = torch.tensor(amask, dtype=torch.float32)
    for ep in range(epochs):
        m.train()
        perm = list(range(n)); random.shuffle(perm)
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            ids, offs, w, x = collate(items, idx)
            if tok_drop > 0:
                w = w * (torch.rand_like(w) > tok_drop).float() / (1 - tok_drop)
            if modality_drop > 0 and text_rows_t is not None:
                lens = torch.diff(torch.cat([offs, torch.tensor([len(ids)])]))
                drop_ex = torch.rand(len(idx)) < modality_drop
                drop_tok = drop_ex.repeat_interleave(lens) & text_rows_t[ids]
                w = w * (~drop_tok).float()
            lr_, la, lg = m(ids, offs, w, x)
            loss = (w_task[0] * F.cross_entropy(lr_, yr_t[idx], label_smoothing=ls) +
                    w_task[1] * (F.binary_cross_entropy_with_logits(la, ya_t[idx], reduction='none') * am_t[idx]).sum() / am_t[idx].sum().clamp(min=1) +
                    w_task[2] * F.binary_cross_entropy_with_logits(lg, yg_t[idx]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


@torch.no_grad()
def predict(m, items):
    m.eval()
    out_r, out_a, out_g = [], [], []
    for k in range(0, len(items), 512):
        ids, offs, w, x = collate(items, range(k, min(k + 512, len(items))))
        r, a, g = m(ids, offs, w, x)
        out_r.append(r.numpy()); out_a.append(a.numpy()); out_g.append(g.numpy())
    return np.concatenate(out_r), np.concatenate(out_a), np.concatenate(out_g)
