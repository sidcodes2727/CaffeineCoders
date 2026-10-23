"""Shared backbone + three heads, torch definition (TRAINING ONLY - serving uses decide/net.py)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PAIR = 5


class SharedNet(nn.Module):
    """
    Token-embedding bag + dense pair scalars -> ONE shared trunk (two residual
    blocks). Heads:
        route      linear(dim -> 5)
        grounded   linear(dim -> 1)
        approve    linear(dim -> 1) + sparse "wide" term over intent / history tokens
                   (a text-evidence logit; the tier/amount policy gate sits downstream)
    The trunk is the only place the decision types meet.
    """

    def __init__(self, vocab: int, dim: int = 128, n_route: int = 5, p: float = 0.3, wide_mask=None):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, dim, mode="sum")
        self.num = nn.Linear(N_PAIR, dim)
        self.ln0 = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(approximate="tanh"), nn.Dropout(p),
                           nn.Linear(dim * 2, dim)) for _ in range(2)])
        self.lns = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.drop = nn.Dropout(p)
        self.h_route = nn.Linear(dim, n_route)
        self.h_approve = nn.Linear(dim, 1)
        self.h_grounded = nn.Linear(dim, 1)
        self.wide = nn.EmbeddingBag(vocab, 1, mode="sum")
        mask = torch.ones(vocab) if wide_mask is None else torch.as_tensor(wide_mask, dtype=torch.float32)
        self.register_buffer("wide_mask", mask.unsqueeze(1))
        nn.init.normal_(self.emb.weight, std=0.05)
        nn.init.zeros_(self.wide.weight)

    def trunk(self, ids, offsets, w, x):
        h = self.emb(ids, offsets, per_sample_weights=w) + self.num(x)
        h = self.drop(F.gelu(self.ln0(h), approximate="tanh"))
        for blk, ln in zip(self.blocks, self.lns):
            h = h + self.drop(blk(ln(h)))
        return h

    def forward(self, ids, offsets, w, x):
        h = self.trunk(ids, offsets, w, x)
        wide = F.embedding_bag(ids, self.wide.weight * self.wide_mask, offsets, mode="sum",
                               per_sample_weights=w)
        a = self.h_approve(h.detach()) + wide
        return self.h_route(h), a.squeeze(-1), self.h_grounded(h).squeeze(-1)


def export(model: SharedNet) -> dict:
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
