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

    def __init__(self, vocab: int, dim: int = 128, n_route: int = 5, p: float = 0.3, wide_mask=None,
                 n_num: int = N_PAIR):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, dim, mode="sum")
        self.num = nn.Linear(n_num, dim)
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


class TypedNet(nn.Module):
    """
    Typed-question model: f(state, question) -> typed answer, one forward pass.

    The question is an INPUT, not a router. A learned question embedding is added
    to the pooled state and modulates every residual block (FiLM: per-question
    scale and shift), so the same trunk computes a question-specific view.
    Outputs are typed by answer shape, not by decision:
        choice  : logits = h . option_embedding  (5 queue options)
        binary  : one logit, shared by every yes/no question (approve, grounded)
    A new yes/no question is a new question embedding; nothing else changes.
    Approve additionally gets a sparse "wide" term over intent/history tokens; its
    tier/amount policy gate sits downstream exactly as in SharedNet.
    """
    Q_ROUTE, Q_APPROVE, Q_GROUNDED = 0, 1, 2

    def __init__(self, vocab: int, dim: int = 128, n_route: int = 5, p: float = 0.3, wide_mask=None,
                 n_q: int = 3, n_num: int = N_PAIR):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, dim, mode="sum")
        self.num = nn.Linear(n_num, dim)
        self.q = nn.Embedding(n_q, dim)
        self.ln0 = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(approximate="tanh"), nn.Dropout(p),
                           nn.Linear(dim * 2, dim)) for _ in range(2)])
        self.lns = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.film = nn.ModuleList([nn.Linear(dim, 2 * dim) for _ in range(2)])
        self.drop = nn.Dropout(p)
        self.options = nn.Parameter(torch.randn(n_route, dim) * 0.05)
        self.opt_bias = nn.Parameter(torch.zeros(n_route))
        self.binary = nn.Linear(dim, 1)
        self.wide = nn.EmbeddingBag(vocab, 1, mode="sum")
        mask = torch.ones(vocab) if wide_mask is None else torch.as_tensor(wide_mask, dtype=torch.float32)
        self.register_buffer("wide_mask", mask.unsqueeze(1))
        nn.init.normal_(self.emb.weight, std=0.05)
        nn.init.zeros_(self.wide.weight)
        for f in self.film:
            nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)

    def trunk(self, ids, offsets, w, x, q):
        qv = self.q(q)
        h = self.emb(ids, offsets, per_sample_weights=w) + self.num(x) + qv
        h = self.drop(F.gelu(self.ln0(h), approximate="tanh"))
        for blk, ln, film in zip(self.blocks, self.lns, self.film):
            g, b = film(qv).chunk(2, -1)
            h = h + self.drop(blk(ln(h) * (1 + g) + b))
        return h

    def forward(self, ids, offsets, w, x):
        """Training convenience: all three questions for the same batch of states."""
        n = len(offsets)
        q = lambda k: torch.full((n,), k, dtype=torch.long)
        hr = self.trunk(ids, offsets, w, x, q(self.Q_ROUTE))
        ha = self.trunk(ids, offsets, w, x, q(self.Q_APPROVE))
        hg = self.trunk(ids, offsets, w, x, q(self.Q_GROUNDED))
        wide = F.embedding_bag(ids, self.wide.weight * self.wide_mask, offsets, mode="sum",
                               per_sample_weights=w)
        route = hr @ self.options.T + self.opt_bias
        a = self.binary(ha) + wide
        return route, a.squeeze(-1), self.binary(hg).squeeze(-1)


def export(model: nn.Module) -> dict:
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
