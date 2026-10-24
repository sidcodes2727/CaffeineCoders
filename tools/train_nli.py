"""
Fine-tune a pretrained NLI cross-encoder for `grounded` (premise = KB article, hypothesis = draft),
then export it to int8 ONNX for CPU serving.

    python tools/train_nli.py --model cross-encoder/nli-MiniLM2-L6-H768 --out artifacts/nli
    python tools/train_nli.py --model MoritzLaurer/xtremedistil-l6-h256-mnli-fever-anli-ling-binary --out ...

The binary score is built from the pretrained NLI head (entailment vs the rest), so fine-tuning
starts from what the model already knows about paraphrase and entailment. Training uses train.jsonl
minus the same 15% holdout as `train.py --holdout 0.15` (that holdout fits the cascade combiner),
plus one augmented copy per row (typos, openers/closers, unrelated article sentences, fact-word
paraphrases from a training-only table).
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from decide.augment import TRAIN_FACT_SYNONYMS, augment_state, build_pools  # noqa: E402


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def split_holdout(rows, frac=0.15, seed=1234):
    """Identical to train.py --holdout."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    k = int(len(rows) * frac)
    hold = set(idx[:k])
    return [r for i, r in enumerate(rows) if i not in hold], [r for i, r in enumerate(rows) if i in hold]


def fact_syn(text, rng, p=0.7):
    for a, b in TRAIN_FACT_SYNONYMS:
        if rng.random() < p:
            text = re.sub(r"\b%s\b" % re.escape(a), b, text, flags=re.I)
    return text


class Binary(nn.Module):
    """Pretrained NLI model -> one 'draft is supported' logit (entailment vs everything else)."""

    def __init__(self, hf, ent_idx):
        super().__init__()
        self.hf, self.ent = hf, ent_idx

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {} if token_type_ids is None else {"token_type_ids": token_type_ids}
        z = self.hf(input_ids=input_ids, attention_mask=attention_mask, **kw).logits
        others = torch.cat([z[:, :self.ent], z[:, self.ent + 1:]], 1)
        return z[:, self.ent] - torch.logsumexp(others, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cross-encoder/nli-MiniLM2-L6-H768")
    ap.add_argument("--out", default="artifacts/nli")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--holdout", type=float, default=0.15, help="0 = train on all of train.jsonl")
    args = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from perturb import synonyms as test_synonyms
    t0 = time.time()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForSequenceClassification.from_pretrained(args.model)
    ent = {v.lower(): k for k, v in hf.config.id2label.items()}["entailment"]
    model = Binary(hf, ent).to(dev)

    tr = load(os.path.join(ROOT, "data", "train.jsonl"))
    if args.holdout > 0:
        tr, _ = split_holdout(tr, args.holdout)
    rng = random.Random(5)
    pools = build_pools(tr)
    pairs = []
    for r in tr:
        s, y = r["state"], float(r["labels"]["grounded"])
        pairs.append((s["kb_article"], s["draft_reply"], y))
        a = augment_state(s, rng, r["labels"]["route"], pools)
        pairs.append((a["kb_article"], fact_syn(a["draft_reply"], rng) if rng.random() < 0.6 else a["draft_reply"], y))
    print("training pairs %d on %s" % (len(pairs), dev), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(pairs) / args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    lossf = nn.BCEWithLogitsLoss()
    step = 0
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(pairs)
        for i in range(0, len(pairs), args.bs):
            b = pairs[i:i + args.bs]
            enc = tok([p[0] for p in b], [p[1] for p in b], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            loss = lossf(model(**enc), torch.tensor([p[2] for p in b], device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print("  epoch %d step %d/%d loss %.4f  %.0fs" % (ep, step, steps, loss.item(), time.time() - t0), flush=True)

    # ---- quick evaluation (torch) ----------------------------------------------------------
    model.eval()

    def score(kbs, drafts):
        out = []
        with torch.no_grad():
            for i in range(0, len(kbs), 128):
                enc = tok(kbs[i:i + 128], drafts[i:i + 128], truncation=True, max_length=args.max_len, padding=True,
                          return_tensors="pt").to(dev)
                out.append(model(**enc).float().cpu().numpy())
        return np.concatenate(out)

    dv = load(os.path.join(ROOT, "data", "dev.jsonl"))
    y = np.array([r["labels"]["grounded"] for r in dv])
    kbs = [r["state"]["kb_article"] for r in dv]
    s0 = score(kbs, [r["state"]["draft_reply"] for r in dv])
    syn = [test_synonyms(r["state"]["draft_reply"]) for r in dv]
    ch = np.array([a != r["state"]["draft_reply"] for a, r in zip(syn, dv)])
    s1 = score(kbs, syn)
    from sklearn.metrics import roc_auc_score
    print("dev clean: AUC %.4f acc@0 %.4f | test-synonym drafts (n=%d): acc@0 %.4f (same rows clean %.4f)" % (
        roc_auc_score(y, s0), ((s0 > 0) == y).mean(), ch.sum(), ((s1 > 0) == y)[ch].mean(), ((s0 > 0) == y)[ch].mean()))

    # ---- export: fp32 ONNX -> dynamic int8 --------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    model = model.to("cpu").eval()
    tok.save_pretrained(args.out)
    enc = tok(kbs[:2], [r["state"]["draft_reply"] for r in dv[:2]], padding=True, return_tensors="pt")
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    fp32 = os.path.join(args.out, "nli_fp32.onnx")
    torch.onnx.export(model, tuple(enc[k] for k in names), fp32, input_names=names, output_names=["logit"],
                      dynamic_axes=dict({k: {0: "b", 1: "t"} for k in names}, logit={0: "b"}), opset_version=14)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = os.path.join(args.out, "nli_int8.onnx")
    quantize_dynamic(fp32, q, weight_type=QuantType.QInt8)
    import onnxruntime as ort
    best = None
    for path in (fp32, q):
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        oq = []
        for i in range(300):
            e = tok(kbs[i], dv[i]["state"]["draft_reply"], truncation=True, max_length=args.max_len, return_tensors="np")
            oq.append(sess.run(None, {k: e[k].astype(np.int64) for k in names})[0][0])
        oq = np.array(oq)
        e = tok(kbs[0], dv[0]["state"]["draft_reply"], return_tensors="np")
        feed = {k: e[k].astype(np.int64) for k in names}
        for _ in range(10):
            sess.run(None, feed)
        t = time.perf_counter()
        for _ in range(100):
            sess.run(None, feed)
        ms = (time.perf_counter() - t) / 100 * 1e3
        agree = ((oq > 0) == (s0[:300] > 0)).mean()
        print("%s: agreement with torch %.4f, mean |diff| %.3f, 1-thread latency %.2f ms, %.1f MB" % (
            os.path.basename(path), agree, np.abs(oq - s0[:300]).mean(), ms, os.path.getsize(path) / 1e6))
        if agree >= 0.99 and (best is None or ms < best[1]):
            best = (path, ms)
    keep = best[0] if best else fp32
    for path in (fp32, q):
        if path != keep:
            os.remove(path)
    os.rename(keep, os.path.join(args.out, "nli.onnx"))
    print("kept", os.path.basename(keep), "as nli.onnx")
    json.dump({"base_model": args.model, "max_len": args.max_len, "holdout": args.holdout, "inputs": names,
               "epochs": args.epochs, "lr": args.lr}, open(os.path.join(args.out, "nli_config.json"), "w"))
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
