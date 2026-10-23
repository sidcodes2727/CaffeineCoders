"""
One-off: export Model2Vec potion-base-8M (MIT) into artifacts/ so serving needs no extra packages.

    pip install model2vec && python tools/export_m2v.py

Folds the model's token mapping and per-token weights into one float16 table, writes the
WordPiece vocabulary, and checks the pure-numpy re-implementation (decide/m2v.py) against
StaticModel.encode on real tickets.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from decide.m2v import M2V  # noqa: E402


def main(out="artifacts"):
    from model2vec import StaticModel
    m = StaticModel.from_pretrained("minishlab/potion-base-8M")
    vocab = json.loads(m.tokenizer.to_str())["model"]["vocab"]
    inv = sorted(vocab, key=vocab.get)
    n = len(inv)
    idx = np.arange(n)
    rows = idx if m.token_mapping is None else np.asarray(m.token_mapping)[idx]
    table = m.embedding[rows].astype(np.float32)
    if m.weights is not None:
        table = table * np.asarray(m.weights)[idx][:, None]
    np.save(os.path.join(ROOT, out, "m2v_emb.npy"), table.astype(np.float16))
    with open(os.path.join(ROOT, out, "m2v_vocab.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(inv) + "\n")

    mine = M2V(os.path.join(ROOT, out))
    texts = [json.loads(l)["state"]["body"] for l in open(os.path.join(ROOT, "data", "dev.jsonl"))][:300]
    texts += ["I can't sign-in to my acount!!", "Café — naïve “quotes” 🙏 你好", "", "   "]
    ref = m.encode(texts)
    got = np.stack([mine.encode(t) for t in texts])
    err = np.abs(ref - got).max()
    print("vocab %d x %d, max abs diff vs model2vec: %.2e" % (table.shape[0], table.shape[1], err))
    assert err < 2e-3, "re-implementation does not match model2vec"


if __name__ == "__main__":
    main()
