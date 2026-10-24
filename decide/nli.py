"""
Grounding cross-encoder (serving side): a fine-tuned NLI MiniLM (cross-encoder/nli-MiniLM2-L6-H768,
Apache-2.0), int8 ONNX, CPU. Premise = KB article, hypothesis = draft reply; returns one logit for
"the draft is supported". Used by the grounded cascade only (see decide/cascade.py).

Needs onnxruntime + tokenizers at serving time; if either is missing or the files are absent the
cascade switches itself off and grounded falls back to the shared network.
"""
import json
import os
from typing import Optional

import numpy as np


class CrossEncoder:
    def __init__(self, nli_dir: str, threads: int = 4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        cfg = json.load(open(os.path.join(nli_dir, "nli_config.json")))
        self.max_len = int(cfg.get("max_len", 192))
        self.inputs = cfg.get("inputs", ["input_ids", "attention_mask"])
        self.tok = Tokenizer.from_file(os.path.join(nli_dir, "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=self.max_len, strategy="longest_first")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(os.path.join(nli_dir, "nli_int8.onnx"), so,
                                         providers=["CPUExecutionProvider"])

    @staticmethod
    def _case(text: str) -> str:
        """The encoder is case-sensitive; SHOUTED text is out of distribution, so give it normal case."""
        letters = [c for c in text if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
            return ". ".join(s[:1].upper() + s[1:] for s in text.lower().split(". "))
        return text

    def logit(self, kb: str, draft: str) -> float:
        e = self.tok.encode(self._case(kb or ""), self._case(draft or ""))
        feed = {"input_ids": np.asarray([e.ids], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask], dtype=np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.asarray([e.type_ids], dtype=np.int64)
        return float(self.sess.run(None, feed)[0][0])


def load(nli_dir: str) -> Optional[CrossEncoder]:
    try:
        if not os.path.exists(os.path.join(nli_dir, "nli_int8.onnx")):
            return None
        return CrossEncoder(nli_dir)
    except Exception:
        return None
