# Decide — a sub-100ms typed decision engine on CPU

One small shared network answers three typed questions about a support request
(`route`, `approve`, `grounded`) with calibrated probabilities and an abstain flag.
CPU only, no network calls at serving time, **p95 ≈ 1 ms** per call.

## Start the service (single command, clean 4-core Linux box)

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000
```

Weights are committed in `artifacts/` (34 MB). Health check: `curl localhost:8000/health`.
Score it: `python selfcheck.py --url http://localhost:8000/v1/decide` (writes `results.json`).

Serving needs only `fastapi`, `uvicorn`, `numpy` — no torch, no LightGBM.

## Where the sharing happens (one model, not three)

`model.py::DecisionModel._trunk()` is the one forward pass behind every request:

```
state ──► decide/features.py (task-agnostic tokens + pair scalars)
      ──► EmbeddingBag ─► shared residual trunk (decide/net.py) ─► h  (128-d)
                                                   │
              ┌────────────────────────────────────┼───────────────────────────┐
         route head                          grounded head                 approve head
     linear(h) → 5 logits                   linear(h) → 1 logit        linear(h) + sparse intent term
     temperature softmax                    Platt scaling              → "text evidence" logit
                                                                       → monotone gate(tier, amount, evidence)
                                                                       → isotonic → p_true
```

Training (`train.py`, torch) optimises all three heads jointly on the same trunk; the request's
`decision` only selects which head is read. A fourth decision = one more head + calibrator.
Prediction is the mean over a 5-network bagged ensemble of that same architecture.

## What each decision compares

| decision | what it is really about | how the model sees it |
|---|---|---|
| `route` | the **primary request** in a ticket, not the subject line or decoy clauses ("Less important, but…") | `decide/textproc.py` extracts the sentence after markers like "Priority for me is:"; unigram+bigram tokens |
| `approve` | **policy**: per-tier amount cap, always-hold intents (2FA reset, deletion, partnerships), history cues ("I raised something similar…"), plus ~14% hidden account history | text-evidence logit from the trunk, then a monotone tier×amount gate; threshold **0.75** (3:1 cost) |
| `grounded` | **draft vs article**, a pair comparison | `decide/pairfeats.py` canonicalises numbers/durations ("fourteen days" = "two weeks" = 20160 min) and emits draft-only / article-only / cross-pair tokens |

## Calibration and abstention

* Every calibrator is fitted on **out-of-fold** logits (5-fold CV on train only): temperature (route),
  Platt (grounded), isotonic over the gate output (approve). Label smoothing + token dropout keep raw logits near-calibrated.
* `value` for approve is `p_true > 0.75`; `p_true` itself stays the calibrated probability.
* Abstain flags are set from OOF quantiles for a **declared coverage of ≈83.5%** (target is 80%; the margin
  protects against test-set sampling noise). Route/grounded abstain on low max-probability, approve on high
  expected cost `min(3(1-p), p)`.

## Pretrained components

**None.** Everything is trained from scratch on `data/train.jsonl`; no open-weight model is used.
Libraries: FastAPI (MIT), Uvicorn (BSD-3), NumPy (BSD-3); training only: PyTorch (BSD-3),
scikit-learn (BSD-3), SciPy (BSD-3), LightGBM (MIT).

## Reproduce training

```bash
pip install -r requirements-train.txt
# put train.jsonl / dev.jsonl in data/
python train.py            # ~35 s on a laptop CPU; trains on train.jsonl, prints honest dev metrics
```

`python train.py --final` additionally folds `dev.jsonl` into training (not used for the shipped artifacts,
so that `results.json` stays an honest held-out measurement).

## Layout

```
app.py            HTTP layer (unchanged from the starter)
model.py          DecisionModel: shared trunk + three heads + calibration + abstention
decide/           features, text/pair processing, numpy net, gate evaluator, torch net (training only)
train.py          CV training, calibration, threshold selection, artifact export
train_lib.py      training loop     evalkit.py   offline copy of the scoring maths
artifacts/        vocab, 5 network weights, calibration, approve gate
results.json      selfcheck output on the dev split     model_card.md   one-page model card
```

Notes: numpy 2.x on Apple-silicon macOS prints spurious "divide by zero in matmul" RuntimeWarnings;
results are unaffected (offline and served numbers match exactly) and Linux does not show them.
