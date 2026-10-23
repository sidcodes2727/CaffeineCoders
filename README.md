# Decide — a sub-100ms typed decision engine on CPU

One small typed-question network answers three questions about a support request
(`route`, `approve`, `grounded`) with calibrated probabilities and an abstain flag:
`f(state, question) → typed answer + probability`, in one forward pass, nothing generated.
CPU only, no network calls at serving time, **p95 ≈ 1.5 ms** per call.
Dev-split self-check: **83.2 / 100**.

## Start the service (single command, clean 4-core Linux box)

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000
```

Weights are committed in `artifacts/` (69 MB). Health check: `curl localhost:8000/health`.
Score it: `python selfcheck.py --url http://localhost:8000/v1/decide` (writes `results.json`).

Serving needs only `fastapi`, `uvicorn`, `numpy` — no torch, no LightGBM.

## Where the sharing happens (one model, not three)

`model.py::DecisionModel._answer()` is the one forward pass behind every request. The
**question is an input to the network**, not a router in front of separate models:

```
state ──► decide/features.py (task-agnostic tokens + pair scalars) ──► EmbeddingBag
question ──► question embedding ──┐                                        │
                                  ├──► + ───► shared residual trunk ──► h (128-d)
                                  └──► FiLM: per-question scale/shift in every block
                                                                           │
                 typed output heads, chosen by ANSWER TYPE ───────────────┤
   choice  (route)     h · option_embeddings → 5 logits → temperature softmax
   binary  (grounded)  shared binary head → 1 logit → Platt scaling
   binary  (approve)   same binary head + sparse intent/history term → evidence logit
                       → monotone policy gate (tier, amount, evidence) → isotonic → p_true
```

What is shared: the embedding table, the whole trunk, and the binary head (used by both yes/no
questions). What is question-specific: one 128-d question embedding and its FiLM projection.
Adding a fourth yes/no decision = one new question embedding + a calibrator; a new choice decision
= a new set of option embeddings. Code: `decide/torchnet.py::TypedNet` (training),
`decide/net.py::np_forward_typed` (serving).

Training (`train.py`, torch) runs every state through all three questions per batch and optimises
the joint loss. Prediction is the mean of a 10-network bagged ensemble (2 × 5-fold) of the same
architecture. The earlier multi-head variant (`SharedNet`, `--arch shared`) is kept for comparison:
81.6 on dev vs 82.2 for the typed net at the same ensemble size.

## What each decision compares

| decision | what it is really about | how the model sees it |
|---|---|---|
| `route` | the **primary request** in a ticket, not the subject line or decoy clauses ("Less important, but…") | `decide/textproc.py` extracts the sentence after markers like "Priority for me is:"; unigram+bigram tokens |
| `approve` | **policy**: per-tier amount cap, always-hold intents (2FA reset, deletion, partnerships), history cues ("I raised something similar…"), plus ~14% hidden account history | text-evidence logit from the trunk, then a monotone tier×amount gate; threshold **0.75** (3:1 cost) |
| `grounded` | **draft vs article**, a pair comparison | `decide/pairfeats.py` canonicalises numbers/durations ("fourteen days" = "two weeks" = 20160 min, "three hundred" = 300, "sixty-five" = 65) and emits draft-only / article-only / cross-pair tokens |

## Calibration and abstention

* Every calibrator is fitted on **out-of-fold** logits (2 × 5-fold CV on train only): temperature (route),
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
python train.py --repeats 2   # shipped config, ~6 min on a laptop CPU; trains on train.jsonl, prints dev metrics
```

`--arch shared` trains the earlier multi-head variant; `--train/--dev` point at other splits.

`python train.py --final` additionally folds `dev.jsonl` into training (not used for the shipped artifacts,
so that `results.json` stays an honest held-out measurement).

## Layout

```
app.py            HTTP layer (unchanged from the starter)
model.py          DecisionModel: typed-question forward pass + calibration + abstention
decide/           features, text/pair processing, numpy net, gate evaluator, torch net (training only)
train.py          CV training, calibration, threshold selection, artifact export
train_lib.py      training loop     evalkit.py   offline copy of the scoring maths
artifacts/        vocab, 10 network weights, calibration, approve gate
results.json      selfcheck output on the dev split     model_card.md   one-page model card
```

Notes: numpy 2.x on Apple-silicon macOS prints spurious "divide by zero in matmul" RuntimeWarnings;
results are unaffected (offline and served numbers match exactly) and Linux does not show them.
