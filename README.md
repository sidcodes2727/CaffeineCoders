# Decide — a sub-100ms typed decision engine on CPU

One small typed-question network answers three questions about a support request
(`route`, `approve`, `grounded`) with calibrated probabilities and an abstain flag:
`f(state, question) → typed answer + probability`, in one forward pass, nothing generated.
CPU only, no network calls at serving time, **p95 ≈ 2.5 ms** per call.

| official `selfcheck.py` | score | p95 |
|---|---|---|
| dev split (`results.json`) | **82.6 / 100** | 2.5 ms |
| hard perturbed dev (`tests/hard_results.json`) | **82.5 / 100** | 2.6 ms |

## Start the service (single command, clean 4-core Linux box)

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000
```

Weights are committed in `artifacts/` (89 MB; largest file 15 MB). Health check: `curl localhost:8000/health`.
Score it: `python selfcheck.py --url http://localhost:8000/v1/decide` (writes `results.json`).
Serving needs only `fastapi`, `uvicorn`, `numpy` — no torch, no LightGBM, no tokenizer library.

## Where the sharing happens (one model, not three)

`model.py::DecisionModel._answer()` is the one forward pass behind every request. The
**question is an input to the network**, not a router in front of separate models:

```
state ─► clean (unicode, case, contractions, spell-correct) ─► featurise
   sparse tokens + hashed char 3-5-grams ───────────────► EmbeddingBag ─┐
   pair scalars + Model2Vec sentence embeddings ────────► Linear ───────┤
question ─► question embedding ─────────────────────────────────────────┼─► + ─► shared residual trunk ─► h (128-d)
                                └─► FiLM: per-question scale/shift in every block
   typed output heads, chosen by ANSWER TYPE:
   choice  (route)     h · option_embeddings → 5 logits → temperature softmax
   binary  (grounded)  shared binary head → 1 logit → Platt scaling
   binary  (approve)   same binary head + sparse intent/history term → evidence logit
                       → monotone policy gate (tier, amount, evidence) → isotonic → p_true
```

Shared: the embedding table, the whole trunk, the binary head (both yes/no questions). Question-specific:
one 128-d question embedding and its FiLM projection. A fourth yes/no decision = one new question embedding
+ a calibrator. Code: `decide/torchnet.py::TypedNet` (training), `decide/net.py::np_forward_typed_ens`
(serving, all 10 ensemble members in one vectorised pass).

## What each decision compares

| decision | what it is really about | how the model sees it |
|---|---|---|
| `route` | the **primary request**, not the subject or decoy clauses ("Less important, but…") | `decide/textproc.py` takes the text after markers like "Priority for me is:"; if the marker is missing, removes the recognisable decoy clauses instead. Word + word-pair tokens, char n-grams, and a pretrained sentence embedding of that request |
| `approve` | **policy**: per-tier amount cap, always-hold intents (2FA reset, deletion, partnerships), history cues ("I raised something similar…"), plus ~14% hidden account history | text-evidence logit from the trunk, then a monotone tier×amount gate; approve iff p > **0.75** (3:1 cost) |
| `grounded` | **draft vs article**, a pair comparison | `decide/pairfeats.py` canonicalises numbers/durations ("fourteen days" = "two weeks", "three hundred" = 300), stems words, ignores filler, and emits draft-only / article-only / cross-pair tokens + soft alignment features |

## Robustness (the model must not depend on exact template wording)

* **Input hygiene** (`decide/textnorm.py`): unicode/quotes/emoji/whitespace normalised, contractions expanded,
  edit-distance-1 spell correction against the training vocabulary (real English words are never rewritten).
* **Pretrained semantics**: Model2Vec sentence embeddings of the request and of the remaining body feed the trunk.
  **Modality dropout** hides all request-wording tokens for 25% of training examples, so the trunk also learns to
  answer from the embedding alone — the fallback for wording it has never seen.
* **Augmentation** (`decide/augment.py`): one perturbed copy per training ticket (typos, marker dropped, sentences
  shuffled, filler/distractor clauses, synonyms, case; for drafts: openers/closers, typos). Its synonym and filler
  lists are **disjoint** from the test perturbations, and several test perturbation types are never used in training.
* **Serving safety**: lenient input coercion in `app.py` (amounts like `"1,200"`, `"$1.2k"`, `"twelve hundred"`;
  wrong-typed fields; numeric `request_id`), text truncated at 4,000 chars, and every decision wrapped so an
  unexpected failure returns a valid abstaining answer instead of an error.

## Calibration and abstention

* Every calibrator is fitted on **out-of-fold** logits (2 × 5-fold CV on train): temperature (route), Platt
  (grounded), and for approve the best of isotonic / Platt / blend chosen on cross-fitted predictions (isotonic won).
* `value` for approve is `p_true > 0.75`; `p_true` itself stays the calibrated probability.
* Abstain thresholds come from OOF quantiles: route and approve target **83.5%** coverage, grounded **88%**
  (fitted on clean + perturbed OOF, so coverage stays above 80% when wording drifts). Approve abstains on the
  highest expected cost `min(3(1-p), p)`, ranked on a tie-free view of the gate (isotonic steps tie many cases).

## Evaluation protocol

* `python train.py --repeats 3 --holdout 0.15` holds out 15% of train that is never used for any choice, reports
  repeated-CV mean ± std, and scores dev, the hard dev set, the holdout and a hard holdout.
* Results (same protocol and holdout for both rows):

| | repeated CV | holdout | hard holdout | dev | hard dev |
|---|---|---|---|---|---|
| without pretrained embedding / spell / augmentation | 80.48 ± 0.16 | 79.99 | 71.90 | 81.12 | 72.10 |
| **shipped configuration** | **81.91 ± 0.36** | **79.41** | **78.38** | **82.82** | **80.84** |

  The shipped artifacts are the same configuration retrained on all of `train.jsonl` (dev still held out):
  82.6 on dev and 82.5 on the hard dev set via the official `selfcheck.py`.
* Robustness suite: `python tests/make_hard.py` builds the hard set; `python tests/robustness.py --url ... --cold`
  runs invariance, directional, slice, edge-case and latency/determinism tests (report: `tests/robustness_report.json`).

## Pretrained components

| component | use | licence |
|---|---|---|
| Model2Vec `minishlab/potion-base-8M` (static embeddings, 29,528 × 256) | frozen sentence/word embeddings as input features; exported to `artifacts/m2v_emb.npy` by `tools/export_m2v.py` | MIT |
| its WordPiece vocabulary (from `BAAI/bge-base-en-v1.5`) | tokenisation, re-implemented in `decide/m2v.py` | MIT |

Everything else is trained from scratch on `data/train.jsonl`. Libraries: FastAPI (MIT), Uvicorn (BSD-3),
NumPy (BSD-3); training only: PyTorch (BSD-3), scikit-learn (BSD-3), SciPy (BSD-3), LightGBM (MIT), model2vec (MIT).

## Reproduce training

```bash
pip install -r requirements-train.txt
# put train.jsonl / dev.jsonl in data/
python tests/make_hard.py                        # hard dev set used in the printed evaluation
python train.py --repeats 2                      # shipped config, ~6 min on a laptop CPU
python train.py --repeats 3 --holdout 0.15       # evaluation protocol (separate --out recommended)
```

Ablation flags: `--no-m2v`, `--no-spell`, `--aug 0`, `--arch shared` (earlier multi-head variant).

## Layout

```
app.py            HTTP layer (starter + lenient input coercion; response shape unchanged)
model.py          DecisionModel: typed-question forward pass + calibration + abstention + safe fallback
decide/           textnorm, textproc, pairfeats, features, m2v, encode, net (numpy), gate,
                  torchnet + augment (training only)
train.py          repeated-CV training, calibration, thresholds, evaluation, artifact export
train_lib.py      training loop (token + modality dropout)     evalkit.py  offline copy of the scoring maths
tests/            perturb.py, make_hard.py, robustness.py, hard_results.json, robustness_report.json
tools/            export_m2v.py (one-off pretrained-embedding export)
artifacts/        vocab, 10 network weight files (fp16), calibration, approve gate, Model2Vec table, speller vocab
results.json      selfcheck output on the dev split      model_card.md   one-page model card
```

Notes: numpy 2.x on Apple-silicon macOS prints spurious "divide by zero in matmul" RuntimeWarnings;
results are unaffected (the numpy path is asserted equal to torch during training) and Linux does not show them.
