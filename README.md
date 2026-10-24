# Decide — a sub-100ms typed decision engine on CPU

One small typed-question network answers three questions about a support request
(`route`, `approve`, `grounded`) with calibrated probabilities and an abstain flag:
`f(state, question) → typed answer + probability`, in one forward pass, nothing generated.
CPU only, no network calls at serving time. p95 per call: route ~3 ms, approve ~3 ms, grounded ~25 ms
(cross-encoder); scored mean of per-decision p95 **~11 ms** (full marks at 25 ms).

| official `selfcheck.py` (version v6) | score | mean p95 |
|---|---|---|
| dev split (`results.json`) | **82.96 / 100** | 12.0 ms |
| hard perturbed dev (`tests/hard_results.json`) | **83.38 / 100** | 11.9 ms |
| 9 dev-derived sets incl. 7 complex-query variants (`tests/stress_report.json`) | mean **81.43** (76.2 – 83.5) | ≤ 15 ms |
| repeated CV, 3 seeds, train only | **82.80 ± 0.08** | |
| untouched 15% holdout of train | **81.24** | |

## Start the service (single command, clean 4-core Linux box)

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000
```

Weights are committed in `artifacts/` (174 MB; largest file 83 MB, under GitHub's 100 MB limit). Health check: `curl localhost:8000/health`.
Score it: `python selfcheck.py --url http://localhost:8000/v1/decide` (writes `results.json`).
Serving needs `fastapi`, `uvicorn`, `numpy`, plus `onnxruntime` + `tokenizers` for the grounded
cross-encoder (prebuilt wheels; without them the service still starts and grounded uses the network alone).
No torch, no transformers, no LightGBM at serving time.

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
   route confidence    p(correct) model on [top prob, margin, entropy, disagreement of two independent
                       ensemble members, marker present] rescales the distribution (argmax unchanged)

grounded cascade (decide/cascade.py):  network p ──┐
   hard case? (draft words absent from article,    ├─► logistic combiner ─► p_true
   or network unsure) → fine-tuned NLI MiniLM ─────┘   [net log-odds, cross-encoder logit,
   cross-encoder, int8 ONNX (decide/nli.py)              lexical gap, number mismatch]
```

The cross-encoder is the only part that reads the draft and the article *together* in context, which
is what paraphrases ("money back" vs "refund") and long articles with unrelated facts need. A startup
latency guard switches it off on a CPU too slow to keep grounded p95 under ~55 ms; grounded then uses
the network alone (`DECIDE_NO_CASCADE=1` does the same by hand).

Shared: the embedding table, the whole trunk, the binary head (both yes/no questions). Question-specific:
one 128-d question embedding and its FiLM projection. A fourth yes/no decision = one new question embedding
+ a calibrator. Code: `decide/torchnet.py::TypedNet` (training), `decide/net.py::np_forward_typed_ens`
(serving, all 10 ensemble members in one vectorised pass).

## What each decision compares

| decision | what it is really about | how the model sees it |
|---|---|---|
| `route` | the **primary request**, not the subject or side issues ("Less important, but…") | `decide/textproc.py` takes the text after markers like "Priority for me is:"; without a marker it removes side issues found by keyword rules **and** by a learned side-opener detector (`decide/roles.py`: a logistic model over the pretrained embedding of a sentence's opening clause, so unseen openers like "Quick aside:" are recognised). Word + word-pair tokens, char n-grams, and a pretrained sentence embedding of that request |
| `approve` | **policy**: per-tier amount cap, always-hold intents (2FA reset, deletion, partnerships), history cues ("I raised something similar…"), plus ~14% hidden account history | text-evidence logit from the trunk (incl. cosine similarity of the body to **cue banks mined from the labels**: sentences / requests that are ~always held where the cap cannot be the reason), then a monotone tier×amount gate; approve iff p > **0.75** (3:1 cost) |
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
  wrong-typed fields; numeric `request_id`), text truncated at 2,000 chars per field, and every decision wrapped so an
  unexpected failure returns a valid abstaining answer instead of an error.

## Calibration and abstention

* Every calibrator is fitted on **out-of-fold** logits (2 × 5-fold CV on train) and every option is kept only if it
  wins on cross-fitted predictions: route = temperature + per-class bias + p(correct) recalibration (chosen over
  temperature alone: ECE 0.031 → 0.012 in the smoke run, small gain at full training); grounded = Platt, then the
  cascade; approve = isotonic (over Platt / blend), with the ensemble-disagreement gate input tried and rejected.
* `value` for approve is `p_true > 0.75`; `p_true` itself stays the calibrated probability.
* Abstain thresholds come from OOF quantiles: route and approve target **83.5%** coverage, grounded **88%**
  (fitted on clean + perturbed OOF, so coverage stays above 80% when wording drifts). Approve abstains on the
  highest expected cost `min(3(1-p), p)`, ranked on a tie-free view of the gate (isotonic steps tie many cases).

## Evaluation protocol

* `python train.py --repeats 3 --holdout 0.15` holds out 15% of train that is never used for any choice, reports
  repeated-CV mean ± std, and scores dev, the hard dev set, the holdout and a hard holdout (network only):

| network | repeated CV | holdout | hard holdout | dev | hard dev |
|---|---|---|---|---|---|
| no pretrained embedding / spell / augmentation (ablation) | 80.48 ± 0.16 | 79.99 | 71.90 | 81.12 | 72.10 |
| v5 | 81.76 ± 0.47 | 80.92 | 79.34 | 82.90 | 81.28 |
| **v6** (+ side-opener detector, cue banks, route p(correct)) | **82.80 ± 0.08** | **81.24** | **79.81** | **82.95** | **81.77** |

* **Version comparison**, official `selfcheck.py`, each version served from its own snapshot of code + artifacts
  (`tests/compare_versions.py`; report `tests/compare_report.json`):

| set | v5 network | v5 + cascade | v6 network | **v6 + cascade (shipped)** |
|---|---|---|---|---|
| dev | 83.02 | **83.17** | 83.00 | 82.96 |
| hard_dev | 82.20 | 83.20 | 82.20 | **83.38** |
| buried_request | 81.63 | 82.48 | 82.59 | **82.97** |
| everything (all stresses + surface noise) | 73.08 | 77.15 | 74.19 | **78.02** |
| long_kb (unrelated facts appended to the article) | 80.56 | 82.74 | 80.98 | **82.75** |
| long_ticket | 81.17 | 80.66 | **81.18** | 80.51 |
| multi_issue | 81.75 | 82.36 | 82.21 | **82.64** |
| multi_issue_fresh (side-issue phrasing held out from rules and detector) | 74.78 | 74.72 | **76.28** | 76.16 |
| multi_issue_new | 83.07 | **83.61** | 83.08 | 83.46 |
| **mean** | 80.14 | 81.12 | 80.64 | **81.43** |
| scored latency (mean p95, dev) | 2.0 ms | 9.5 ms | 2.1 ms | 11.0 ms |

  Earlier versions were trained with code that has since changed, so they cannot be re-served; measured at the
  time on dev / hard dev: v1 multi-head net 81.79 / –, v2 typed net 83.22 / **71.20**, v3 robust 82.61 / 82.47.
* **v6-final** (same recipe trained on train + dev, 8,000 rows) is kept outside the repo as an option: repeated CV
  83.13 ± 0.81; its dev-based scores are in-sample, so `results.json` comes from the train-only v6.
* Robustness suite: `python tests/make_hard.py`, `python tests/make_complex.py`, then
  `python tests/stress.py --url ...` and `python tests/robustness.py --url ... --cold` (invariance, directional,
  slice, edge-case, latency/determinism; report `tests/robustness_report.json`).

## Pretrained components

| component | use | licence |
|---|---|---|
| Model2Vec `minishlab/potion-base-8M` (static embeddings, 29,528 × 256) | frozen sentence/word embeddings as input features; exported to `artifacts/m2v_emb.npy` by `tools/export_m2v.py` | MIT |
| its WordPiece vocabulary (from `BAAI/bge-base-en-v1.5`) | tokenisation, re-implemented in `decide/m2v.py` | MIT |
| `cross-encoder/nli-MiniLM2-L6-H768` (82M, NLI cross-encoder) | **fine-tuned** on grounded pairs (`tools/train_nli.py`), int8 ONNX in `artifacts/nli/`, grounded cascade | Apache-2.0 |

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

Ablation flags: `--no-m2v`, `--no-spell`, `--aug 0`, `--no-v6`, `--arch shared` (earlier multi-head variant);
`--final` also trains on dev.

Grounded cascade (needs `transformers`, `onnxruntime`; uses the same 15% holdout as above):

```bash
python tools/train_nli.py --model cross-encoder/nli-MiniLM2-L6-H768 --epochs 1 --lr 2e-5 --bs 16 --out artifacts/nli
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType as Q; \
  quantize_dynamic('artifacts/nli/nli.onnx','artifacts/nli/nli_int8.onnx',weight_type=Q.QInt8,per_channel=True)"
python tools/fit_cascade.py --nn-holdout <dir of the --holdout 0.15 run> --nli artifacts/nli --ship artifacts
```

## Layout

```
app.py            HTTP layer (starter + lenient input coercion; response shape unchanged)
model.py          DecisionModel: typed-question forward pass + calibration + abstention + safe fallback
decide/           textnorm, textproc, roles (side-opener detector), pairfeats, features, m2v, encode, net (numpy),
                  gate, nli + cascade (grounded cross-encoder), torchnet + augment (training only)
train.py          repeated-CV training, calibration, thresholds, evaluation, artifact export
train_lib.py      training loop (token + modality dropout)     evalkit.py  offline copy of the scoring maths
tests/            perturb.py, make_hard.py, make_complex.py, stress.py, robustness.py, compare_versions.py + reports
tools/            export_m2v.py, train_nli.py (cross-encoder fine-tune + ONNX), fit_cascade.py
artifacts/        vocab, 10 network weight files (fp16), calibration, approve gate, Model2Vec table, speller vocab,
                  side_opener.json, cue_bank.npz,
                  nli/ (int8 cross-encoder, tokenizer, cascade combiner)
results.json      selfcheck output on the dev split      model_card.md   one-page model card
```

Notes: numpy 2.x on Apple-silicon macOS prints spurious "divide by zero in matmul" RuntimeWarnings;
results are unaffected (the numpy path is asserted equal to torch during training) and Linux does not show them.
