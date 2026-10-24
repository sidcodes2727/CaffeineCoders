# Model card — Decide (v9)

**What it is.** A typed-question decision model, `f(state, question) → typed answer + probability`, in one
forward pass. The question is an input: a question embedding modulates every residual block of one shared trunk
(FiLM); outputs are typed by answer shape (a choice head for route, one binary head shared by approve and
grounded). Inputs: cleaned, spell-corrected text → word / word-pair / sentence tokens + hashed char n-grams; a
learned side-issue detector; draft-vs-article comparison (canonical numbers and durations, business days kept
distinct, scope-hierarchy tokens and a mined forbidden-scope flag); frozen Model2Vec (potion-base-8M, MIT)
embeddings; similarity to hold-cue banks mined from the labels. Approve passes a monotone gate over tier, amount,
evidence and cue similarity. 15-network bagged ensemble (3 seeds × 5 folds), numpy on CPU, no network calls, 219 MB. A fine-tuned NLI
cross-encoder (MiniLM, Apache-2.0) for grounded ships as an opt-in cascade (`DECIDE_CASCADE=1`).

**Trained on.** Shipped artifacts (v9-final): train + dev (8,000) plus one augmented copy per row; calibrators and
abstain thresholds fitted on 3 × 5-fold out-of-fold predictions. All reported held-out numbers come from the same
recipe trained on train only (v9), with dev never used for training, calibration or thresholds.

## Measured (official `selfcheck.py`, v9 train-only, one call at a time)

| | route | approve | grounded |
|---|---|---|---|
| dev quality | macro-F1 **0.944** (majority 0.081) | cost **0.171** (always-hold 0.353) | macro-F1 **0.999** |
| dev accuracy | 92.2% | ~89.5% @0.5 | 99.1% |
| dev ECE | 0.017 | 0.013 | 0.006 |
| p95 latency | ~2 ms | ~2 ms | ~2 ms |

Composite **85.07** on dev, **84.28** on the hard perturbed set, mean **82.59** over 9 dev-derived stress sets.
Repeated CV (3 seeds) **84.54 ± 0.21**; untouched 15% holdout **82.10**. v9-final (shipped) repeated CV on all 8,000
rows: 84.82 ± 0.38. Gates pass; with the unchanged starter `app.py`, 43/61 deliberately malformed inputs get a valid
answer and the rest (string amounts, wrong-typed fields) get a clean HTTP 422; clean-clone start verified
(scored latency 2.3 ms).

## Where it fails

* **Label noise sets the ceiling** (realistic maximum ≈ 86–87; a go/no-go test found no further learnable rule:
  every "will this be wrong?" classifier stays below AUC 0.66 once the known zones are accounted for). Route: ~8% random wrong labels (verified: they do
  not follow subject, side clause or first sentence). Approve: ~16% of safe-looking requests are held for hidden
  account history, flat across every subgroup; the cap edge is fuzzy.
* **Side issues with unfamiliar openers** ("If you get a chance, …"): 78.4 on that set vs 85.4 with known phrasings.
* **Heavily perturbed grounded inputs** (long articles + noise): 76.5 without the cascade, 79.4 with it.
* **Fact synonyms in drafts** and **paraphrased hold cues / red flags** are still read weakly.

## When it abstains

Route: lowest ~16.5% recalibrated top probability. Approve: ~16.5% of cases whose tie-free probability is closest to
0.5. Grounded: lowest ~12% max(p, 1−p). Internal errors abstain with base-rate probabilities.

## Trade-offs made

* **Cascade off by default:** −0.3 on clean dev and 5× slower when on; it is kept for messy, free-form input.
* **Shipping v9-final:** +0.3–0.5 expected from 23% more data (CV 84.85 vs 84.36); `results.json` stays the
  held-out train-only measurement.
* **Rules recovered from labels, not hand-guessed:** cap, history cues, always-hold requests, business days and the
  scope hierarchy were all found in the training data and are learned or mined there.
