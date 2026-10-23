# Model card — Decide

**What it is.** A typed-question decision model, `f(state, question) → typed answer + probability`, in one
forward pass. The question is an input: a question embedding is added to the pooled state and modulates every
residual block of one shared trunk (FiLM). Outputs are typed by answer shape: a choice head (route, 5 queue options)
and one binary head shared by approve and grounded. Inputs: cleaned and spell-corrected text → sparse word / word-pair
/ sentence tokens + hashed char n-grams, draft-vs-article comparison features, and frozen pretrained Model2Vec
(potion-base-8M, MIT) sentence embeddings. 10-network bagged ensemble, 89 MB, numpy-only CPU inference, no network calls.

**Trained on.** `data/train.jsonl` (6,500 requests) plus one augmented copy per row (typos, dropped markers,
shuffles, filler, synonyms). Dev (1,500) was held out from training, calibration and thresholds. Calibrators and
abstain thresholds come from 2 × 5-fold out-of-fold predictions. Separately, an evaluation run held out 15% of train
that was never used for any choice.

## Measured (official `selfcheck.py`, one call at a time)

| | route | approve | grounded |
|---|---|---|---|
| dev accuracy / error | 92.5% acc | 11.9% err @0.5 | 98.3% acc |
| dev quality | macro-F1 **0.946** (majority 0.081) | cost **0.193** (always-hold 0.353) | macro-F1 **0.995** (majority 0.378) |
| dev ECE / Brier skill | 0.026 / 0.83 | 0.015 / 0.65 | 0.011 / 0.94 |
| dev error @ full → @ top-80% confident | 7.5% → 5.5% | 11.9% → 6.2% | 1.7% → 0.4% |
| hard-dev quality (perturbed text) | macro-F1 0.944 | cost 0.193 | macro-F1 0.991 |

Composite: **82.6** on dev, **82.5** on the hard perturbed dev set (previous version: 83.2 / 71.2). Clean holdout
(evaluation run, never used for choices): **79.4**; repeated CV 81.9 ± 0.4. p95 latency 2.5 ms per decision type;
12k-character inputs 19 ms. Gates pass; 61/61 malformed-input cases return schema-valid responses; outputs are
deterministic; cold start < 1 s.

## Where it fails

* **Approve has a hard ceiling.** Where the amount cap is not binding and no hold cue is present, ~14% of requests
  are still held — the account history the brief says the ticket does not contain. The model says ≈0.86 there.
  Within ±10% of each tier's cap (≈$23 / $84 / $300, learned) accuracy is only 58%; it mostly abstains.
* **Route labels are noisy** (~8%), worst for `general` (83.6% accuracy). Confident-but-"wrong" predictions on clear
  tickets are label noise.
* **Synonyms of facts in the draft.** If a draft says "money back" where the article says "refund", 19% of grounded
  answers flip: static word vectors cannot tell a synonym ("refund"/"money" 0.17) from a real change
  ("refund"/"replacement" 0.12). Fixing this needs a contextual entailment model.
* **Paraphrased cues.** Hold cues and red-flag requests in wording never seen in training lower approval only ~40% of
  the time; the exact training wording works (90–96%).
* Removing the "Priority for me is:" marker still flips 4.8% of routes (bar: 3%).

## When it abstains

Route when max-probability < 0.902; approve when the expected cost of the action taken, min(3(1−p), p), is in the
top ~16.5% (near-cap cases, no usable text); grounded when max(p, 1−p) < 0.973 (fitted on clean + perturbed text;
~10% of clean cases). Declared coverage on dev: 84.9% / 83.5% / 90.0%; on the hard set 82.6% / 83.6% / 83.0%.
Empty or unintelligible requests get near-uniform probabilities and abstain; internal errors abstain with base rates.

## Trade-offs made

* **Robustness over the last point on clean data:** on dev the new model is 0.6 below the previous one (within
  noise), and 11 points better on perturbed text.
* **Calibrated over sharp:** label smoothing, token/modality dropout and a heavy-dropout trunk keep raw logits
  near-calibrated (route temperature 1.07).
* **Approve = evidence + gate:** text evidence is trained only where the cap is not binding; a monotone gate applies
  tier/amount, so the cap cannot be learned backwards.
