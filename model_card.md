# Model card — Decide

**What it is.** A typed-question decision model, `f(state, question) → typed answer + probability`, in one forward
pass. The question is an input: a question embedding is added to the pooled state and modulates every residual block of
one shared trunk (FiLM). Outputs are typed by answer shape: a choice head (route, 5 queue options) and one binary head
shared by both yes/no questions (approve, grounded). ~1.9 M parameters per network, 10-network bagged ensemble, 69 MB.
Numpy-only CPU inference; no pretrained weights, no network calls. See README and `model.py::_answer`.

**Trained on.** `data/train.jsonl` only (6,500 labelled requests). The dev split (1,500) was held out from training,
calibration and threshold selection. Calibrators and abstention thresholds come from 2 × 5-fold out-of-fold
predictions on train. Dev was used to compare a handful of model variants, so the dev score is mildly optimistic.

**Features.** Route: the *primary* request sentence (the part after "Priority for me is:" etc.; decoy clauses ignored).
Approve: normalised body sentences (history cues), intent n-grams, then a tier/amount policy gate. Grounded:
draft-vs-article diff after canonicalising numbers and durations (word numbers, compounds, units).

## Measured on dev (via `selfcheck.py`, 4,500 one-at-a-time calls)

| | route | approve | grounded |
|---|---|---|---|
| accuracy / error | 92.5% acc | 12.5% err @0.5 | 98.7% acc |
| quality metric | macro-F1 **0.943** (majority 0.082) | cost **0.217** (always-hold 0.362) | macro-F1 **0.998** (majority 0.371) |
| ECE | 0.022 | 0.015 | 0.005 |
| Brier skill | 0.83 | 0.62 | 0.95 |
| error @ full → @ top-80% confident | 7.5% → 5.3% | 12.5% → 6.8% | 1.3% → 0.1% |
| declared coverage | 85.3% | 86.5% | 85.2% |

Quality metrics are on answered cases only. Latency: p95 **1.5 ms** per decision type, warm, one request at a time,
laptop CPU. Self-check composite on dev: **83.2 / 100** (quality 0.83, calibration 0.87, latency 1.00, abstention
0.56). Gates: schema 4500/4500 valid, latency and reachability pass.

## Where it fails

* **Approve has a hard ceiling.** Where the amount cap is not binding and no hold cue is present, ~14% of requests are
  still held — consistent with the account history the brief says the ticket does not contain. The model outputs ≈0.86
  there rather than pretending to be sure. Near each tier's cap (≈$23 free / ≈$84 pro / ≈$300 enterprise, learned from
  data) probabilities are mid-range and the model usually holds or abstains.
* **Route labels are noisy** (~8% of tickets carry a queue a careful reader would not choose). Confident-but-"wrong"
  predictions on clear tickets are label noise; confidence stays near 0.9 for that reason.
* **Grounded relies on structure.** It compares canonicalised facts (numbers, durations, scope words), which suits this
  templated KB/draft style. Free-form paraphrase outside that vocabulary would degrade it; it has no semantic encoder.
* Typos and phrasing outside the training vocabulary fall back to unknown tokens (a few common typos are normalised).

## When it abstains

`abstain=true` for roughly the lowest-confidence 15% of cases per decision (thresholds from out-of-fold quantiles,
target declared coverage 83.5% to stay above the 80% scoring target): route when max-probability < 0.900; grounded when
max(p,1−p) < 0.988; approve when the expected cost of the action taken, min(3(1−p), p), exceeds 0.579 — near-cap cases
and cases with no usable text. Empty or unintelligible requests get near-uniform route probabilities and abstain.

## Trade-offs made

* **Calibrated over sharp:** label smoothing, token dropout and a heavy-dropout trunk cost ≈0 accuracy but keep raw
  logits near-calibrated (route temperature 1.13 instead of ≈4).
* **Approve split into evidence + gate:** the text-evidence logit is trained only where the amount cap is not binding,
  so memorising noise there cannot leak into the cap; a monotone gate then applies tier/amount.
* **Typed-question net over per-decision heads:** same accuracy at equal ensemble size (82.2 vs 81.6 on dev), but a new
  decision costs one question embedding instead of a new head.
