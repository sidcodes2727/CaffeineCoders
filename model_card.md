# Model card — Decide

**What it is.** One small shared network (token-embedding bag → residual trunk, 128-d, ~1.9 M parameters per
network, 5-network bagged ensemble, 34 MB total) with three heads: `route` (5-way), `approve` (yes/no with a 3:1
cost-aware threshold) and `grounded` (yes/no). Numpy-only CPU inference; no pretrained weights, no network calls.
Sharing: all three decisions read the same trunk output `h` (see README, `model.py::_trunk`).

**Trained on.** `data/train.jsonl` only (6,500 labelled requests). The dev split (1,500) was held out for every
number below. Calibrators and abstention thresholds come from 5-fold out-of-fold predictions on train.

**Features.** Route: the *primary* request sentence (the part after "Priority for me is:" etc.; decoy clauses ignored).
Approve: normalised body sentences (history cues), intent n-grams, tier and amount. Grounded: draft-vs-article diff after
canonicalising numbers and durations.

## Measured on dev (via `selfcheck.py`, 4,500 one-at-a-time calls)

| | route | approve | grounded |
|---|---|---|---|
| accuracy / error | 92.5% acc | 12.8% err @0.5 | 98.1% acc |
| quality metric | macro-F1 **0.941** (majority 0.085) | cost **0.219** (always-hold 0.355) | macro-F1 **0.996** (majority 0.384) |
| ECE | 0.019 | 0.015 | 0.011 |
| Brier skill | 0.83 | 0.61 | 0.93 |
| error @ full → @ top-80% confident | 7.5% → 5.7% | 12.8% → 7.3% | 1.9% → 0.3% |
| declared coverage | 87.1% | 82.7% | 86.4% |

Quality metrics are on answered cases only. Latency: p95 **1.0 ms** per decision type (p50 ≈ 1.0 ms), warm, one
request at a time, laptop CPU. Self-check composite on dev: **81.8 / 100** (quality 0.82, calibration 0.86,
latency 1.00, abstention 0.50). Gates: schema 4500/4500 valid, latency and reachability pass.

## Where it fails

* **Approve has a hard ceiling.** In the region where the amount cap is not binding and no hold cue is present, ~14% of
  requests are still held — consistent with the account history the brief says the ticket does not contain. The model
  outputs ≈0.86 there rather than pretending to be sure; this is why approve scores far below the other two.
  Near each tier's cap (≈$23 free / ≈$84 pro / ≈$300 enterprise, learned from data) probabilities are mid-range and the
  model usually holds or abstains.
* **Route labels are noisy** (~8% of tickets carry a queue a careful reader would not choose). Confident-but-"wrong"
  predictions on clear tickets are label noise, not a bug; confidence is capped near 0.9 for that reason.
* **Grounded relies on structure.** It is near-perfect on this templated KB/draft style because it compares canonicalised
  facts (numbers, durations, scope words). Free-form paraphrase outside that vocabulary, or a fact it has never seen
  contradicted, would degrade it; it has no semantic encoder.
* Typos and phrasing outside the training vocabulary fall back to unknown tokens (a few common typos are normalised).

## When it abstains

`abstain=true` for the lowest-confidence ≈16.5% of cases per decision (thresholds from out-of-fold quantiles, target
declared coverage 83.5% to stay above the 80% scoring target): route when max-probability < 0.877; grounded when
max(p,1−p) < 0.960; approve when the expected cost of the action taken, min(3(1−p), p), exceeds 0.547 — i.e. near-cap
cases and cases with no usable text. Empty or unintelligible requests get near-uniform route probabilities and abstain.
Abstention changes only what is scored for quality; the abstention score reads the probabilities, which are calibrated.

## Trade-off made

Calibrated probabilities over raw accuracy: label smoothing, token dropout and a heavy-dropout trunk cost ≈0 accuracy but
brought route ECE from ≈0.05 (T≈4 post-hoc) to 0.019 without post-hoc distortion. The approve head separates *text
evidence* (trained only where the amount cap is not binding, so memorising noise there cannot leak into the cap) from the
tier/amount gate.
