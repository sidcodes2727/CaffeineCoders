# Model card — Decide (v7)

**What it is.** A typed-question decision model, `f(state, question) → typed answer + probability`, in one
forward pass. The question is an input: a question embedding is added to the pooled state and modulates every
residual block of one shared trunk (FiLM). Outputs are typed by answer shape: a choice head (route) and one binary
head shared by approve and grounded. Inputs: cleaned, spell-corrected text → word / word-pair / sentence tokens +
hashed char n-grams, draft-vs-article comparison features, frozen Model2Vec (potion-base-8M, MIT) embeddings, and
cosine similarity to cue banks mined from the labels; the approve gate reads those cue similarities directly. A learned side-opener detector finds side issues the keyword
rules do not know. Route confidence is recalibrated by a p(correct) model (margin, entropy, ensemble disagreement).
**Grounded** adds a cascade with a fine-tuned NLI cross-encoder (cross-encoder/nli-MiniLM2-L6-H768, Apache-2.0,
int8 ONNX). 10-network bagged ensemble, CPU only, no network calls, 174 MB of artifacts.

**Trained on.** `data/train.jsonl` (6,500) + one augmented copy per row (typos, dropped markers, side issues,
shuffles, unrelated KB sentences, synonyms). Dev (1,500) was never used for training, calibration or thresholds.
Calibration and abstain thresholds come from 2 × 5-fold out-of-fold predictions; every calibration option was kept
only if it won on cross-fitted predictions. The cross-encoder was fine-tuned on train minus a 15% holdout; the
cascade combiner was fitted on that holdout only.

## Measured (official `selfcheck.py`, one call at a time)

| | route | approve | grounded |
|---|---|---|---|
| dev accuracy | 92.4% | 89.5% @0.5 | 99.2% |
| dev quality | macro-F1 **0.944** (majority 0.081) | cost **0.178** (always-hold 0.353) | macro-F1 **0.997** |
| holdout approve cost | | **0.162** | |
| p95 latency | 2.5 ms | 2.6 ms | 19 ms |

Composite **83.06** on dev, **83.75** on the hard perturbed dev set, mean **81.91** over 9 dev-derived stress sets
(best of all versions compared; wins all 9 sets against v6, `tests/compare_report.json`). Repeated CV (3 seeds)
**83.79 ± 0.10**; untouched 15% holdout **81.90**. Scored latency (mean of per-decision p95) ~8–9 ms. Gates pass on every set; 61/61
malformed inputs answered schema-valid; deterministic; cold start ~1.2 s.

## Where it fails

* **Approve has a hard ceiling.** ~14% of requests below the cap with no hold cue are still held — account
  history the ticket does not contain; the cap boundary itself is fuzzy and does not depend on request type
  (checked). Within ±10% of a tier's cap (≈$23 / $84 / $300) accuracy is ~58%.
* **Route labels are noisy** (~8%), worst for `general`; checked: the wrong labels do not follow the subject,
  side clause or first sentence, and no model predicts them better than plain confidence.
* **Side issues with unfamiliar openers** ("If you get a chance, …"): 76.2 on that set vs 83.5 with known
  phrasings; the model often cannot tell which issue is primary and abstains.
* **Fact synonyms in drafts** ("money back" for "refund") still flip ~19% of grounded answers.
* **Paraphrased hold cues / red flags** lower approval only ~35–43% of the time (exact wording: 94–99%); static
  embeddings separate paraphrased cues from neutral sentences only weakly.
* Extreme inputs (12k characters) push grounded past 100 ms; real tickets are < 450 characters.

## When it abstains

Route when the recalibrated top probability is in its lowest ~16.5%; approve when the expected cost of the action
taken, min(3(1−p), p), is in the top ~18% (near-cap cases, no usable text); grounded when max(p, 1−p) < 0.970.
Declared coverage on dev: 85% / 84% / 93%. Empty or unintelligible requests get near-uniform probabilities and
abstain; internal errors abstain with base rates; if the cross-encoder is unavailable or too slow on the host,
grounded falls back to the network.

## Trade-offs made

* **Grounded cascade vs latency:** +0.4 to +3.8 on the hard / long-article / combined sets for grounded p95
  ~2 → ~25 ms (scored mean ~11 ms, still full marks). A startup guard disables it on hosts too slow for it.
* **v7 over v6:** repeated CV +1.0, holdout +0.7, all 9 stress sets better. **v7-final** (train + dev, CV
  83.34 ± 0.73) showed no measurable gain, so the train-only model is shipped and `results.json` stays held-out.
* **Calibrated over sharp**, and **approve = evidence + monotone gate** so the amount cap cannot be learned backwards.
