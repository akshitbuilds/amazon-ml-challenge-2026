# Phase 3 — ML Candidate Scoring Evaluation

## Scope

This framework evaluates an ML candidate scorer at **S1 entity level**. It does not train an ML model and does not modify blocking rules or Phase 2 artifacts.

Phase 2 is frozen: `src/blocking.py`, B1–B7 rules, and the Phase 2 benchmark/report are not modified.

## Data boundary

Only training/validation data may be used for threshold selection. The default ground-truth path is `dataset/train/train_ground_truth.tsv`. **Test ground truth must not be used** for this evaluation.

## Inputs

Ground truth columns:
- `source1_entity_id`
- `matched_entity_ids`

Scored candidate columns:
- `source1_entity_id`
- `matched_entity_id`
- `probability`

An optional blocking candidate table with `source1_entity_id` and `matched_entity_id` can be supplied to validate that predictions remain inside the candidate set.

## S1-level metrics

For every S1 in the union of ground truth and predictions:

- `TP = |predicted IDs ∩ ground-truth IDs|`
- `precision = TP / |predicted IDs|`
- `recall = TP / |ground-truth IDs|`
- `F0.5 = (1.25 * P * R) / (0.25 * P + R)`

A zero denominator produces `0.0` for that metric. Thus S1s with zero predictions, zero ground-truth matches, and multiple ground-truth matches are explicitly retained rather than dropped. Duplicate `(S1, matched ID)` pairs are deduplicated for set-based scoring.

The reported metrics are arithmetic macro-averages across S1 entities:
`macro_precision`, `macro_recall`, and `macro_f0.5`.

Ordinary accuracy is not used for threshold selection. Threshold selection is based on **macro F0.5**.

## Thresholds

The current Phase 3 continuation specifies:

`0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98`

The CLI accepts a custom threshold list, so the broader earlier grid can be evaluated without changing the code.

For every threshold, `experiments/phase3_results.csv` records:

`threshold, macro_precision, macro_recall, macro_f0.5, S1_count, S1_with_prediction, S1_with_zero_prediction`

Before real scored data are supplied, metric cells remain `NA`; no fake benchmark result is inserted.

## Sanity checks

The evaluator checks for:
- duplicate predicted `(S1, matched ID)` pairs;
- invalid predicted IDs that do not begin with `S2-` or `S3-`;
- predictions outside the supplied blocking candidate set.

S1 IDs missing from predictions are represented as S1s with zero predictions. A non-zero sanity-check count raises an error instead of producing a misleading evaluation.

## Reuse

Example:

```bash
python src/evaluate_phase3.py \
  --scored path/to/scored_candidates.tsv \
  --ground-truth dataset/train/train_ground_truth.tsv \
  --candidate-set path/to/candidate_pairs.tsv \
  --output experiments/phase3_results.csv
```

The framework is reusable by changing input paths and does not train the scorer.

## Self-test

Run:

```bash
python src/evaluate_phase3.py --self-test
```

The self-test uses only tiny artificial in-memory data and never reads test ground truth.
