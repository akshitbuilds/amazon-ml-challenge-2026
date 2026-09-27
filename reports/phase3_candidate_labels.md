# Phase 3 — Candidate Pair Labels

## Configuration

- Sampled S1 count: 10,000
- Random seed: 42
- Candidate generation: controlled Phase-2 blocking rules
- Target sources: S2 and S3
- Maximum token frequency: 300
- Maximum candidates per S1: 500
- Test data used: No
- ML model trained: No
- Random candidate pairs added: No
- Production `src/blocking.py` modified: No

## Required Statistics

| Statistic | Value |
|---|---:|
| Sampled S1 count | 10,000 |
| S1s with at least one GT match | 9,428 |
| Total candidate pairs | 914,172 |
| Positive candidate pairs | 17,440 |
| Negative candidate pairs | 896,732 |
| Positive rate | 1.9077% |
| S1s with at least one positive candidate | 6,887 |
| S1s whose true matches are completely missing from candidates | 2,541 |
| Candidate recall | 50.1611% |

## Recall

Candidate recall = true GT matched IDs recovered inside candidate set / total GT matched IDs for sampled S1s.

- Total GT matched IDs: 34,768
- Recovered true matched IDs: 17,440
- Candidate recall: 50.1611%

## Validation

- Duplicate candidate pairs: none
- Candidate direction: S1 → S2/S3 only
- Labels: 0/1 only
- Every positive exists in training ground truth: yes
- No random candidate pairs added: yes
- Test data used: no
- Production `src/blocking.py` modified: no

## Output

`experiments/phase3_training_candidates.tsv`

This is a controlled Phase-3 training candidate set. Token-frequency and per-S1 candidate caps are used to keep training-data generation computationally manageable.

