# Phase 2 Blocking Analysis

## Development Sample

- S1 sample size: 10,000
- Random seed: 42
- S1 rows with ground-truth matches: 9,462
- S1 rows with zero ground-truth matches: 538
- Total true matched IDs in sample: 34,723

Ground truth was used only for evaluation.
It was not used to create blocking keys.

## Blocking Passes Tested

### B1
Country + exact normalized business name.

### B2
Country + longest normalized name token with length >= 4.

### B3
Country + extracted address number + significant normalized name token.

### B4
Country + compact address signature consisting of the first three normalized address tokens.

## Benchmark Results

| Experiment | Candidate pairs | Recall | Avg candidates/S1 |
|---|---:|---:|---:|
| B1 | 4,574,358,265 | 36.6673% | 457435.83 |
| B2 | 0 | 0.0000% | 0.00 |
| B3 | 0 | 0.0000% | 0.00 |
| B4 | 18,379,287,595 | 56.7808% | 1837928.76 |
| B1+B2 | 4,574,358,265 | 36.6673% | 457435.83 |
| B1+B2+B3 | 4,574,358,265 | 36.6673% | 457435.83 |
| B1+B2+B3+B4 | 21,393,298,553 | 72.0272% | 2139329.86 |

## Main Observations

- **B1** produced 4,574,358,265 candidate pairs with 36.6673% recall and 457435.83 average candidates/S1.
- **B2** produced 0 candidate pairs with 0.0000% recall and 0.00 average candidates/S1.
- **B3** produced 0 candidate pairs with 0.0000% recall and 0.00 average candidates/S1.
- **B4** produced 18,379,287,595 candidate pairs with 56.7808% recall and 1837928.76 average candidates/S1.

## Union Results

- **B1+B2** produced 4,574,358,265 candidate pairs with 36.6673% recall and 457435.83 average candidates/S1.
- **B1+B2+B3** produced 4,574,358,265 candidate pairs with 36.6673% recall and 457435.83 average candidates/S1.
- **B1+B2+B3+B4** produced 21,393,298,553 candidate pairs with 72.0272% recall and 2139329.86 average candidates/S1.

## Configuration to Investigate Further

The measured trade-off to investigate further is **B1+B2+B3+B4 (B1 UNION B2 UNION B3 UNION B4)**.

In this 10,000-S1 development sample it produced 21,393,298,553 candidate pairs, 72.0272% measured candidate recall, and 2139329.86 average candidates/S1.

This is a measured development-sample trade-off, not a final production conclusion. It should be validated on a larger sample before integration.

## Leakage Check

Ground truth was not used to create B1, B2, B3, or B4. Ground truth was used only to calculate recall.
