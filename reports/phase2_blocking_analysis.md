# Phase 2 Blocking Analysis

## Development Sample

- S1 sample size: 10,000
- Random seed: 42
- S1 rows with ground-truth matches: 9,462
- S1 rows with zero ground-truth matches: 538
- Total true matched IDs in sample: 34,723

Ground truth was used only for evaluation after candidate generation.
It was not used to construct blocking keys.

## Blocking Passes Tested

### B1
Country + exact normalized business name.

### B2
Country + significant normalized name token. Tokens shorter than 4 characters or appearing more than 100000 times were ignored.

### B3
Country + extracted address number + longest normalized name token of length >= 4.

### B4
Country + compact address signature consisting of the first three normalized address tokens.

## Benchmark Results

| Experiment | Candidate pairs | Recall | Avg candidates/S1 |
|---|---:|---:|---:|
| B1 | 96,210 | 22.2936% | 9.62 |
| B2 | 25,247,947 | 63.2491% | 2561.94 |
| B3 | 112,840 | 43.5504% | 13.64 |
| B4 | 692,706 | 33.4908% | 69.27 |
| B1+B2 | 25,249,224 | 63.6207% | 2535.06 |
| B1+B2+B3 | 25,249,224 | 63.6207% | 2535.06 |
| B1+B2+B3+B4 | 25,934,247 | 76.7964% | 2595.24 |

## Main Observations

- **B1** produced 96,210 candidate pairs with 22.2936% candidate recall and 9.62 average candidates/S1.
- **B2** produced 25,247,947 candidate pairs with 63.2491% candidate recall and 2561.94 average candidates/S1.
- **B3** produced 112,840 candidate pairs with 43.5504% candidate recall and 13.64 average candidates/S1.
- **B4** produced 692,706 candidate pairs with 33.4908% candidate recall and 69.27 average candidates/S1.

## Union Results

- **B1+B2** produced 25,249,224 candidate pairs with 63.6207% recall and 2535.06 average candidates/S1.
- **B1+B2+B3** produced 25,249,224 candidate pairs with 63.6207% recall and 2535.06 average candidates/S1.
- **B1+B2+B3+B4** produced 25,934,247 candidate pairs with 76.7964% recall and 2595.24 average candidates/S1.

## Configuration to Investigate Further

The measured trade-off to investigate further is **B1+B2+B3+B4 (B1 UNION B2 UNION B3 UNION B4)**.

In this 50,000-S1 development sample it produced 25,934,247 candidate pairs, 76.7964% measured candidate recall, and 2595.24 average candidates/S1.

This is a measured development-sample trade-off, not a final production conclusion. It should be validated on a larger sample before integration.

## Leakage Check

Ground truth was not used to create B1, B2, B3, or B4. Ground truth was used only after candidate generation to calculate recall.

## Notes

The experiments use the training TSV files and a deterministic 50,000-row S1 development sample. Original business fields are preserved in the source views; normalized fields are derived fields.
