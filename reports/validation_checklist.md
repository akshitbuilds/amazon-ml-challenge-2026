# Validation Checklist

## Phase 1 — Validator Understanding

Phase 1 does not require the official challenge validator or dataset to be downloaded.
The validator is challenge-provided and will be used later with the official challenge resources.

### Known submission outputs

- [ ] `output/candidate_pairs.tsv` is the candidate-pair output passed to the matching stage.
- [ ] `output/matching_results.tsv` is the final matching output used for leaderboard scoring.

### Validator requirements currently documented

Based on the team's supplied validator documentation/instructions:

- [ ] Required S1 rows are present.
- [ ] Duplicate IDs are checked.
- [ ] S2/S3 prefixes are respected.
- [ ] Submission-format rules are satisfied.
- [ ] Candidate and matching output paths are supplied in the expected form.
- [ ] The validator is run against the official `dataset/test` resources when those resources are available.

### Example validator invocation

The team instructions show the validator being run in this form:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

This command is documentation only for Phase 1. The official validator and dataset are not being downloaded or modified during this phase.

### Phase 1 restrictions

- [x] Do not download a random validator from GitHub.
- [x] Do not create a replacement validator.
- [x] Do not download the official dataset for Phase 1.
- [x] Do not modify the official validator.
- [x] Do not fabricate validation results without the challenge resources.

### Phase 2 validation

Once the official challenge resources are provided by the team, the checklist should be executed against the actual files and the results documented.
