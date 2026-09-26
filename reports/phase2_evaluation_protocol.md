# Phase 2 Evaluation Protocol

## Purpose

This document defines how candidate-generation/blocking experiments are evaluated.

Member 3 owns the evaluation framework. Blocking rules and the ML matcher are outside this scope.

## Ground truth

The evaluator expects the official training ground truth at:

`dataset/train/train_ground_truth.tsv`

Ground-truth columns:

- `source1_entity_id`
- `matched_entity_ids`

Candidate tables must contain:

- `source1_entity_id`
- `matched_entity_id`

The official ground truth is not modified or fabricated by this framework.

## Candidate recall

Candidate recall measures how many true matched IDs are present in the candidate table:

`candidate recall = covered true matched IDs / total true matched IDs`

Only S1 entities with at least one true match contribute true matched IDs to this denominator.

An S1 entity with zero ground-truth matches is **not** treated as a false-negative candidate-recall case. Zero-ground-truth S1 entities are reported separately.

## Candidate count

`candidate_count` is the number of unique `(source1_entity_id, matched_entity_id)` candidate pairs after duplicate pairs are removed.

## S1 entities represented

The count of distinct `source1_entity_id` values appearing in the candidate table.

## Average candidates per S1

For the Phase 2 blocking comparison, average candidates per S1 is:

`unique candidate pairs / S1 entities represented`

This excludes S1 entities that have no candidate rows. The zero-candidate count is reported separately.

## Maximum candidates per S1

The largest number of unique candidate IDs associated with any single S1 entity in the candidate table.

## Zero-candidate count

`zero_candidate_s1` is the number of S1 entities present in the ground truth that have no candidate rows.

The evaluator also reports `zero_ground_truth_s1`: S1 entities whose ground-truth matched-ID set is empty.

## Missing true matched IDs

The evaluator reports the number of unique true matched IDs that are not present among the candidates for their corresponding S1 entity.

## Experiment naming convention

Use:

`EXP-<four-digit-number>`

Examples:

- `EXP-0002` — Phase 2 initialization
- `EXP-0003` — first blocking experiment
- `EXP-0004` — next blocking experiment

The `blocking_version` field should identify the exact blocking configuration/version being evaluated.

## Recording rule

Every blocking experiment must be recorded in `experiments/phase2_results.csv`.

Do not overwrite prior experiment rows. Add a new row for each experiment.

## Phase 2 scope

This framework does not:

- design blocking rules,
- modify Member 2's blocking benchmark,
- train an ML matcher,
- modify the official ground truth,
- download or fabricate a full dataset,
- claim full-dataset evaluation results before the official local resources are available.
