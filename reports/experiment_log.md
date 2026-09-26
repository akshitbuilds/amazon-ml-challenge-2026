# Experiment Log

## Phase 1 — Experiment Tracking

### Scope
Phase 1 is limited to setting up experiment tracking and documenting validation requirements.
No dataset-based experiments are being run in this phase.

### Tracking file
Experiment results are recorded in:

`experiments/results.csv`

The CSV is initialized with a standard schema for recording future experiments:

- `experiment_id` — unique experiment identifier
- `date` — date of the experiment
- `stage` — pipeline stage being tested
- `method` — method/model/approach used
- `parameters` — important configuration or hyperparameters
- `metric` — evaluation metric used
- `result` — measured result
- `artifact` — relevant output/artifact path
- `status` — planned, completed, failed, or skipped
- `notes` — observations and relevant context

### Phase 1 status
No model or dataset experiment has been executed yet. Therefore, no experimental metric or performance result is recorded.

### Phase 2 handoff
When the team begins dataset-based work, each experiment should be added as a separate row in `experiments/results.csv`, with enough information to reproduce and compare the run.
