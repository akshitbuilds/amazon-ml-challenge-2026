"""Unit tests for the Phase 2 candidate evaluator."""

from pathlib import Path
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_evaluator import evaluate_candidates


def make_truth() -> dict[str, set[str]]:
    return {
        "S1-1": {"S2-1", "S3-1"},
        "S1-2": {"S2-2"},
        "S1-3": set(),
    }


def make_candidates(include_all: bool = True) -> pd.DataFrame:
    rows = [
        ("S1-1", "S2-1"),
        ("S1-1", "S3-1"),
        ("S1-2", "S2-2"),
    ]
    if not include_all:
        rows.remove(("S1-1", "S3-1"))

    return pd.DataFrame(
        rows,
        columns=["source1_entity_id", "matched_entity_id"],
    )


def test_full_recall() -> None:
    metrics = evaluate_candidates(make_candidates(), make_truth())

    assert metrics["candidate_recall"] == 1.0
    assert metrics["candidate_pairs"] == 3
    assert metrics["s1_entities_represented"] == 2
    assert metrics["avg_candidates_per_s1"] == 1.5
    assert metrics["max_candidates_per_s1"] == 2
    assert metrics["zero_candidate_s1"] == 1
    assert metrics["zero_ground_truth_s1"] == 1
    assert metrics["true_matched_ids_missing"] == 0


def test_recall_drops_when_true_candidate_removed() -> None:
    metrics = evaluate_candidates(make_candidates(include_all=False), make_truth())

    # Two of the three true matched IDs remain covered.
    assert metrics["candidate_recall"] == 2 / 3
    assert metrics["true_matched_ids_missing"] == 1


def test_zero_ground_truth_is_not_in_recall_denominator() -> None:
    truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    candidates = pd.DataFrame(
        [("S1-1", "S2-1")],
        columns=["source1_entity_id", "matched_entity_id"],
    )

    metrics = evaluate_candidates(candidates, truth)

    assert metrics["candidate_recall"] == 1.0
    assert metrics["total_true_matched_ids"] == 1
    assert metrics["zero_ground_truth_s1"] == 1


if __name__ == "__main__":
    tests = [
        test_full_recall,
        test_recall_drops_when_true_candidate_removed,
        test_zero_ground_truth_is_not_in_recall_denominator,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("All candidate evaluator tests passed.")
