"""
Evaluate blocking/candidate-generation output against training ground truth.

Expected candidate columns:
    source1_entity_id
    matched_entity_id

Expected ground-truth columns:
    source1_entity_id
    matched_entity_ids

The real ground truth is expected at:
    dataset/train/train_ground_truth.tsv

This module does not build blocking rules or an ML matcher.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_CANDIDATE_COLUMNS = {"source1_entity_id", "matched_entity_id"}
REQUIRED_TRUTH_COLUMNS = {"source1_entity_id", "matched_entity_ids"}


def _normalise_id(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_matched_ids(value: object) -> list[str]:
    """Parse a ground-truth matched_entity_ids cell into unique IDs.

    Supports common representations such as:
      S2-1,S3-1
      S2-1;S3-1
      ["S2-1", "S3-1"]
      ('S2-1', 'S3-1')
      S2-1|S3-1

    Empty/null cells represent zero true matches.
    """
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]", "()"}:
        return []

    # JSON/Python-list/tuple representation.
    if text.startswith(("[", "(", "{")):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, dict):
                    parsed = list(parsed.values())
                if isinstance(parsed, (list, tuple, set)):
                    items = parsed
                    break
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
        else:
            items = [text]
    else:
        items = re.split(r"[,;|]", text)

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        item_id = _normalise_id(item)
        if item_id and item_id not in seen:
            result.append(item_id)
            seen.add(item_id)
    return result


def load_candidate_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = REQUIRED_CANDIDATE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Candidate file is missing required columns: {sorted(missing)}"
        )

    df = df[["source1_entity_id", "matched_entity_id"]].copy()
    df["source1_entity_id"] = df["source1_entity_id"].map(_normalise_id)
    df["matched_entity_id"] = df["matched_entity_id"].map(_normalise_id)
    df = df[
        (df["source1_entity_id"] != "") & (df["matched_entity_id"] != "")
    ].drop_duplicates()

    return df


def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = REQUIRED_TRUTH_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Ground-truth file is missing required columns: {sorted(missing)}"
        )

    truth: dict[str, set[str]] = {}

    for _, row in df.iterrows():
        s1 = _normalise_id(row["source1_entity_id"])
        if not s1:
            continue

        ids = set(parse_matched_ids(row["matched_entity_ids"]))
        # Merge if an S1 occurs on multiple rows.
        truth.setdefault(s1, set()).update(ids)

    return truth


def evaluate_candidates(
    candidates: pd.DataFrame, ground_truth: dict[str, set[str]]
) -> dict[str, float | int]:
    """Calculate the Phase 2 candidate-evaluation metrics."""
    candidate_map: dict[str, set[str]] = {}
    for row in candidates.itertuples(index=False):
        candidate_map.setdefault(row.source1_entity_id, set()).add(
            row.matched_entity_id
        )

    all_true_ids = {
        matched_id
        for matched_ids in ground_truth.values()
        for matched_id in matched_ids
    }

    true_s1 = {s1 for s1, matched_ids in ground_truth.items() if matched_ids}
    zero_truth_s1 = {
        s1 for s1, matched_ids in ground_truth.items() if not matched_ids
    }

    represented_s1 = set(candidate_map)
    covered_true_ids = {
        matched_id
        for s1 in true_s1
        for matched_id in ground_truth[s1]
        if matched_id in candidate_map.get(s1, set())
    }

    missing_true_ids = all_true_ids - covered_true_ids

    candidate_counts = [len(ids) for ids in candidate_map.values()]
    zero_candidate_s1 = set(ground_truth) - represented_s1

    recall = (
        len(covered_true_ids) / len(all_true_ids)
        if all_true_ids
        else 1.0
    )

    return {
        "candidate_recall": recall,
        "candidate_pairs": int(sum(candidate_counts)),
        "s1_entities_represented": int(len(represented_s1)),
        # Average is defined over S1 entities represented in the candidate table.
        "avg_candidates_per_s1": (
            sum(candidate_counts) / len(candidate_counts)
            if candidate_counts
            else 0.0
        ),
        "max_candidates_per_s1": max(candidate_counts, default=0),
        "zero_candidate_s1": int(len(zero_candidate_s1)),
        "zero_ground_truth_s1": int(len(zero_truth_s1)),
        "true_matched_ids_missing": int(len(missing_true_ids)),
        "total_true_matched_ids": int(len(all_true_ids)),
        "covered_true_matched_ids": int(len(covered_true_ids)),
    }


def evaluate_files(
    candidate_path: str | Path,
    ground_truth_path: str | Path = "dataset/train/train_ground_truth.tsv",
) -> dict[str, float | int]:
    candidates = load_candidate_table(candidate_path)
    ground_truth = load_ground_truth(ground_truth_path)
    return evaluate_candidates(candidates, ground_truth)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate pairs against training ground truth."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidate TSV containing source1_entity_id and matched_entity_id.",
    )
    parser.add_argument(
        "--ground-truth",
        default="dataset/train/train_ground_truth.tsv",
        help="Path to train_ground_truth.tsv.",
    )
    args = parser.parse_args()

    metrics = evaluate_files(args.candidates, args.ground_truth)

    print(f"Candidate recall: {metrics['candidate_recall']:.2%}")
    print(f"Candidate pairs: {metrics['candidate_pairs']}")
    print(f"S1 entities represented: {metrics['s1_entities_represented']}")
    print(f"Average candidates per S1: {metrics['avg_candidates_per_s1']:.4f}")
    print(f"Maximum candidates for one S1: {metrics['max_candidates_per_s1']}")
    print(f"Zero-candidate S1: {metrics['zero_candidate_s1']}")
    print(f"Zero-ground-truth S1: {metrics['zero_ground_truth_s1']}")
    print(
        "True matched IDs missing from candidates: "
        f"{metrics['true_matched_ids_missing']}"
    )


if __name__ == "__main__":
    main()
