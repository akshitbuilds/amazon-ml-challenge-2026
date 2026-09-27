from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from blocking import (
    address_numbers,
    normalize_country,
    normalize_name,
    name_tokens,
    validate_candidate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "tsv" / "train"

S1_FILE = DATA_DIR / "train_source1.tsv"
S2_FILE = DATA_DIR / "train_source2.tsv"
S3_FILE = DATA_DIR / "train_source3.tsv"
GT_FILE = DATA_DIR / "train_ground_truth.tsv"

OUTPUT_FILE = ROOT / "experiments" / "phase3_training_candidates.tsv"
REPORT_FILE = ROOT / "reports" / "phase3_candidate_labels.md"

SAMPLE_SIZE = 10_000
RANDOM_SEED = 42

# Training-time controls.
# These prevent extremely common tokens from exploding the candidate set.
MAX_TOKEN_FREQ = 300
MAX_CANDIDATES_PER_S1 = 500


def build_gt_pairs(
    gt: pd.DataFrame,
    sampled_ids: set[str],
) -> pd.DataFrame:
    """Extract ground-truth S1 -> target pairs for sampled S1 IDs."""

    pairs: set[tuple[str, str]] = set()

    for row in gt.itertuples(index=False):
        s1_id = str(row.source1_entity_id)

        if s1_id not in sampled_ids:
            continue

        matched = str(row.matched_entity_ids).strip()

        if matched:
            for target_id in matched.split(","):
                target_id = target_id.strip()

                if target_id:
                    pairs.add((s1_id, target_id))

    return pd.DataFrame(
        sorted(pairs),
        columns=[
            "source1_entity_id",
            "matched_entity_id",
        ],
    )


def build_compact_indexes(
    source: pd.DataFrame,
    relevant_exact_keys: set[tuple[str, str]],
    relevant_token_keys: set[tuple[str, str]],
    relevant_number_keys: set[tuple[str, str]],
):
    """
    Build only target indexes that can actually be queried by the
    sampled S1 rows.

    This avoids constructing huge indexes for irrelevant names/tokens.
    """

    exact_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    token_index: dict[tuple[str, str], set[str]] = defaultdict(set)
    number_index: dict[tuple[str, str], set[str]] = defaultdict(set)

    # ---------------------------------------------------------------
    # First pass:
    #   - exact relevant names
    #   - frequency of relevant name tokens
    #   - relevant address numbers
    # ---------------------------------------------------------------

    token_frequency: dict[tuple[str, str], int] = defaultdict(int)

    for row in source.itertuples(index=False):
        entity_id = str(row.entity_id)
        country = normalize_country(row.country)

        if not country:
            continue

        # Exact-name index: only keep keys that sampled S1 can query.
        normalized_name = normalize_name(row.business_name)

        if normalized_name:
            exact_key = (country, normalized_name)

            if exact_key in relevant_exact_keys:
                exact_index[exact_key].append(entity_id)

        # Token frequency: only count tokens that sampled S1 can query.
        for token in set(name_tokens(row.business_name)):
            token_key = (country, token)

            if token_key in relevant_token_keys:
                token_frequency[token_key] += 1

        # Address-number index: only keep numbers sampled S1 can query.
        for number in set(address_numbers(row.business_address)):
            number_key = (country, number)

            if number_key in relevant_number_keys:
                number_index[number_key].add(entity_id)

    # ---------------------------------------------------------------
    # Only retain name-token buckets that are not too common.
    # ---------------------------------------------------------------

    allowed_token_keys = {
        key
        for key, frequency in token_frequency.items()
        if 0 < frequency <= MAX_TOKEN_FREQ
    }

    # ---------------------------------------------------------------
    # Second pass:
    # Build only allowed token buckets.
    # ---------------------------------------------------------------

    if allowed_token_keys:
        for row in source.itertuples(index=False):
            entity_id = str(row.entity_id)
            country = normalize_country(row.country)

            if not country:
                continue

            for token in set(name_tokens(row.business_name)):
                token_key = (country, token)

                if token_key in allowed_token_keys:
                    token_index[token_key].add(entity_id)

    return exact_index, token_index, number_index, token_frequency


def generate_candidates_phase3(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """
    Controlled Phase-3 training candidate generation.

    Uses the same three Phase-2 blocking concepts:

      1. country + exact normalized name
      2. country + significant normalized name token
      3. country + address number + name token

    The indexes are restricted to keys that actually occur in the
    sampled S1 rows. This avoids constructing enormous irrelevant
    Python dictionaries.

    Training-time controls:
      - MAX_TOKEN_FREQ = 300
      - MAX_CANDIDATES_PER_S1 = 500
    """

    print()
    print("Preparing relevant blocking keys from sampled S1...")

    # ---------------------------------------------------------------
    # Build the exact/token/address keys that the sampled S1 rows
    # can actually request.
    # ---------------------------------------------------------------

    relevant_exact_keys: set[tuple[str, str]] = set()
    relevant_token_keys: set[tuple[str, str]] = set()
    relevant_number_keys: set[tuple[str, str]] = set()

    for row in source1.itertuples(index=False):
        country = normalize_country(row.country)

        if not country:
            continue

        normalized_name = normalize_name(row.business_name)

        if normalized_name:
            relevant_exact_keys.add(
                (country, normalized_name)
            )

        for token in set(name_tokens(row.business_name)):
            relevant_token_keys.add(
                (country, token)
            )

        for number in set(address_numbers(row.business_address)):
            relevant_number_keys.add(
                (country, number)
            )

    print(
        f"Relevant exact-name keys:       "
        f"{len(relevant_exact_keys):,}"
    )

    print(
        f"Relevant name-token keys:       "
        f"{len(relevant_token_keys):,}"
    )

    print(
        f"Relevant address-number keys:   "
        f"{len(relevant_number_keys):,}"
    )

    # ---------------------------------------------------------------
    # Build compact indexes for S2 and S3.
    # ---------------------------------------------------------------

    print()
    print("Indexing S2 using only relevant keys...")

    (
        s2_exact,
        s2_tokens,
        s2_numbers,
        s2_token_frequency,
    ) = build_compact_indexes(
        source2,
        relevant_exact_keys,
        relevant_token_keys,
        relevant_number_keys,
    )

    print(
        f"S2 exact buckets:               "
        f"{len(s2_exact):,}"
    )

    print(
        f"S2 allowed token buckets:       "
        f"{len(s2_tokens):,}"
    )

    print(
        f"S2 address-number buckets:      "
        f"{len(s2_numbers):,}"
    )

    print()
    print("Indexing S3 using only relevant keys...")

    (
        s3_exact,
        s3_tokens,
        s3_numbers,
        s3_token_frequency,
    ) = build_compact_indexes(
        source3,
        relevant_exact_keys,
        relevant_token_keys,
        relevant_number_keys,
    )

    print(
        f"S3 exact buckets:               "
        f"{len(s3_exact):,}"
    )

    print(
        f"S3 allowed token buckets:       "
        f"{len(s3_tokens):,}"
    )

    print(
        f"S3 address-number buckets:      "
        f"{len(s3_numbers):,}"
    )

    # ---------------------------------------------------------------
    # Candidate generation.
    # ---------------------------------------------------------------

    print()
    print("Generating controlled candidate pairs...")

    all_pairs: set[tuple[str, str]] = set()

    def generate_for_target(
        row,
        exact_index,
        token_index,
        number_index,
    ) -> set[tuple[str, str]]:

        s1_id = str(row.entity_id)
        country = normalize_country(row.country)

        if not country:
            return set()

        normalized_name = normalize_name(row.business_name)
        tokens = set(name_tokens(row.business_name))
        numbers = set(address_numbers(row.business_address))

        local_pairs: set[tuple[str, str]] = set()

        # -----------------------------------------------------------
        # Rule 1:
        # country + exact normalized name
        #
        # Do this first because exact matches are highly informative.
        # -----------------------------------------------------------

        if normalized_name:
            exact_candidates = exact_index.get(
                (country, normalized_name),
                (),
            )

            for target_id in exact_candidates:
                local_pairs.add(
                    (s1_id, target_id)
                )

        # -----------------------------------------------------------
        # Rule 3:
        # country + address number + name token
        #
        # Do this before the broad token rule so useful address/name
        # combinations are not crowded out by common token matches.
        # -----------------------------------------------------------

        if numbers and tokens:
            for number in numbers:
                number_candidates = number_index.get(
                    (country, number),
                    (),
                )

                for target_id in number_candidates:
                    target_matches_token = False

                    for token in tokens:
                        bucket = token_index.get(
                            (country, token),
                            (),
                        )

                        if target_id in bucket:
                            target_matches_token = True
                            break

                    if target_matches_token:
                        local_pairs.add(
                            (s1_id, target_id)
                        )

                    if (
                        len(local_pairs)
                        >= MAX_CANDIDATES_PER_S1
                    ):
                        return local_pairs

        # -----------------------------------------------------------
        # Rule 2:
        # country + significant normalized name token
        #
        # This is the broadest rule, so apply the per-S1 cap here.
        # -----------------------------------------------------------

        if len(local_pairs) < MAX_CANDIDATES_PER_S1:
            for token in tokens:
                token_candidates = token_index.get(
                    (country, token),
                    (),
                )

                for target_id in token_candidates:
                    local_pairs.add(
                        (s1_id, target_id)
                    )

                    if (
                        len(local_pairs)
                        >= MAX_CANDIDATES_PER_S1
                    ):
                        return local_pairs

        return local_pairs

    # ---------------------------------------------------------------
    # Process S1 one row at a time.
    # ---------------------------------------------------------------

    total_rows = len(source1)

    for row_number, row in enumerate(
        source1.itertuples(index=False),
        start=1,
    ):

        s2_pairs = generate_for_target(
            row,
            s2_exact,
            s2_tokens,
            s2_numbers,
        )

        s3_pairs = generate_for_target(
            row,
            s3_exact,
            s3_tokens,
            s3_numbers,
        )

        all_pairs.update(s2_pairs)
        all_pairs.update(s3_pairs)

        if row_number % 250 == 0:
            print(
                f"Processed {row_number:,}/{total_rows:,} S1 rows | "
                f"candidate pairs={len(all_pairs):,}"
            )

    print()
    print(
        f"Finished candidate generation: "
        f"{len(all_pairs):,} candidate pairs"
    )

    return pd.DataFrame(
        sorted(all_pairs),
        columns=[
            "source1_entity_id",
            "matched_entity_id",
        ],
    )


def main():
    print("=" * 70)
    print("PHASE 3 - TRAINING CANDIDATE DATASET")
    print("=" * 70)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading training data...")

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s2 = pd.read_csv(
        S2_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s3 = pd.read_csv(
        S3_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print()
    print(
        f"Selecting deterministic "
        f"{SAMPLE_SIZE:,}-row S1 sample..."
    )

    s1_sample = (
        s1.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_SEED,
            replace=False,
        )
        .reset_index(drop=True)
    )

    sampled_ids = set(
        s1_sample["entity_id"].astype(str)
    )

    gt_pairs = build_gt_pairs(
        gt,
        sampled_ids,
    )

    gt_key = set(
        zip(
            gt_pairs["source1_entity_id"],
            gt_pairs["matched_entity_id"],
        )
    )

    print(
        f"Sampled S1 count: "
        f"{len(s1_sample):,}"
    )

    print(
        "S1s with at least one GT match: "
        f"{gt_pairs['source1_entity_id'].nunique():,}"
    )

    print(
        f"Total GT matched IDs: "
        f"{len(gt_pairs):,}"
    )

    print()
    print(
        "Generating candidates using controlled "
        "Phase-2 blocking rules..."
    )

    candidates = generate_candidates_phase3(
        s1_sample,
        s2,
        s3,
    )

    print()
    print("Validating candidate contract...")

    validate_candidate_contract(
        candidates,
        s1_sample,
        s2,
        s3,
    )

    candidates["source1_entity_id"] = (
        candidates["source1_entity_id"].astype(str)
    )

    candidates["matched_entity_id"] = (
        candidates["matched_entity_id"].astype(str)
    )

    candidate_keys = list(
        zip(
            candidates["source1_entity_id"],
            candidates["matched_entity_id"],
        )
    )

    candidates["label"] = [
        1 if key in gt_key else 0
        for key in candidate_keys
    ]

    # ---------------------------------------------------------------
    # Validation.
    # ---------------------------------------------------------------

    assert not candidates.duplicated(
        [
            "source1_entity_id",
            "matched_entity_id",
        ]
    ).any()

    assert set(
        candidates["label"].unique()
    ).issubset({0, 1})

    target_ids = (
        set(s2["entity_id"].astype(str))
        |
        set(s3["entity_id"].astype(str))
    )

    assert set(
        candidates["matched_entity_id"]
    ).issubset(target_ids)

    positives = candidates[
        candidates["label"] == 1
    ]

    positive_keys = set(
        zip(
            positives["source1_entity_id"],
            positives["matched_entity_id"],
        )
    )

    assert positive_keys.issubset(gt_key)

    # ---------------------------------------------------------------
    # Statistics.
    # ---------------------------------------------------------------

    total_candidates = len(candidates)

    positive_count = int(
        candidates["label"].sum()
    )

    negative_count = (
        total_candidates - positive_count
    )

    positive_rate = (
        positive_count / total_candidates
        if total_candidates
        else 0.0
    )

    s1_with_positive = (
        positives["source1_entity_id"].nunique()
    )

    recovered_true_ids = len(
        positive_keys
    )

    total_true_ids = len(gt_key)

    candidate_recall = (
        recovered_true_ids / total_true_ids
        if total_true_ids
        else 0.0
    )

    gt_s1_ids = set(
        gt_pairs["source1_entity_id"]
    )

    recovered_s1_ids = set(
        positives["source1_entity_id"]
    )

    completely_missing = len(
        gt_s1_ids - recovered_s1_ids
    )

    candidates = (
        candidates[
            [
                "source1_entity_id",
                "matched_entity_id",
                "label",
            ]
        ]
        .sort_values(
            [
                "source1_entity_id",
                "matched_entity_id",
            ]
        )
        .reset_index(drop=True)
    )

    candidates.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    # ---------------------------------------------------------------
    # Report.
    # ---------------------------------------------------------------

    report = f"""# Phase 3 — Candidate Pair Labels

## Configuration

- Sampled S1 count: {len(s1_sample):,}
- Random seed: {RANDOM_SEED}
- Candidate generation: controlled Phase-2 blocking rules
- Target sources: S2 and S3
- Maximum token frequency: {MAX_TOKEN_FREQ}
- Maximum candidates per S1: {MAX_CANDIDATES_PER_S1}
- Test data used: No
- ML model trained: No
- Random candidate pairs added: No
- Production `src/blocking.py` modified: No

## Required Statistics

| Statistic | Value |
|---|---:|
| Sampled S1 count | {len(s1_sample):,} |
| S1s with at least one GT match | {gt_pairs['source1_entity_id'].nunique():,} |
| Total candidate pairs | {total_candidates:,} |
| Positive candidate pairs | {positive_count:,} |
| Negative candidate pairs | {negative_count:,} |
| Positive rate | {positive_rate:.4%} |
| S1s with at least one positive candidate | {s1_with_positive:,} |
| S1s whose true matches are completely missing from candidates | {completely_missing:,} |
| Candidate recall | {candidate_recall:.4%} |

## Recall

Candidate recall = true GT matched IDs recovered inside candidate set / total GT matched IDs for sampled S1s.

- Total GT matched IDs: {total_true_ids:,}
- Recovered true matched IDs: {recovered_true_ids:,}
- Candidate recall: {candidate_recall:.4%}

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

"""

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Final console summary.
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)

    print(
        f"Sampled S1 count:                         "
        f"{len(s1_sample):,}"
    )

    print(
        f"S1s with at least one GT match:          "
        f"{gt_pairs['source1_entity_id'].nunique():,}"
    )

    print(
        f"Total candidate pairs:                    "
        f"{total_candidates:,}"
    )

    print(
        f"Positive candidate pairs:                 "
        f"{positive_count:,}"
    )

    print(
        f"Negative candidate pairs:                 "
        f"{negative_count:,}"
    )

    print(
        f"Positive rate:                             "
        f"{positive_rate:.4%}"
    )

    print(
        f"S1s with at least one positive candidate: "
        f"{s1_with_positive:,}"
    )

    print(
        f"S1s whose true matches are completely missing: "
        f"{completely_missing:,}"
    )

    print(
        f"Candidate recall:                          "
        f"{candidate_recall:.4%}"
    )

    print()
    print(f"Created: {OUTPUT_FILE}")
    print(f"Created: {REPORT_FILE}")


if __name__ == "__main__":
    main()