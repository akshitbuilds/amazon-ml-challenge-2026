from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import pandas as pd


ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"


def normalize_text(value: object) -> str:
    """
    Basic normalization used only for blocking.

    Keeps alphanumeric characters and converts whitespace
    to a single space.
    """
    if value is None:
        return ""

    text = str(value).lower().strip()

    if not text:
        return ""

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(value: object) -> str:
    """Normalize a business name for exact/token blocking."""
    return normalize_text(value)


def normalize_address(value: object) -> str:
    """Normalize an address for blocking."""
    return normalize_text(value)


def normalize_country(value: object) -> str:
    """Normalize country while keeping it open-set."""
    return normalize_text(value)


def name_tokens(value: object, min_length: int = 3) -> list[str]:
    """
    Return meaningful name tokens.

    Very short tokens are ignored because they create
    excessively large candidate buckets.
    """
    normalized = normalize_name(value)

    if not normalized:
        return []

    return [
        token
        for token in normalized.split()
        if len(token) >= min_length
    ]


def address_numbers(value: object) -> list[str]:
    """
    Extract numeric components from an address.
    """
    normalized = normalize_address(value)

    if not normalized:
        return []

    return re.findall(r"\d+", normalized)


def build_exact_name_index(source: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """
    Build:

        (country, normalized_name) -> entity IDs

    Only non-empty normalized names are indexed.
    """
    index: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in source.itertuples(index=False):
        entity_id = str(getattr(row, ID_COL))
        country = normalize_country(getattr(row, COUNTRY_COL))
        name = normalize_name(getattr(row, NAME_COL))

        if not country or not name:
            continue

        index[(country, name)].append(entity_id)

    return dict(index)


def build_name_token_index(
    source: pd.DataFrame,
    min_token_length: int = 3,
) -> dict[tuple[str, str], list[str]]:
    """
    Build:

        (country, name_token) -> entity IDs

    This creates broader candidates than exact-name blocking.
    """
    index: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in source.itertuples(index=False):
        entity_id = str(getattr(row, ID_COL))
        country = normalize_country(getattr(row, COUNTRY_COL))

        if not country:
            continue

        tokens = set(
            name_tokens(
                getattr(row, NAME_COL),
                min_length=min_token_length,
            )
        )

        for token in tokens:
            index[(country, token)].append(entity_id)

    return dict(index)


def build_address_number_index(
    source: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
    """
    Build:

        (country, address_number) -> entity IDs

    Used as a supporting blocking signal.
    """
    index: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in source.itertuples(index=False):
        entity_id = str(getattr(row, ID_COL))
        country = normalize_country(getattr(row, COUNTRY_COL))

        if not country:
            continue

        numbers = set(address_numbers(getattr(row, ADDRESS_COL)))

        for number in numbers:
            index[(country, number)].append(entity_id)

    return dict(index)


def generate_source_candidates(
    source1: pd.DataFrame,
    source_other: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate candidate pairs between Source 1 and one other source.

    Blocking passes:
      1. country + exact normalized name
      2. country + significant normalized name token
      3. country + address number + name token

    Returns:
        source1_entity_id
        matched_entity_id
    """
    exact_name_index = build_exact_name_index(source_other)
    token_index = build_name_token_index(source_other)
    address_number_index = build_address_number_index(source_other)

    candidate_pairs: set[tuple[str, str]] = set()

    for row in source1.itertuples(index=False):
        s1_id = str(getattr(row, ID_COL))
        country = normalize_country(getattr(row, COUNTRY_COL))

        if not country:
            continue

        name = getattr(row, NAME_COL)
        address = getattr(row, ADDRESS_COL)

        # ---------------------------------------------------------
        # Blocking pass 1:
        # country + exact normalized name
        # ---------------------------------------------------------
        normalized_name = normalize_name(name)

        if normalized_name:
            for matched_id in exact_name_index.get(
                (country, normalized_name),
                [],
            ):
                candidate_pairs.add((s1_id, matched_id))

        # ---------------------------------------------------------
        # Blocking pass 2:
        # country + significant name token
        # ---------------------------------------------------------
        tokens = set(name_tokens(name))

        for token in tokens:
            for matched_id in token_index.get(
                (country, token),
                [],
            ):
                candidate_pairs.add((s1_id, matched_id))

        # ---------------------------------------------------------
        # Blocking pass 3:
        # country + address number + name token
        # ---------------------------------------------------------
        numbers = set(address_numbers(address))

        if numbers and tokens:
            for number in numbers:
                address_candidates = address_number_index.get(
                    (country, number),
                    [],
                )

                for matched_id in address_candidates:
                    # Candidate is further supported by the
                    # fact that the target name shares at least
                    # one indexed token.
                    target_row_ids = token_index

                    for token in tokens:
                        if matched_id in target_row_ids.get(
                            (country, token),
                            [],
                        ):
                            candidate_pairs.add(
                                (s1_id, matched_id)
                            )
                            break

    result = pd.DataFrame(
        sorted(candidate_pairs),
        columns=[
            "source1_entity_id",
            "matched_entity_id",
        ],
    )

    return result


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate the final union of S1-S2 and S1-S3 candidate pairs.

    The returned table is the candidate set that can later be
    written to candidate_pairs.tsv immediately before ML scoring.
    """
    s2_candidates = generate_source_candidates(
        source1,
        source2,
    )

    s3_candidates = generate_source_candidates(
        source1,
        source3,
    )

    candidates = pd.concat(
        [
            s2_candidates,
            s3_candidates,
        ],
        ignore_index=True,
    )

    candidates = candidates.drop_duplicates(
        subset=[
            "source1_entity_id",
            "matched_entity_id",
        ]
    )

    candidates = candidates.sort_values(
        [
            "source1_entity_id",
            "matched_entity_id",
        ]
    ).reset_index(drop=True)

    return candidates


def validate_candidate_contract(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> None:
    """
    Validate basic candidate-pair constraints.

    Raises AssertionError if the candidate table violates
    the Phase 2 contract.
    """
    required_columns = {
        "source1_entity_id",
        "matched_entity_id",
    }

    assert required_columns.issubset(
        candidates.columns
    ), "Candidate columns are missing."

    s1_ids = set(
        source1[ID_COL].astype(str)
    )

    target_ids = set(
        pd.concat(
            [
                source2[ID_COL],
                source3[ID_COL],
            ],
            ignore_index=True,
        ).astype(str)
    )

    candidate_s1_ids = set(
        candidates["source1_entity_id"].astype(str)
    )

    candidate_target_ids = set(
        candidates["matched_entity_id"].astype(str)
    )

    assert candidate_s1_ids.issubset(
        s1_ids
    ), "Candidate contains unknown Source 1 IDs."

    assert candidate_target_ids.issubset(
        target_ids
    ), "Candidate contains IDs outside Source 2/3."

    assert not candidates.duplicated(
        subset=[
            "source1_entity_id",
            "matched_entity_id",
        ]
    ).any(), "Duplicate candidate pairs found."

    assert not (
        candidates["source1_entity_id"].astype(str)
        == candidates["matched_entity_id"].astype(str)
    ).any(), "S1-S1 candidate detected."


if __name__ == "__main__":
    print("blocking.py loaded successfully.")
    print("Candidate generation foundation is ready.")