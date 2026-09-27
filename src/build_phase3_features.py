from __future__ import annotations

from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from blocking import (
    normalize_name,
    normalize_address,
    normalize_country,
    name_tokens,
    address_numbers,
)
from config import S1_TRAIN, S2_TRAIN, S3_TRAIN


CANDIDATES = Path("experiments/phase3_training_candidates.tsv")
OUTPUT = Path("experiments/phase3_features.tsv")


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value)


def suffix_stripped(name):
    text = normalize_name(name)

    suffixes = [
        "private limited",
        "pvt ltd",
        "pvt",
        "limited",
        "ltd",
        "llc",
        "incorporated",
        "inc",
        "corp",
        "corporation",
        "company",
        "co",
    ]

    for suffix in suffixes:
        if text.endswith(" " + suffix):
            text = text[: -(len(suffix) + 1)].strip()

    return text


def token_jaccard(a, b):
    a_tokens = set(name_tokens(a))
    b_tokens = set(name_tokens(b))

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def address_token_jaccard(a, b):
    a_tokens = set(normalize_address(a).split())
    b_tokens = set(normalize_address(b).split())

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def number_overlap(a, b):
    a_numbers = set(address_numbers(a))
    b_numbers = set(address_numbers(b))

    if not a_numbers or not b_numbers:
        return 0.0

    return float(bool(a_numbers & b_numbers))


def make_features(s1_row, target_row):

    s1_name = clean(s1_row.business_name)
    s2_name = clean(target_row.business_name)

    s1_address = clean(s1_row.business_address)
    s2_address = clean(target_row.business_address)

    s1_country = normalize_country(
        clean(s1_row.country)
    )

    s2_country = normalize_country(
        clean(target_row.country)
    )

    n1 = normalize_name(s1_name)
    n2 = normalize_name(s2_name)

    ns1 = suffix_stripped(s1_name)
    ns2 = suffix_stripped(s2_name)

    a1 = normalize_address(s1_address)
    a2 = normalize_address(s2_address)

    name_len_1 = len(n1)
    name_len_2 = len(n2)

    address_len_1 = len(a1)
    address_len_2 = len(a2)

    return {
        "country_exact": int(
            s1_country == s2_country
        ),

        "name_exact": int(
            n1 != "" and n1 == n2
        ),

        "name_suffix_exact": int(
            ns1 != "" and ns1 == ns2
        ),

        "name_ratio": fuzz.ratio(
            n1,
            n2,
        ) / 100.0,

        "name_token_ratio": fuzz.token_sort_ratio(
            n1,
            n2,
        ) / 100.0,

        "name_token_set_ratio": fuzz.token_set_ratio(
            n1,
            n2,
        ) / 100.0,

        "name_partial_ratio": fuzz.partial_ratio(
            n1,
            n2,
        ) / 100.0,

        "name_token_jaccard": token_jaccard(
            s1_name,
            s2_name,
        ),

        "name_length_diff": abs(
            name_len_1 - name_len_2
        ),

        "name_length_ratio": (
            min(name_len_1, name_len_2)
            / max(name_len_1, name_len_2)
            if max(name_len_1, name_len_2)
            else 0.0
        ),

        "address_exact": int(
            a1 != "" and a1 == a2
        ),

        "address_ratio": fuzz.ratio(
            a1,
            a2,
        ) / 100.0,

        "address_token_ratio": fuzz.token_sort_ratio(
            a1,
            a2,
        ) / 100.0,

        "address_token_set_ratio": fuzz.token_set_ratio(
            a1,
            a2,
        ) / 100.0,

        "address_token_jaccard": address_token_jaccard(
            s1_address,
            s2_address,
        ),

        "address_number_overlap": number_overlap(
            s1_address,
            s2_address,
        ),

        "address_length_diff": abs(
            address_len_1 - address_len_2
        ),

        "address_length_ratio": (
            min(address_len_1, address_len_2)
            / max(address_len_1, address_len_2)
            if max(address_len_1, address_len_2)
            else 0.0
        ),

        "both_name_address_exact": int(
            n1 != ""
            and n1 == n2
            and a1 != ""
            and a1 == a2
        ),
    }


def main():

    print("Loading candidate pairs...")

    candidates = pd.read_csv(
        CANDIDATES,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Candidate rows: {len(candidates):,}"
    )

    s1_ids = set(
        candidates[
            "source1_entity_id"
        ]
    )

    target_ids = set(
        candidates[
            "matched_entity_id"
        ]
    )

    print(
        f"Unique S1 IDs: {len(s1_ids):,}"
    )

    print(
        f"Unique target IDs: {len(target_ids):,}"
    )

    print("Loading S1...")

    s1 = pd.read_csv(
        S1_TRAIN,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s1 = s1[
        s1["entity_id"].isin(s1_ids)
    ].copy()

    print("Loading S2...")

    s2 = pd.read_csv(
        S2_TRAIN,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s2 = s2[
        s2["entity_id"].isin(target_ids)
    ].copy()

    print(
        f"S2 required rows: {len(s2):,}"
    )

    print("Loading S3...")

    s3 = pd.read_csv(
        S3_TRAIN,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s3 = s3[
        s3["entity_id"].isin(target_ids)
    ].copy()

    print(
        f"S3 required rows: {len(s3):,}"
    )

    target = pd.concat(
        [
            s2,
            s3,
        ],
        ignore_index=True,
    ).drop_duplicates(
        subset=["entity_id"]
    )

    s1_lookup = (
        s1.set_index("entity_id")
    )

    target_lookup = (
        target.set_index("entity_id")
    )

    print(
        "\nComputing fuzzy features..."
    )

    feature_rows = []

    total = len(candidates)

    for i, row in enumerate(
        candidates.itertuples(index=False),
        1,
    ):

        s1_id = row.source1_entity_id
        target_id = row.matched_entity_id

        s1_row = s1_lookup.loc[s1_id]
        target_row = target_lookup.loc[target_id]

        features = make_features(
            s1_row,
            target_row,
        )

        features[
            "source1_entity_id"
        ] = s1_id

        features[
            "matched_entity_id"
        ] = target_id

        features[
            "label"
        ] = int(row.label)

        feature_rows.append(features)

        if i % 10000 == 0:

            print(
                f"  processed "
                f"{i:,}/{total:,}",
                flush=True,
            )

    features_df = pd.DataFrame(
        feature_rows
    )

    first_columns = [
        "source1_entity_id",
        "matched_entity_id",
        "label",
    ]

    feature_columns = [
        c
        for c in features_df.columns
        if c not in first_columns
    ]

    features_df = features_df[
        first_columns + feature_columns
    ]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_df.to_csv(
        OUTPUT,
        sep="\t",
        index=False,
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )

    print(
        f"Rows: {len(features_df):,}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        "\nFeature columns:"
    )

    for column in feature_columns:
        print(
            f"  {column}"
        )


if __name__ == "__main__":
    main()