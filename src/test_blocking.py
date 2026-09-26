import pandas as pd

from blocking import (
    generate_candidates,
    validate_candidate_contract,
)


def build_test_data():
    source1 = pd.DataFrame(
        [
            {
                "entity_id": "S1-1",
                "business_name": "ABC Restaurant",
                "business_address": "123 Main Street",
                "country": "US",
            },
            {
                "entity_id": "S1-2",
                "business_name": "Global Tech",
                "business_address": "45 MG Road",
                "country": "India",
            },
            {
                "entity_id": "S1-3",
                "business_name": "No Match Business",
                "business_address": "999 Unknown Road",
                "country": "France",
            },
        ]
    )

    source2 = pd.DataFrame(
        [
            {
                "entity_id": "S2-1",
                "business_name": "ABC Restaurant",
                "business_address": "123 Main St",
                "country": "US",
            },
            {
                "entity_id": "S2-2",
                "business_name": "Global Technologies",
                "business_address": "45 MG Road",
                "country": "India",
            },
            {
                "entity_id": "S2-3",
                "business_name": "ABC Restaurant",
                "business_address": "500 Other Road",
                "country": "US",
            },
        ]
    )

    source3 = pd.DataFrame(
        [
            {
                "entity_id": "S3-1",
                "business_name": "ABC Restaurant",
                "business_address": "123 Main Street",
                "country": "US",
            },
            {
                "entity_id": "S3-2",
                "business_name": "Global Tech",
                "business_address": "45 MG Road",
                "country": "India",
            },
        ]
    )

    return source1, source2, source3


def test_exact_normalized_name():
    s1, s2, s3 = build_test_data()

    candidates = generate_candidates(s1, s2, s3)

    pairs = set(
        zip(
            candidates["source1_entity_id"],
            candidates["matched_entity_id"],
        )
    )

    assert ("S1-1", "S2-1") in pairs
    assert ("S1-1", "S3-1") in pairs

    print("PASS: exact normalized name candidate found")


def test_same_country_blocking():
    s1, s2, s3 = build_test_data()

    candidates = generate_candidates(s1, s2, s3)

    pairs = set(
        zip(
            candidates["source1_entity_id"],
            candidates["matched_entity_id"],
        )
    )

    assert ("S1-1", "S2-1") in pairs
    assert ("S1-2", "S2-2") in pairs

    print("PASS: same-country candidates found")


def test_s2_and_s3_ids():
    s1, s2, s3 = build_test_data()

    candidates = generate_candidates(s1, s2, s3)

    ids = set(
        candidates["matched_entity_id"].astype(str)
    )

    assert "S2-1" in ids
    assert "S3-1" in ids

    print("PASS: S2/S3 IDs handled")


def test_duplicates_removed():
    s1, s2, s3 = build_test_data()

    candidates = generate_candidates(s1, s2, s3)

    duplicate_count = candidates.duplicated(
        subset=[
            "source1_entity_id",
            "matched_entity_id",
        ]
    ).sum()

    assert duplicate_count == 0

    print("PASS: duplicate candidates removed")


def test_no_s1_s1_candidates():
    s1, s2, s3 = build_test_data()

    candidates = generate_candidates(s1, s2, s3)

    s1_ids = set(s1["entity_id"].astype(str))

    assert not any(
        matched_id in s1_ids
        for matched_id in candidates[
            "matched_entity_id"
        ].astype(str)
    )

    print("PASS: no S1-S1 candidates")


def test_candidate_ids_valid():
    s1, s2, s3 = build_test_data()

    candidates = generate_candidates(s1, s2, s3)

    validate_candidate_contract(
        candidates,
        s1,
        s2,
        s3,
    )

    print("PASS: candidate contract valid")


def test_unknown_country_does_not_cross_match():
    s1 = pd.DataFrame(
        [
            {
                "entity_id": "S1-1",
                "business_name": "ABC Restaurant",
                "business_address": "123 Main Street",
                "country": "US",
            }
        ]
    )

    s2 = pd.DataFrame(
        [
            {
                "entity_id": "S2-1",
                "business_name": "ABC Restaurant",
                "business_address": "123 Main Street",
                "country": "India",
            }
        ]
    )

    s3 = pd.DataFrame(
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    )

    candidates = generate_candidates(s1, s2, s3)

    assert len(candidates) == 0

    print("PASS: country blocking prevents cross-country match")


if __name__ == "__main__":
    test_exact_normalized_name()
    test_same_country_blocking()
    test_s2_and_s3_ids()
    test_duplicates_removed()
    test_no_s1_s1_candidates()
    test_candidate_ids_valid()
    test_unknown_country_does_not_cross_match()

    print("\nALL BLOCKING TESTS PASSED")