from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_phase3_features import make_features


EXPECTED_FEATURES = [
    "country_exact",
    "name_exact",
    "name_suffix_exact",
    "name_ratio",
    "name_token_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "name_length_diff",
    "name_length_ratio",
    "address_exact",
    "address_ratio",
    "address_token_ratio",
    "address_token_set_ratio",
    "address_token_jaccard",
    "address_number_overlap",
    "address_length_diff",
    "address_length_ratio",
    "both_name_address_exact",
]

DEFAULT_PARTITION = (
    ROOT
    / "experiments"
    / "phase4_tmp"
    / "hash_partitions"
    / "partition_id=104"
    / "data_140.parquet"
)

S1_FILE = ROOT / "data" / "tsv" / "test" / "test_source1.tsv"
S2_FILE = ROOT / "data" / "tsv" / "test" / "test_source2.tsv"
S3_FILE = ROOT / "data" / "tsv" / "test" / "test_source3.tsv"

MODEL_FILE = ROOT / "models" / "entity_matcher.joblib"


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def read_partition(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(
            "SELECT source1_entity_id, matched_entity_id "
            "FROM read_parquet(?)",
            [str(path)],
        ).fetchdf()
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition",
        default=str(DEFAULT_PARTITION),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
    )
    args = parser.parse_args()

    partition = Path(args.partition)

    print("=" * 70)
    print("PHASE 4 - ONE PARTITION PROTOTYPE")
    print("=" * 70)
    print(f"Partition: {partition}")
    print(f"Threshold: {args.threshold}")
    print()

    if not partition.exists():
        raise FileNotFoundError(partition)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(MODEL_FILE)

    # ------------------------------------------------------------
    # Model verification
    # ------------------------------------------------------------
    print("Loading model...")
    model_bundle = joblib.load(MODEL_FILE)

    required_keys = {"model", "feature_columns", "best_threshold"}
    missing = required_keys - set(model_bundle.keys())
    if missing:
        raise ValueError(f"Model bundle missing keys: {sorted(missing)}")

    model = model_bundle["model"]
    model_features = list(model_bundle["feature_columns"])
    stored_threshold = model_bundle["best_threshold"]

    print(f"Model type: {type(model).__name__}")
    print(f"Stored best_threshold: {stored_threshold}")
    print(f"Model feature count: {len(model_features)}")

    if model_features != EXPECTED_FEATURES:
        raise ValueError(
            "MODEL FEATURE ORDER MISMATCH\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Model:    {model_features}"
        )

    print("Feature order verification: PASS")
    print()

    # ------------------------------------------------------------
    # Read existing partition ONLY
    # ------------------------------------------------------------
    print("Reading existing Parquet partition...")
    total_start = time.perf_counter()

    candidates = read_partition(partition)

    partition_rows = len(candidates)

    print(f"Partition rows: {partition_rows:,}")

    if candidates["source1_entity_id"].duplicated().any():
        print(
            "Note: partition contains multiple candidates per S1, "
            "as expected."
        )

    # ------------------------------------------------------------
    # Load only required test records
    # ------------------------------------------------------------
    s1_ids = set(candidates["source1_entity_id"].astype(str))
    target_ids = set(candidates["matched_entity_id"].astype(str))

    print(f"Unique S1 IDs required: {len(s1_ids):,}")
    print(f"Unique target IDs required: {len(target_ids):,}")

    print("\nLoading test Source 1...")
    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    s1 = s1[s1["entity_id"].isin(s1_ids)].copy()

    print(f"S1 rows loaded: {len(s1):,}")

    print("Loading test Source 2...")
    s2 = pd.read_csv(
        S2_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    s2 = s2[s2["entity_id"].isin(target_ids)].copy()

    print(f"S2 rows loaded: {len(s2):,}")

    print("Loading test Source 3...")
    s3 = pd.read_csv(
        S3_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    s3 = s3[s3["entity_id"].isin(target_ids)].copy()

    print(f"S3 rows loaded: {len(s3):,}")

    target = pd.concat([s2, s3], ignore_index=True).drop_duplicates(
        subset=["entity_id"]
    )

    s1_lookup = s1.set_index("entity_id")
    target_lookup = target.set_index("entity_id")

    missing_s1 = s1_ids - set(s1_lookup.index.astype(str))
    missing_target = target_ids - set(target_lookup.index.astype(str))

    if missing_s1:
        raise ValueError(
            f"Missing {len(missing_s1)} S1 records required by partition."
        )

    if missing_target:
        raise ValueError(
            f"Missing {len(missing_target)} target records required by partition."
        )

    # ------------------------------------------------------------
    # Exact Phase 3 feature generation
    # ------------------------------------------------------------
    print("\nGenerating exact Phase 3 features...")
    feature_start = time.perf_counter()

    feature_rows = []

    for row in candidates.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        target_id = str(row.matched_entity_id)

        features = make_features(
            s1_lookup.loc[s1_id],
            target_lookup.loc[target_id],
        )

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    feature_time = time.perf_counter() - feature_start

    if list(features_df.columns) != EXPECTED_FEATURES:
        raise ValueError(
            "GENERATED FEATURE ORDER MISMATCH\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Generated: {list(features_df.columns)}"
        )

    print(f"Feature-generation time: {feature_time:.4f} sec")
    print(f"Generated feature rows: {len(features_df):,}")
    print("Generated feature order verification: PASS")

    # ------------------------------------------------------------
    # Model scoring
    # ------------------------------------------------------------
    print("\nScoring with exact Phase 3 model...")
    score_start = time.perf_counter()

    probabilities = model.predict_proba(
        features_df[EXPECTED_FEATURES]
    )[:, 1]

    scoring_time = time.perf_counter() - score_start

    predicted_matches = int(
        (probabilities >= args.threshold).sum()
    )

    total_time = time.perf_counter() - total_start

    memory = rss_mb()

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("PROTOTYPE RESULTS")
    print("=" * 70)
    print(f"Partition row count:       {partition_rows:,}")
    print(f"Feature-generation time:   {feature_time:.4f} sec")
    print(f"Model-scoring time:        {scoring_time:.4f} sec")
    print(f"Total runtime:             {total_time:.4f} sec")
    print(f"Threshold:                 {args.threshold}")
    print(f"Predicted matches:         {predicted_matches:,}")

    if memory is not None:
        print(f"Process RSS:               {memory:.2f} MB")
    else:
        print("Process RSS:               unavailable (psutil not installed)")

    print()
    print("NO OUTPUT FILE WAS WRITTEN.")
    print("Existing Parquet partition was read-only.")
    print("Candidate generation was not run.")


if __name__ == "__main__":
    main()
