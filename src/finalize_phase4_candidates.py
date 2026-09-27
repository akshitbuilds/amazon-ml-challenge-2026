from pathlib import Path
import duckdb
import time
import shutil


ROOT = Path(__file__).resolve().parents[1]

RULE_DIR = ROOT / "experiments" / "phase4_tmp" / "rules"

PARTITION_DIR = (
    ROOT
    / "experiments"
    / "phase4_tmp"
    / "hash_partitions"
)

DEDUP_DIR = (
    ROOT
    / "experiments"
    / "phase4_tmp"
    / "dedup_parts"
)

OUTPUT_DIR = ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "candidate_pairs.tsv"

DUCKDB_TMP = (
    ROOT
    / "experiments"
    / "duckdb_tmp"
    / "finalize_phase4"
)

NUM_PARTITIONS = 256


def main():

    overall_start = time.perf_counter()

    print("=" * 80)
    print("PHASE 4 - FINALIZE EXISTING B1-B7 CANDIDATES")
    print("=" * 80)

    rules = [
        "B1",
        "B5",
        "B2",
        "B3",
        "B4",
        "B6",
        "B7",
    ]

    # --------------------------------------------------------------
    # Check existing Parquet files.
    # --------------------------------------------------------------

    parquet_files = []

    print()
    print("Existing B1-B7 files:")

    for rule in rules:

        path = RULE_DIR / f"{rule}.parquet"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}"
            )

        parquet_files.append(path)

        print(
            f"  {rule}: "
            f"{path.stat().st_size / (1024 ** 3):.2f} GB"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DUCKDB_TMP.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Clean ONLY finalization intermediates.
    #
    # DO NOT touch RULE_DIR.
    # --------------------------------------------------------------

    print()
    print("Cleaning old finalization intermediates...")

    if PARTITION_DIR.exists():
        shutil.rmtree(PARTITION_DIR)

    if DEDUP_DIR.exists():
        shutil.rmtree(DEDUP_DIR)

    PARTITION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEDUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    # --------------------------------------------------------------
    # DuckDB.
    # --------------------------------------------------------------

    con = duckdb.connect()

    con.execute(
        f"""
        SET temp_directory =
        '{DUCKDB_TMP.as_posix()}'
        """
    )

    con.execute("SET threads = 4")
    con.execute(
        "SET preserve_insertion_order = false"
    )

    # --------------------------------------------------------------
    # Build a single read_parquet view over B1-B7.
    # --------------------------------------------------------------

    parquet_list = ", ".join(
        f"'{p.as_posix()}'"
        for p in parquet_files
    )

    print()
    print("[1/3] Creating combined B1-B7 candidate stream...")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW all_candidates AS

        SELECT
            CAST(source1_entity_id AS VARCHAR)
                AS source1_entity_id,

            CAST(matched_entity_id AS VARCHAR)
                AS matched_entity_id

        FROM read_parquet(
            [{parquet_list}]
        )
        """
    )

    print("  Ready.")

    # --------------------------------------------------------------
    # ONE-PASS HASH PARTITIONING.
    #
    # This is the important optimization.
    #
    # DuckDB scans all B1-B7 files once and writes rows into
    # 256 physical partition directories.
    # --------------------------------------------------------------

    print()
    print(
        "[2/3] One-pass hash partitioning into "
        f"{NUM_PARTITIONS} partitions..."
    )

    start = time.perf_counter()

    partition_output = (
        PARTITION_DIR
        .as_posix()
        .replace("'", "''")
    )

    con.execute(
        f"""
        COPY (

            SELECT
                source1_entity_id,
                matched_entity_id,

                hash(source1_entity_id)
                    % {NUM_PARTITIONS}
                    AS partition_id

            FROM all_candidates

        )

        TO '{partition_output}'

        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            PARTITION_BY (
                partition_id
            ),
            OVERWRITE_OR_IGNORE TRUE
        )
        """
    )

    partition_seconds = (
        time.perf_counter() - start
    )

    print(
        f"  Partitioning completed in "
        f"{partition_seconds / 60:.1f} minutes"
    )

    # --------------------------------------------------------------
    # Discover generated partition directories.
    # --------------------------------------------------------------

    partition_dirs = sorted(
        PARTITION_DIR.glob("partition_id=*")
    )

    print(
        f"  Physical partitions created: "
        f"{len(partition_dirs)}"
    )

    if not partition_dirs:
        raise RuntimeError(
            "No hash partitions were created."
        )

    # --------------------------------------------------------------
    # Deduplicate each physical partition.
    #
    # Each source1_entity_id belongs to exactly one partition,
    # therefore duplicate pairs cannot cross partitions.
    # --------------------------------------------------------------

    print()
    print(
        "[3/3] Deduplicating physical partitions "
        "and writing final TSV..."
    )

    start = time.perf_counter()

    # Create output header.
    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as out:

        out.write(
            "source1_entity_id\tmatched_entity_id\n"
        )

    total_pairs = 0

    for index, partition_dir in enumerate(
        partition_dirs
    ):

        if index % 8 == 0:
            print(
                f"  Deduplicating partition "
                f"{index + 1}/{len(partition_dirs)}..."
            )

        parquet_glob = (
            partition_dir
            / "*.parquet"
        )

        temp_tsv = (
            DEDUP_DIR
            / f"dedup_{index:03d}.tsv"
        )

        if temp_tsv.exists():
            temp_tsv.unlink()

        sql_glob = (
            parquet_glob
            .as_posix()
            .replace("'", "''")
        )

        sql_output = (
            temp_tsv
            .as_posix()
            .replace("'", "''")
        )

        con.execute(
            f"""
            COPY (

                SELECT DISTINCT
                    source1_entity_id,
                    matched_entity_id

                FROM read_parquet(
                    '{sql_glob}'
                )

            )

            TO '{sql_output}'

            (
                DELIMITER '\\t',
                HEADER FALSE
            )
            """
        )

        # Append deduplicated partition.
        with temp_tsv.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as inp:

            with OUTPUT_FILE.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as out:

                shutil.copyfileobj(
                    inp,
                    out,
                    length=1024 * 1024
                )

        # Count rows in this partition.
        with temp_tsv.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as inp:

            partition_count = sum(
                1 for _ in inp
            )

        total_pairs += partition_count

        temp_tsv.unlink()

    output_size = (
        OUTPUT_FILE.stat().st_size
        / (1024 ** 3)
    )

    # --------------------------------------------------------------
    # Final checks.
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("PHASE 4 COMPLETE")
    print("=" * 80)

    print(
        f"Final candidate pairs: {total_pairs:,}"
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print(
        f"Output size: {output_size:.2f} GB"
    )

    print(
        f"Total runtime: "
        f"{(time.perf_counter() - overall_start) / 60:.1f} minutes"
    )

    print()
    print(
        "Configuration: "
        "B1+B5+B2+B3+B4+B6+B7"
    )

    print("Ground truth used: NO")
    print("External data used: NO")

    con.close()


if __name__ == "__main__":
    main()