from __future__ import annotations

import shutil
import time
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = ROOT / "data" / "tsv" / "test"

S1_FILE = TEST_DIR / "test_source1.tsv"
S2_FILE = TEST_DIR / "test_source2.tsv"
S3_FILE = TEST_DIR / "test_source3.tsv"

OUTPUT_DIR = ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "candidate_pairs.tsv"

PHASE4_DIR = ROOT / "experiments" / "phase4_tmp"
PARQUET_DIR = PHASE4_DIR / "rules"
PART_DIR = PHASE4_DIR / "partitions"
DUCKDB_TMP = ROOT / "experiments" / "duckdb_tmp" / "phase4"

NUM_PARTITIONS = 256

TOKEN_MIN_LEN = 4
TOKEN_MAX_FREQ = 2000

NAME_PREFIX_LEN = 6
PREFIX_MIN_LEN = 4
PREFIX_MAX_FREQ = 2000

LEGAL_SUFFIX_TOKENS = [
    "llc",
    "inc",
    "ltd",
    "ltda",
    "pvt",
    "private",
    "limited",
    "corp",
    "corporation",
    "co",
    "plc",
    "llp",
    "gmbh",
    "srl",
    "sa",
    "ag",
    "group",
    "holdings",
    "holding",
    "enterprise",
    "enterprises",
    "company",
    "companies",
    "pc",
    "pllc",
    "pty",
]


def normalized_text_sql(column: str) -> str:
    return f"""
        lower(
            trim(
                regexp_replace(
                    regexp_replace(
                        coalesce(CAST({column} AS VARCHAR), ''),
                        '[^\\p{{L}}\\p{{N}}\\s]',
                        ' ',
                        'g'
                    ),
                    '\\s+',
                    ' ',
                    'g'
                )
            )
        )
    """


def build_source_sql(csv_path: Path) -> str:

    name_expr = normalized_text_sql("business_name")
    address_expr = normalized_text_sql("business_address")

    suffix_sql = (
        "["
        + ",".join(f"'{x}'" for x in LEGAL_SUFFIX_TOKENS)
        + "]"
    )

    return f"""
        SELECT
            *,
            substr(
                replace(name_stripped, ' ', ''),
                1,
                {NAME_PREFIX_LEN}
            ) AS name_prefix
        FROM (
            SELECT
                entity_id,
                business_name,
                business_address,
                country,

                {name_expr} AS name_norm,

                {address_expr} AS address_norm,

                lower(
                    trim(
                        coalesce(
                            CAST(country AS VARCHAR),
                            ''
                        )
                    )
                ) AS country_norm,

                regexp_extract(
                    {address_expr},
                    '\\d+'
                ) AS address_number,

                concat_ws(
                    ' ',
                    split_part({address_expr}, ' ', 1),
                    split_part({address_expr}, ' ', 2),
                    split_part({address_expr}, ' ', 3)
                ) AS address_signature,

                trim(
                    coalesce(
                        array_to_string(
                            list_filter(
                                string_split(
                                    {name_expr},
                                    ' '
                                ),
                                x ->
                                    NOT list_contains(
                                        {suffix_sql},
                                        x
                                    )
                            ),
                            ' '
                        ),
                        ''
                    )
                ) AS name_stripped

            FROM read_csv(
                '{csv_path.as_posix()}',
                sep='\\t',
                header=true,
                all_varchar=true
            )
        ) base
    """


def elapsed(start):
    return time.perf_counter() - start


def write_rule_parquet(con, rule_name: str, query: str):

    output = PARQUET_DIR / f"{rule_name}.parquet"

    if output.exists():
        print(f"    {rule_name}: existing parquet found, replacing...")
        output.unlink()

    sql_path = output.as_posix().replace("'", "''")

    con.execute(
        f"""
        COPY (
            {query}
        )
        TO '{sql_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    size_gb = output.stat().st_size / (1024 ** 3)

    print(
        f"    {rule_name}: {size_gb:.2f} GB parquet"
    )

    return output


def main():

    overall_start = time.perf_counter()

    print("=" * 80)
    print("PHASE 4 - FULL TEST CANDIDATE GENERATION")
    print("=" * 80)

    print()
    print("Configuration:")
    print("  B1 + B5 + B2 + B3 + B4 + B6 + B7")
    print(f"  Partitions: {NUM_PARTITIONS}")
    print()

    for path in [S1_FILE, S2_FILE, S3_FILE]:

        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PHASE4_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PARQUET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DUCKDB_TMP.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Clean previous failed Phase-4 intermediate files.
    # --------------------------------------------------------------

    print("[0/7] Cleaning previous Phase-4 intermediates...")

    for path in PARQUET_DIR.glob("*.parquet"):
        path.unlink()

    for path in PART_DIR.glob("part_*.tsv"):
        path.unlink()

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    # --------------------------------------------------------------
    # DuckDB
    # --------------------------------------------------------------

    con = duckdb.connect()

    con.execute(
        f"SET temp_directory = '{DUCKDB_TMP.as_posix()}'"
    )

    con.execute("SET threads = 4")
    con.execute("SET preserve_insertion_order = false")

    # --------------------------------------------------------------
    # 1. Normalize sources.
    # --------------------------------------------------------------

    print()
    print("[1/7] Loading and normalizing full test sources...")

    start = time.perf_counter()

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s1_norm AS
        {build_source_sql(S1_FILE)}
        """
    )

    print(
        "  S1 rows:",
        con.execute(
            "SELECT COUNT(*) FROM s1_norm"
        ).fetchone()[0]
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s2_norm AS
        {build_source_sql(S2_FILE)}
        """
    )

    print(
        "  S2 rows:",
        con.execute(
            "SELECT COUNT(*) FROM s2_norm"
        ).fetchone()[0]
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s3_norm AS
        {build_source_sql(S3_FILE)}
        """
    )

    print(
        "  S3 rows:",
        con.execute(
            "SELECT COUNT(*) FROM s3_norm"
        ).fetchone()[0]
    )

    print(f"  Completed in {elapsed(start):.1f}s")

    # --------------------------------------------------------------
    # 2. Token frequencies and token rankings.
    # --------------------------------------------------------------

    print()
    print("[2/7] Building token-frequency tables...")

    start = time.perf_counter()

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE token_freq AS

        SELECT
            token,
            COUNT(*) AS freq

        FROM (

            SELECT u.token
            FROM s1_norm s,
            unnest(
                string_split(s.name_norm, ' ')
            ) AS u(token)

            WHERE length(u.token) >= {TOKEN_MIN_LEN}
              AND u.token <> ''

            UNION ALL

            SELECT u.token
            FROM s2_norm s,
            unnest(
                string_split(s.name_norm, ' ')
            ) AS u(token)

            WHERE length(u.token) >= {TOKEN_MIN_LEN}
              AND u.token <> ''

            UNION ALL

            SELECT u.token
            FROM s3_norm s,
            unnest(
                string_split(s.name_norm, ' ')
            ) AS u(token)

            WHERE length(u.token) >= {TOKEN_MIN_LEN}
              AND u.token <> ''

        ) tokens

        GROUP BY token
        """
    )

    print(
        "  Token rows:",
        con.execute(
            "SELECT COUNT(*) FROM token_freq"
        ).fetchone()[0]
    )

    for label in ["s1", "s2", "s3"]:

        print(f"  Ranking {label.upper()} tokens...")

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {label}_tokens_ranked AS

            SELECT
                s.entity_id,
                s.country_norm,
                u.token,
                f.freq,

                ROW_NUMBER() OVER (
                    PARTITION BY s.entity_id

                    ORDER BY
                        f.freq ASC,
                        length(u.token) DESC,
                        u.token
                ) AS rn

            FROM {label}_norm s

            CROSS JOIN LATERAL (
                SELECT DISTINCT token

                FROM unnest(
                    string_split(s.name_norm, ' ')
                ) AS x(token)

                WHERE length(token) >= {TOKEN_MIN_LEN}
                  AND token <> ''
            ) u

            INNER JOIN token_freq f
                ON f.token = u.token

            WHERE f.freq <= {TOKEN_MAX_FREQ}
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {label}_tokens AS

            SELECT
                entity_id,
                country_norm,
                token

            FROM {label}_tokens_ranked

            WHERE rn = 1
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {label}_tokens2 AS

            SELECT
                entity_id,
                country_norm,
                token

            FROM {label}_tokens_ranked

            WHERE rn = 2
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {label}_signature AS

            SELECT
                entity_id,
                country_norm,

                string_agg(
                    token,
                    '_' ORDER BY token
                ) AS sig

            FROM {label}_tokens_ranked

            WHERE rn <= 2

            GROUP BY
                entity_id,
                country_norm
            """
        )

    print(f"  Completed in {elapsed(start):.1f}s")

    # --------------------------------------------------------------
    # 3. B7 prefix frequencies.
    # --------------------------------------------------------------

    print()
    print("[3/7] Building B7 prefix-frequency tables...")

    start = time.perf_counter()

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE prefix_freq AS

        SELECT
            country_norm,
            name_prefix,
            COUNT(*) AS freq

        FROM (

            SELECT
                country_norm,
                name_prefix

            FROM s1_norm

            WHERE length(name_prefix) >= {PREFIX_MIN_LEN}

            UNION ALL

            SELECT
                country_norm,
                name_prefix

            FROM s2_norm

            WHERE length(name_prefix) >= {PREFIX_MIN_LEN}

            UNION ALL

            SELECT
                country_norm,
                name_prefix

            FROM s3_norm

            WHERE length(name_prefix) >= {PREFIX_MIN_LEN}

        ) p

        GROUP BY
            country_norm,
            name_prefix
        """
    )

    for label in ["s1", "s2", "s3"]:

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {label}_prefix AS

            SELECT
                s.entity_id,
                s.country_norm,
                s.name_prefix

            FROM {label}_norm s

            INNER JOIN prefix_freq f
                ON f.country_norm = s.country_norm
               AND f.name_prefix = s.name_prefix

            WHERE length(s.name_prefix) >= {PREFIX_MIN_LEN}
              AND f.freq <= {PREFIX_MAX_FREQ}
            """
        )

    print(
        "  Qualifying S1 prefixes:",
        con.execute(
            "SELECT COUNT(*) FROM s1_prefix"
        ).fetchone()[0]
    )

    print(f"  Completed in {elapsed(start):.1f}s")

    # --------------------------------------------------------------
    # 4. Generate B1-B7 and immediately write each to Parquet.
    # --------------------------------------------------------------

    print()
    print("[4/7] Generating B1-B7 and writing disk-backed Parquet...")

    start = time.perf_counter()

    rules = {}

    rules["B1"] = """
        SELECT
            s1.entity_id AS source1_entity_id,
            s2.entity_id AS matched_entity_id

        FROM s1_norm s1

        INNER JOIN s2_norm s2
            ON s2.country_norm = s1.country_norm
           AND s2.name_norm = s1.name_norm

        WHERE s1.name_norm <> ''

        UNION ALL

        SELECT
            s1.entity_id AS source1_entity_id,
            s3.entity_id AS matched_entity_id

        FROM s1_norm s1

        INNER JOIN s3_norm s3
            ON s3.country_norm = s1.country_norm
           AND s3.name_norm = s1.name_norm

        WHERE s1.name_norm <> ''
    """

    rules["B5"] = """
        SELECT
            s1.entity_id AS source1_entity_id,
            s2.entity_id AS matched_entity_id

        FROM s1_norm s1

        INNER JOIN s2_norm s2
            ON s2.country_norm = s1.country_norm
           AND s2.name_stripped = s1.name_stripped

        WHERE s1.name_stripped <> ''

        UNION ALL

        SELECT
            s1.entity_id AS source1_entity_id,
            s3.entity_id AS matched_entity_id

        FROM s1_norm s1

        INNER JOIN s3_norm s3
            ON s3.country_norm = s1.country_norm
           AND s3.name_stripped = s1.name_stripped

        WHERE s1.name_stripped <> ''
    """

    rules["B2"] = """
        SELECT
            s1.entity_id AS source1_entity_id,
            s2.entity_id AS matched_entity_id

        FROM s1_tokens s1

        INNER JOIN s2_tokens s2
            ON s1.country_norm = s2.country_norm
           AND s1.token = s2.token

        UNION ALL

        SELECT
            s1.entity_id AS source1_entity_id,
            s3.entity_id AS matched_entity_id

        FROM s1_tokens s1

        INNER JOIN s3_tokens s3
            ON s1.country_norm = s3.country_norm
           AND s1.token = s3.token
    """

    rules["B3"] = """
        SELECT
            t1.entity_id AS source1_entity_id,
            a2.entity_id AS matched_entity_id

        FROM s1_tokens2 t1

        INNER JOIN s1_norm a1
            ON a1.entity_id = t1.entity_id
           AND a1.address_number <> ''

        INNER JOIN s2_norm a2
            ON a2.country_norm = t1.country_norm
           AND a2.address_number = a1.address_number

        INNER JOIN s2_tokens2 t2
            ON t2.entity_id = a2.entity_id
           AND t2.token = t1.token

        UNION ALL

        SELECT
            t1.entity_id AS source1_entity_id,
            a3.entity_id AS matched_entity_id

        FROM s1_tokens2 t1

        INNER JOIN s1_norm a1
            ON a1.entity_id = t1.entity_id
           AND a1.address_number <> ''

        INNER JOIN s3_norm a3
            ON a3.country_norm = t1.country_norm
           AND a3.address_number = a1.address_number

        INNER JOIN s3_tokens2 t3
            ON t3.entity_id = a3.entity_id
           AND t3.token = t1.token
    """

    rules["B4"] = """
        SELECT
            s1.entity_id AS source1_entity_id,
            s2.entity_id AS matched_entity_id

        FROM s1_norm s1

        INNER JOIN s2_norm s2
            ON s1.country_norm = s2.country_norm
           AND s1.address_signature = s2.address_signature
           AND s1.address_signature <> ''

        UNION ALL

        SELECT
            s1.entity_id AS source1_entity_id,
            s3.entity_id AS matched_entity_id

        FROM s1_norm s1

        INNER JOIN s3_norm s3
            ON s1.country_norm = s3.country_norm
           AND s1.address_signature = s3.address_signature
           AND s1.address_signature <> ''
    """

    rules["B6"] = """
        SELECT
            s1.entity_id AS source1_entity_id,
            s2.entity_id AS matched_entity_id

        FROM s1_signature s1

        INNER JOIN s2_signature s2
            ON s1.country_norm = s2.country_norm
           AND s1.sig = s2.sig

        UNION ALL

        SELECT
            s1.entity_id AS source1_entity_id,
            s3.entity_id AS matched_entity_id

        FROM s1_signature s1

        INNER JOIN s3_signature s3
            ON s1.country_norm = s3.country_norm
           AND s1.sig = s3.sig
    """

    rules["B7"] = """
        SELECT
            s1.entity_id AS source1_entity_id,
            s2.entity_id AS matched_entity_id

        FROM s1_prefix s1

        INNER JOIN s2_prefix s2
            ON s1.country_norm = s2.country_norm
           AND s1.name_prefix = s2.name_prefix

        UNION ALL

        SELECT
            s1.entity_id AS source1_entity_id,
            s3.entity_id AS matched_entity_id

        FROM s1_prefix s1

        INNER JOIN s3_prefix s3
            ON s1.country_norm = s3.country_norm
           AND s1.name_prefix = s3.name_prefix
    """

    rule_order = [
        "B1",
        "B5",
        "B2",
        "B3",
        "B4",
        "B6",
        "B7",
    ]

    rule_files = {}

    for rule in rule_order:

        print(f"  {rule}: generating...")

        rule_files[rule] = write_rule_parquet(
            con,
            rule,
            rules[rule],
        )

    print(f"  Completed in {elapsed(start):.1f}s")

    # --------------------------------------------------------------
    # Release the giant source/rule temporary tables before
    # partitioned deduplication.
    # --------------------------------------------------------------

    print()
    print("[5/7] Releasing large intermediate tables...")

    for table in [
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
    ]:

        try:
            con.execute(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass

    # --------------------------------------------------------------
    # 5. Partitioned deduplication.
    #
    # Every candidate pair for the same S1 goes to the same
    # partition. Therefore DISTINCT inside each partition is
    # globally sufficient.
    # --------------------------------------------------------------

    print()
    print(
        f"[6/7] Deduplicating in {NUM_PARTITIONS} disk-backed partitions..."
    )

    start = time.perf_counter()

    parquet_glob = (
        (PARQUET_DIR / "*.parquet")
        .as_posix()
        .replace("'", "''")
    )

    partition_files = []

    for partition in range(NUM_PARTITIONS):

        if partition % 16 == 0:
            print(
                f"  Partition {partition}/{NUM_PARTITIONS}..."
            )

        part_file = PART_DIR / f"part_{partition:03d}.tsv"

        if part_file.exists():
            part_file.unlink()

        part_sql = part_file.as_posix().replace("'", "''")

        con.execute(
            f"""
            COPY (
                SELECT DISTINCT
                    source1_entity_id,
                    matched_entity_id

                FROM read_parquet(
                    '{parquet_glob}'
                )

                WHERE
                    hash(
                        CAST(source1_entity_id AS VARCHAR)
                    ) % {NUM_PARTITIONS} = {partition}
            )

            TO '{part_sql}'

            (
                DELIMITER '\\t',
                HEADER TRUE
            )
            """
        )

        partition_files.append(part_file)

    print(
        f"  Partition dedup completed in "
        f"{elapsed(start) / 60:.1f} minutes"
    )

    # --------------------------------------------------------------
    # 6. Concatenate already-deduplicated partitions.
    #
    # Partitions are disjoint, so no additional DISTINCT is needed.
    # --------------------------------------------------------------

    print()
    print("[7/7] Creating final candidate_pairs.tsv...")

    start = time.perf_counter()

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as out:

        out.write(
            "source1_entity_id\tmatched_entity_id\n"
        )

        for index, part_file in enumerate(partition_files):

            if index % 32 == 0:
                print(
                    f"  Combining partition "
                    f"{index}/{NUM_PARTITIONS}..."
                )

            with part_file.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as inp:

                # Skip partition header.
                inp.readline()

                shutil.copyfileobj(
                    inp,
                    out,
                    length=1024 * 1024,
                )

    output_size_gb = (
        OUTPUT_FILE.stat().st_size
        / (1024 ** 3)
    )

    print(
        f"  Final output size: {output_size_gb:.2f} GB"
    )

    # --------------------------------------------------------------
    # Verification using the partition files.
    # --------------------------------------------------------------

    total_pairs = 0

    for part_file in partition_files:

        with part_file.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as inp:

            # Header
            inp.readline()

            for _ in inp:
                total_pairs += 1

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
        f"Output size: {output_size_gb:.2f} GB"
    )

    print(
        f"Total runtime: "
        f"{elapsed(overall_start) / 60:.1f} minutes"
    )   

    print()
    print("Configuration: B1+B5+B2+B3+B4+B6+B7")
    print("Ground truth used: NO")
    print("External data used: NO")

    con.close()


if __name__ == "__main__":
    main()  