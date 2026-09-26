from pathlib import Path
import csv
import re
import duckdb

# ============================================================
# PHASE 2 - BLOCKING EXPERIMENTS
# Member 2
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN = ROOT / "data" / "tsv" / "train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_FILE = REPORT_DIR / "blocking_benchmark.csv"
ANALYSIS_FILE = REPORT_DIR / "phase2_blocking_analysis.md"

SAMPLE_SIZE = 10_000
RANDOM_SEED = 42

# Ignore short/common name tokens.
TOKEN_MIN_LEN = 4
TOKEN_MAX_FREQ = 100_000


def normalize_text_python(value):
    """
    Same basic normalization logic as src/normalize.py.
    Used for validation/documentation of the normalization approach.
    """
    if value is None:
        return ""

    value = str(value)

    import unicodedata

    value = unicodedata.normalize("NFKC", value)
    value = value.lower().strip()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def sql_normalized_text(column):
    """
    SQL equivalent of the project's normalize_text logic.

    FIX: previous version used 4 backslashes in the Python source
    (e.g. '[^\\\\w\\\\s]'), which produces the 2-character string
    '[^\\w\\s]'. To DuckDB's regex engine that means "not a literal
    backslash, not a literal w, not a literal s" -- NOT "not a word
    character, not a whitespace character". That shredded almost
    every business name down to scattered w/s fragments, which is
    why significant-name-token counts were ~0 and B1 (exact name
    match) was matching almost everything to everything.

    Using 2 backslashes here produces the string '[^\\w\\s]', which
    DuckDB passes through as \w \s to the regex engine -- the
    correct "not word or whitespace character" class.
    """
    return f"""
        lower(
            trim(
                regexp_replace(
                    regexp_replace(
                        coalesce(CAST({column} AS VARCHAR), ''),
                        '[^\\w\\s]',
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


def main():

    print("=" * 70)
    print("PHASE 2 - BLOCKING EXPERIMENTS")
    print("=" * 70)

    print("\nDataset files:")
    print("S1:", S1_FILE)
    print("S2:", S2_FILE)
    print("S3:", S3_FILE)
    print("GT:", GT_FILE)

    # --------------------------------------------------------
    # DuckDB setup
    # --------------------------------------------------------

    con = duckdb.connect()

    # Allow DuckDB to spill temporary work to disk if necessary.
    temp_dir = ROOT / "experiments" / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con.execute(
        f"SET temp_directory = '{temp_dir.as_posix()}'"
    )

    # Keep sampling deterministic.
    con.execute("SET threads = 2")
    con.execute("SET preserve_insertion_order = false")

    # --------------------------------------------------------
    # READ SOURCE 1
    # --------------------------------------------------------

    print(f"\n[1/8] Creating deterministic {SAMPLE_SIZE:,}-row S1 sample...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s1_sample AS
        SELECT
            entity_id,
            business_name,
            business_address,
            country,

            {sql_normalized_text("business_name")} AS name_norm,

            {sql_normalized_text("business_address")} AS address_norm,

            lower(trim(coalesce(CAST(country AS VARCHAR), '')))
                AS country_norm,

            regexp_extract(
                {sql_normalized_text("business_address")},
                '\\d+'
            ) AS address_number,

            concat_ws(
                ' ',
                split_part(
                    {sql_normalized_text("business_address")},
                    ' ',
                    1
                ),
                split_part(
                    {sql_normalized_text("business_address")},
                    ' ',
                    2
                ),
                split_part(
                    {sql_normalized_text("business_address")},
                    ' ',
                    3
                )
            ) AS address_signature

        FROM read_csv(
            '{S1_FILE.as_posix()}',
            sep='\\t',
            header=true,
            all_varchar=true
        )
        USING SAMPLE reservoir({SAMPLE_SIZE} ROWS)
        REPEATABLE ({RANDOM_SEED})
        """
    )

    sample_count = con.execute(
        "SELECT COUNT(*) FROM s1_sample"
    ).fetchone()[0]

    print("S1 sample rows:", sample_count)

    if sample_count != SAMPLE_SIZE:
        raise RuntimeError(
            f"Expected {SAMPLE_SIZE} S1 rows, got {sample_count}"
        )

    # --------------------------------------------------------
    # READ GROUND TRUTH
    # --------------------------------------------------------

    print("\n[2/8] Preparing ground truth...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW gt_raw AS
        SELECT *
        FROM read_csv(
            '{GT_FILE.as_posix()}',
            sep='\\t',
            header=true,
            all_varchar=true
        )
        """
    )

    # One row per true S1 -> S2/S3 match.
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW gt_pairs AS
        SELECT
            g.source1_entity_id AS s1_id,
            trim(x) AS candidate_id
        FROM gt_raw g,
        unnest(
            string_split(
                coalesce(g.matched_entity_ids, ''),
                ','
            )
        ) AS u(x)
        INNER JOIN s1_sample s
            ON s.entity_id = g.source1_entity_id
        WHERE trim(x) <> ''
        """
    )

    total_true_matches = con.execute(
        "SELECT COUNT(*) FROM gt_pairs"
    ).fetchone()[0]

    matched_s1_count = con.execute(
        "SELECT COUNT(DISTINCT s1_id) FROM gt_pairs"
    ).fetchone()[0]

    zero_match_s1_count = SAMPLE_SIZE - matched_s1_count

    print("S1 rows with ground-truth matches:", matched_s1_count)
    print("S1 rows with zero ground-truth matches:", zero_match_s1_count)
    print("Total true matched IDs:", total_true_matches)

    # --------------------------------------------------------
    # READ AND NORMALIZE SOURCE 2
    # --------------------------------------------------------

    print("\n[3/8] Preparing Source 2 view...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s2_norm AS
        SELECT
            entity_id,

            {sql_normalized_text("business_name")}
                AS name_norm,

            {sql_normalized_text("business_address")}
                AS address_norm,

            lower(trim(coalesce(CAST(country AS VARCHAR), '')))
                AS country_norm,

            regexp_extract(
                {sql_normalized_text("business_address")},
                '\\d+'
            ) AS address_number,

            concat_ws(
                ' ',
                split_part(
                    {sql_normalized_text("business_address")},
                    ' ',
                    1
                ),
                split_part(
                    {sql_normalized_text("business_address")},
                    ' ',
                    2
                ),
                split_part(
                    {sql_normalized_text("business_address")},
                    ' ',
                    3
                )
            ) AS address_signature

        FROM read_csv(
            '{S2_FILE.as_posix()}',
            sep='\\t',
            header=true,
            all_varchar=true
        )
        """
    )

    # --------------------------------------------------------
    # READ AND NORMALIZE SOURCE 3
    # --------------------------------------------------------

    print("\n[4/8] Preparing Source 3 view...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s3_norm AS
        SELECT
            entity_id,

            {sql_normalized_text("business_name")}
                AS name_norm,

            {sql_normalized_text("business_address")}
                AS address_norm,

            lower(trim(coalesce(CAST(country AS VARCHAR), '')))
                AS country_norm,

            regexp_extract(
                {sql_normalized_text("business_address")},
                '\\d+'
            ) AS address_number,

            concat_ws(
                ' ',
                split_part(
                    {sql_normalized_text("business_address")},
                    ' ',
                    1
                ),
                split_part(
                    {sql_normalized_text("business_address")},
                    ' ',
                    2
                ),
                split_part(
                    {sql_normalized_text("business_address")},
                    ' ',
                    3
                )
            ) AS address_signature

        FROM read_csv(
            '{S3_FILE.as_posix()}',
            sep='\\t',
            header=true,
            all_varchar=true
        )
        """
    )

    # --------------------------------------------------------
    # BUILD SIGNIFICANT NAME TOKEN FIELDS
    # --------------------------------------------------------

    print("\n[5/8] Building significant-name-token fields...")
    print(
        f"Token rule: rarest normalized name token with length >= {TOKEN_MIN_LEN} "
        f"and global frequency <= {TOKEN_MAX_FREQ:,}"
    )

    # FIX: the previous version picked each entity's *longest* token
    # >= 4 chars, but never enforced TOKEN_MAX_FREQ even though it was
    # defined. Once normalization is correct, the "longest token" is
    # very often a generic corporate word ("international",
    # "corporation", "group", "limited") that appears in a huge
    # fraction of ALL rows across S1/S2/S3. Blocking on that token
    # turns the join into a near cross-join, which is why B2/B3 hang.
    #
    # Fix: build a global token-frequency table across all three
    # sources, then for each entity pick the RAREST qualifying token
    # (lowest global frequency, not the longest), and drop the token
    # entirely if it exceeds TOKEN_MAX_FREQ.

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE token_freq AS
        SELECT token, COUNT(*) AS freq
        FROM (
            SELECT u.token
            FROM s1_sample s,
            unnest(string_split(s.name_norm, ' ')) AS u(token)
            WHERE length(u.token) >= {TOKEN_MIN_LEN}
              AND u.token <> ''

            UNION ALL

            SELECT u.token
            FROM s2_norm s,
            unnest(string_split(s.name_norm, ' ')) AS u(token)
            WHERE length(u.token) >= {TOKEN_MIN_LEN}
              AND u.token <> ''

            UNION ALL

            SELECT u.token
            FROM s3_norm s,
            unnest(string_split(s.name_norm, ' ')) AS u(token)
            WHERE length(u.token) >= {TOKEN_MIN_LEN}
              AND u.token <> ''
        ) t
        GROUP BY token
        """
    )

    token_count = con.execute("SELECT COUNT(*) FROM token_freq").fetchone()[0]
    print(f"Distinct tokens (length >= {TOKEN_MIN_LEN}) across all sources: {token_count:,}")

    print("\nTop 10 most common tokens (these get excluded once over the freq cap):")
    for row in con.execute(
        "SELECT token, freq FROM token_freq ORDER BY freq DESC LIMIT 10"
    ).fetchall():
        print(f"  {row[0]!r}: {row[1]:,}")

    over_cap = con.execute(
        f"SELECT COUNT(*) FROM token_freq WHERE freq > {TOKEN_MAX_FREQ}"
    ).fetchone()[0]
    print(f"Tokens excluded for exceeding freq cap ({TOKEN_MAX_FREQ:,}): {over_cap:,}")

    def build_tokens_view(view_name, source_table):
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {view_name} AS
            SELECT entity_id, country_norm, token
            FROM (
                SELECT
                    s.entity_id,
                    s.country_norm,
                    u.token,
                    f.freq,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.entity_id
                        ORDER BY f.freq ASC, length(u.token) DESC, u.token
                    ) AS rn
                FROM {source_table} s,
                unnest(string_split(s.name_norm, ' ')) AS u(token)
                INNER JOIN token_freq f
                    ON f.token = u.token
                WHERE length(u.token) >= {TOKEN_MIN_LEN}
                  AND u.token <> ''
                  AND f.freq <= {TOKEN_MAX_FREQ}
            )
            WHERE rn = 1
            """
        )

    build_tokens_view("s1_tokens", "s1_sample")
    build_tokens_view("s2_tokens", "s2_norm")
    build_tokens_view("s3_tokens", "s3_norm")

    print("\nCorrected significant-token coverage:")
    print(
        "S1 rows with non-empty significant token:",
        con.execute("SELECT COUNT(*) FROM s1_tokens").fetchone()[0],
    )
    print(
        "S2 rows with non-empty significant token:",
        con.execute("SELECT COUNT(*) FROM s2_tokens").fetchone()[0],
    )
    print(
        "S3 rows with non-empty significant token:",
        con.execute("SELECT COUNT(*) FROM s3_tokens").fetchone()[0],
    )

    print("\nShared B2 blocking-key check:")
    shared_s2 = con.execute(
        """
        SELECT COUNT(*)
        FROM (SELECT DISTINCT country_norm, token FROM s1_tokens) a
        INNER JOIN (SELECT DISTINCT country_norm, token FROM s2_tokens) b
          ON a.country_norm = b.country_norm
         AND a.token = b.token
        """
    ).fetchone()[0]

    shared_s3 = con.execute(
        """
        SELECT COUNT(*)
        FROM (SELECT DISTINCT country_norm, token FROM s1_tokens) a
        INNER JOIN (SELECT DISTINCT country_norm, token FROM s3_tokens) b
          ON a.country_norm = b.country_norm
         AND a.token = b.token
        """
    ).fetchone()[0]

    print("Shared S1-S2 (country, token) keys:", shared_s2)
    print("Shared S1-S3 (country, token) keys:", shared_s3)

    # --------------------------------------------------------
    # DIAGNOSTIC: distinct key counts + sample keys per source
    # (added per debugging request -- run once, informational only)
    # --------------------------------------------------------

    print("\n--- Diagnostics: distinct (country, token) keys ---")
    for label, view in (("S1", "s1_tokens"), ("S2", "s2_tokens"), ("S3", "s3_tokens")):
        distinct_count = con.execute(
            f"SELECT COUNT(DISTINCT (country_norm, token)) FROM {view}"
        ).fetchone()[0]
        print(f"{label} distinct (country, token) keys: {distinct_count}")

        sample_rows = con.execute(
            f"SELECT DISTINCT country_norm, token FROM {view} LIMIT 5"
        ).fetchall()
        print(f"{label} example (country, token) pairs: {sample_rows}")

    # --------------------------------------------------------
    # B3: country + address number + significant name token
    # --------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW s1_tokens_b3 AS
        SELECT
            t.entity_id,
            t.country_norm,
            s.address_number,
            t.token
        FROM s1_tokens t
        INNER JOIN s1_sample s
            ON s.entity_id = t.entity_id
        WHERE s.address_number <> ''
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW s2_tokens_b3 AS
        SELECT
            t.entity_id,
            t.country_norm,
            s.address_number,
            t.token
        FROM s2_tokens t
        INNER JOIN s2_norm s
            ON s.entity_id = t.entity_id
        WHERE s.address_number <> ''
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW s3_tokens_b3 AS
        SELECT
            t.entity_id,
            t.country_norm,
            s.address_number,
            t.token
        FROM s3_tokens t
        INNER JOIN s3_norm s
            ON s.entity_id = t.entity_id
        WHERE s.address_number <> ''
        """
    )

    print("\nB3 key coverage:")
    print(
        "S1 rows with B3 key:",
        con.execute("SELECT COUNT(*) FROM s1_tokens_b3").fetchone()[0],
    )
    print(
        "S2 rows with B3 key:",
        con.execute("SELECT COUNT(*) FROM s2_tokens_b3").fetchone()[0],
    )
    print(
        "S3 rows with B3 key:",
        con.execute("SELECT COUNT(*) FROM s3_tokens_b3").fetchone()[0],
    )

    # --------------------------------------------------------
    # HELPER FOR RUNNING AN EXPERIMENT
    # --------------------------------------------------------

    results = []

    def run_experiment(experiment_id, blocking_rule, candidate_sql, notes=""):
        print()
        print("=" * 70)
        print(f"{experiment_id} - {blocking_rule}")
        print("=" * 70)
        print("Running blocking query...")

        # B1 and B4 are evaluated without materializing candidate pairs.
        # This avoids huge intermediate joins for repeated names/addresses.
        if experiment_id in {"B1", "B4"}:
            if experiment_id == "B1":
                candidate_count_sql = """
                    WITH
                    s1_counts AS (
                        SELECT country_norm, name_norm, COUNT(*) AS s1_count
                        FROM s1_sample
                        WHERE name_norm <> ''
                        GROUP BY country_norm, name_norm
                    ),
                    s2_counts AS (
                        SELECT s2.country_norm, s2.name_norm, COUNT(*) AS s2_count
                        FROM s2_norm s2
                        INNER JOIN s1_b1_keys k
                            ON k.country_norm = s2.country_norm
                           AND k.name_norm = s2.name_norm
                        GROUP BY s2.country_norm, s2.name_norm
                    ),
                    s3_counts AS (
                        SELECT s3.country_norm, s3.name_norm, COUNT(*) AS s3_count
                        FROM s3_norm s3
                        INNER JOIN s1_b1_keys k
                            ON k.country_norm = s3.country_norm
                           AND k.name_norm = s3.name_norm
                        GROUP BY s3.country_norm, s3.name_norm
                    )
                    SELECT
                        COALESCE((
                            SELECT SUM(s1.s1_count * s2.s2_count)
                            FROM s1_counts s1
                            INNER JOIN s2_counts s2
                                ON s1.country_norm = s2.country_norm
                               AND s1.name_norm = s2.name_norm
                        ), 0)
                        +
                        COALESCE((
                            SELECT SUM(s1.s1_count * s3.s3_count)
                            FROM s1_counts s1
                            INNER JOIN s3_counts s3
                                ON s1.country_norm = s3.country_norm
                               AND s1.name_norm = s3.name_norm
                        ), 0)
                """
            else:
                candidate_count_sql = """
                    WITH
                    s1_counts AS (
                        SELECT country_norm, address_signature, COUNT(*) AS s1_count
                        FROM s1_sample
                        WHERE address_signature <> ''
                        GROUP BY country_norm, address_signature
                    ),
                    s2_counts AS (
                        SELECT country_norm, address_signature, COUNT(*) AS s2_count
                        FROM s2_norm
                        WHERE address_signature <> ''
                        GROUP BY country_norm, address_signature
                    ),
                    s3_counts AS (
                        SELECT country_norm, address_signature, COUNT(*) AS s3_count
                        FROM s3_norm
                        WHERE address_signature <> ''
                        GROUP BY country_norm, address_signature
                    )
                    SELECT
                        COALESCE((
                            SELECT SUM(s1.s1_count * s2.s2_count)
                            FROM s1_counts s1
                            INNER JOIN s2_counts s2
                                ON s1.country_norm = s2.country_norm
                               AND s1.address_signature = s2.address_signature
                        ), 0)
                        +
                        COALESCE((
                            SELECT SUM(s1.s1_count * s3.s3_count)
                            FROM s1_counts s1
                            INNER JOIN s3_counts s3
                                ON s1.country_norm = s3.country_norm
                               AND s1.address_signature = s3.address_signature
                        ), 0)
                """

            candidate_pairs = con.execute(candidate_count_sql).fetchone()[0] or 0

            if experiment_id == "B1":
                matched_pairs = con.execute("""
                    SELECT COUNT(*)
                    FROM gt_pairs g
                    INNER JOIN s1_sample s1
                        ON s1.entity_id = g.s1_id
                    LEFT JOIN s2_norm s2
                        ON s2.entity_id = g.candidate_id
                    LEFT JOIN s3_norm s3
                        ON s3.entity_id = g.candidate_id
                    WHERE s1.name_norm <> ''
                      AND (
                        (s2.entity_id IS NOT NULL
                         AND s2.country_norm = s1.country_norm
                         AND s2.name_norm = s1.name_norm)
                        OR
                        (s3.entity_id IS NOT NULL
                         AND s3.country_norm = s1.country_norm
                         AND s3.name_norm = s1.name_norm)
                      )
                """).fetchone()[0] or 0
            else:
                matched_pairs = con.execute("""
                    SELECT COUNT(*)
                    FROM gt_pairs g
                    INNER JOIN s1_sample s1
                        ON s1.entity_id = g.s1_id
                    LEFT JOIN s2_norm s2
                        ON s2.entity_id = g.candidate_id
                    LEFT JOIN s3_norm s3
                        ON s3.entity_id = g.candidate_id
                    WHERE s1.address_signature <> ''
                      AND (
                        (s2.entity_id IS NOT NULL
                         AND s2.country_norm = s1.country_norm
                         AND s2.address_signature = s1.address_signature)
                        OR
                        (s3.entity_id IS NOT NULL
                         AND s3.country_norm = s1.country_norm
                         AND s3.address_signature = s1.address_signature)
                      )
                """).fetchone()[0] or 0

            recall = matched_pairs / total_true_matches if total_true_matches else 0
            avg_candidates = candidate_pairs / SAMPLE_SIZE

            print(f"Candidate pairs:       {candidate_pairs:,}")
            print(f"True IDs recovered:    {matched_pairs:,}")
            print(f"Missing true IDs:      {total_true_matches - matched_pairs:,}")
            print(f"Candidate recall:      {recall:.4%}")
            print(f"Average candidates/S1: {avg_candidates:.2f}")

            return {
                "experiment_id": experiment_id,
                "blocking_rule": blocking_rule,
                "s1_rows": SAMPLE_SIZE,
                "candidate_pairs": candidate_pairs,
                "candidate_recall": recall,
                "avg_candidates_per_s1": avg_candidates,
                "notes": notes,
            }

        # B2/B3 and unions: aggregate directly without persistent TEMP tables.
        stats_sql = f"""
            WITH candidates AS (
                SELECT DISTINCT s1_id, candidate_id
                FROM ({candidate_sql})
            ),
            candidate_counts AS (
                SELECT s1_id, COUNT(*) AS candidate_count
                FROM candidates
                GROUP BY s1_id
            ),
            candidate_stats AS (
                SELECT
                    (SELECT COUNT(*) FROM candidates) AS candidate_pairs,
                    COUNT(*) AS represented_s1,
                    COALESCE(AVG(candidate_count), 0) AS avg_candidates_per_s1
                FROM candidate_counts
            ),
            matched AS (
                SELECT DISTINCT c.s1_id, c.candidate_id
                FROM candidates c
                INNER JOIN gt_pairs g
                    ON c.s1_id = g.s1_id
                   AND c.candidate_id = g.candidate_id
            )
            SELECT
                cs.candidate_pairs,
                cs.represented_s1,
                cs.avg_candidates_per_s1,
                (SELECT COUNT(*) FROM matched) AS matched_true_ids
            FROM candidate_stats cs
        """

        result = con.execute(stats_sql).fetchone()
        candidate_pairs = result[0] or 0
        represented_s1 = result[1] or 0
        avg_candidates = result[2] or 0
        matched_true_ids = result[3] or 0

        recall = matched_true_ids / total_true_matches if total_true_matches else 0

        print(f"Candidate pairs:       {candidate_pairs:,}")
        print(f"S1 represented:        {represented_s1:,}")
        print(f"Avg candidates / S1:   {avg_candidates:.2f}")
        print(f"True IDs recovered:    {matched_true_ids:,}")
        print(f"Missing true IDs:      {total_true_matches - matched_true_ids:,}")
        print(f"Candidate recall:      {recall:.4%}")
        print(f"Zero-candidate S1:     {SAMPLE_SIZE - represented_s1:,}")

        return {
            "experiment_id": experiment_id,
            "blocking_rule": blocking_rule,
            "s1_rows": SAMPLE_SIZE,
            "candidate_pairs": candidate_pairs,
            "candidate_recall": recall,
            "avg_candidates_per_s1": avg_candidates,
            "notes": notes,
        }

    print("\nPreparing B1 lookup keys...")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE s1_b1_keys AS
        SELECT DISTINCT
            country_norm,
            name_norm
        FROM s1_sample
        WHERE name_norm <> ''
        """
    )

    b1_key_count = con.execute(
        """
        SELECT COUNT(*)
        FROM s1_b1_keys
        """
    ).fetchone()[0]

    print("B1 unique country+name keys:", b1_key_count)

    # ========================================================
    # B1
    # country + exact normalized business name
    # ========================================================

    b1_sql = """
        SELECT
            s1.entity_id AS s1_id,
            s2.entity_id AS candidate_id
        FROM s1_sample s1
        INNER JOIN s1_b1_keys k
            ON k.country_norm = s1.country_norm
           AND k.name_norm = s1.name_norm
        INNER JOIN s2_norm s2
            ON s2.country_norm = k.country_norm
           AND s2.name_norm = k.name_norm
        WHERE s1.name_norm <> ''

        UNION ALL

        SELECT
            s1.entity_id AS s1_id,
            s3.entity_id AS candidate_id
        FROM s1_sample s1
        INNER JOIN s1_b1_keys k
            ON k.country_norm = s1.country_norm
           AND k.name_norm = s1.name_norm
        INNER JOIN s3_norm s3
            ON s3.country_norm = k.country_norm
           AND s3.name_norm = k.name_norm
        WHERE s1.name_norm <> ''
    """

    results.append(run_experiment(
        "B1",
        "country + exact normalized business name",
        b1_sql,
        "Exact country + normalized name equality."
    ))

    # ========================================================
    # B2
    # country + significant normalized name token
    # ========================================================

    b2_sql = """
        SELECT
            s1.entity_id AS s1_id,
            s2.entity_id AS candidate_id
        FROM s1_tokens s1
        INNER JOIN s2_tokens s2
            ON s1.country_norm = s2.country_norm
           AND s1.token = s2.token

        UNION

        SELECT
            s1.entity_id AS s1_id,
            s3.entity_id AS candidate_id
        FROM s1_tokens s1
        INNER JOIN s3_tokens s3
            ON s1.country_norm = s3.country_norm
           AND s1.token = s3.token
    """

    results.append(run_experiment(
        "B2",
        "country + significant normalized name token",
        b2_sql,
        f"Longest normalized name token with length >= {TOKEN_MIN_LEN} is used per entity; short tokens are ignored."
    ))

    b3_sql = """
        SELECT
            s1.entity_id AS s1_id,
            s2.entity_id AS candidate_id
        FROM s1_tokens_b3 s1
        INNER JOIN s2_tokens_b3 s2
            ON s1.country_norm = s2.country_norm
           AND s1.address_number = s2.address_number
           AND s1.token = s2.token

        UNION

        SELECT
            s1.entity_id AS s1_id,
            s3.entity_id AS candidate_id
        FROM s1_tokens_b3 s1
        INNER JOIN s3_tokens_b3 s3
            ON s1.country_norm = s3.country_norm
           AND s1.address_number = s3.address_number
           AND s1.token = s3.token
    """

    results.append(run_experiment(
        "B3",
        "country + address number + name token",
        b3_sql,
        "Country + extracted address number + longest normalized name token of length >= 4."
    ))

    # ========================================================
    # B4
    # country + address signature
    # ========================================================

    b4_sql = """
        SELECT
            s1.entity_id AS s1_id,
            s2.entity_id AS candidate_id
        FROM s1_sample s1
        INNER JOIN s2_norm s2
            ON s1.country_norm = s2.country_norm
           AND s1.address_signature = s2.address_signature
           AND s1.address_signature <> ''

        UNION

        SELECT
            s1.entity_id AS s1_id,
            s3.entity_id AS candidate_id
        FROM s1_sample s1
        INNER JOIN s3_norm s3
            ON s1.country_norm = s3.country_norm
           AND s1.address_signature = s3.address_signature
           AND s1.address_signature <> ''
    """

    results.append(run_experiment(
        "B4",
        "country + address signature",
        b4_sql,
        "Country + first three normalized address tokens; full address equality is not required."
    ))

    # ========================================================
    # UNION 1
    # B1 UNION B2
    # ========================================================

    union12_sql = f"""
        {b1_sql}

        UNION

        {b2_sql}
    """

    results.append(run_experiment(
        "B1+B2",
        "B1 UNION B2",
        union12_sql,
        "Union of exact normalized-name blocking and significant-name-token blocking."
    ))

    # ========================================================
    # UNION 2
    # B1 UNION B2 UNION B3
    # ========================================================

    union123_sql = f"""
        {b1_sql}

        UNION

        {b2_sql}

        UNION

        {b3_sql}
    """

    results.append(run_experiment(
        "B1+B2+B3",
        "B1 UNION B2 UNION B3",
        union123_sql,
        "Adds address-number + informative-name-token blocking to B1+B2."
    ))

    # ========================================================
    # UNION 3
    # B1 UNION B2 UNION B3 UNION B4
    # ========================================================

    union1234_sql = f"""
        {b1_sql}

        UNION

        {b2_sql}

        UNION

        {b3_sql}

        UNION

        {b4_sql}
    """

    results.append(run_experiment(
        "B1+B2+B3+B4",
        "B1 UNION B2 UNION B3 UNION B4",
        union1234_sql,
        "Full tested blocking union."
    ))

    # ========================================================
    # WRITE BENCHMARK CSV
    # ========================================================

    print("\n[7/8] Writing benchmark CSV...")

    fieldnames = [
        "experiment_id",
        "blocking_rule",
        "s1_rows",
        "candidate_pairs",
        "candidate_recall",
        "avg_candidates_per_s1",
        "notes",
    ]

    with open(
        BENCHMARK_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    # ========================================================
    # CREATE ANALYSIS
    # ========================================================

    print("[8/8] Writing blocking analysis...")

    def fmt_recall(x):
        return f"{x:.4%}"

    lines = []

    lines.append("# Phase 2 Blocking Analysis")
    lines.append("")
    lines.append("## Development Sample")
    lines.append("")
    lines.append(f"- S1 sample size: {SAMPLE_SIZE:,}")
    lines.append(f"- Random seed: {RANDOM_SEED}")
    lines.append(f"- S1 rows with ground-truth matches: {matched_s1_count:,}")
    lines.append(f"- S1 rows with zero ground-truth matches: {zero_match_s1_count:,}")
    lines.append(f"- Total true matched IDs in sample: {total_true_matches:,}")
    lines.append("")
    lines.append("Ground truth was used only for evaluation after candidate generation.")
    lines.append("It was not used to construct blocking keys.")
    lines.append("")

    lines.append("## Blocking Passes Tested")
    lines.append("")
    lines.append("### B1")
    lines.append("Country + exact normalized business name.")
    lines.append("")
    lines.append("### B2")
    lines.append(
        f"Country + significant normalized name token. "
        f"Tokens shorter than {TOKEN_MIN_LEN} characters or appearing more than "
        f"{TOKEN_MAX_FREQ} times were ignored."
    )
    lines.append("")
    lines.append("### B3")
    lines.append(
        "Country + extracted address number + longest normalized name token of length >= 4."
    )
    lines.append("")
    lines.append("### B4")
    lines.append(
        "Country + compact address signature consisting of the first three "
        "normalized address tokens."
    )
    lines.append("")

    lines.append("## Benchmark Results")
    lines.append("")
    lines.append(
        "| Experiment | Candidate pairs | Recall | Avg candidates/S1 |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )

    for row in results:
        lines.append(
            f"| {row['experiment_id']} | "
            f"{row['candidate_pairs']:,} | "
            f"{fmt_recall(row['candidate_recall'])} | "
            f"{row['avg_candidates_per_s1']:.2f} |"
        )

    lines.append("")
    lines.append("## Main Observations")
    lines.append("")

    # Individual passes
    individual = [
        r for r in results
        if r["experiment_id"] in {"B1", "B2", "B3", "B4"}
    ]

    for r in individual:
        lines.append(
            f"- **{r['experiment_id']}** produced "
            f"{r['candidate_pairs']:,} candidate pairs with "
            f"{fmt_recall(r['candidate_recall'])} candidate recall and "
            f"{r['avg_candidates_per_s1']:.2f} average candidates/S1."
        )

    lines.append("")
    lines.append("## Union Results")
    lines.append("")

    for r in results:
        if "+" in r["experiment_id"]:
            lines.append(
                f"- **{r['experiment_id']}** produced "
                f"{r['candidate_pairs']:,} candidate pairs with "
                f"{fmt_recall(r['candidate_recall'])} recall and "
                f"{r['avg_candidates_per_s1']:.2f} average candidates/S1."
            )

    lines.append("")
    lines.append("## Configuration to Investigate Further")
    lines.append("")

    union_results = [
        r for r in results
        if "+" in r["experiment_id"]
    ]

    # Prefer a union with at least 95% measured recall and the
    # smallest candidate volume. If none reaches 95%, use the
    # union with the highest measured recall, breaking ties by
    # candidate volume.
    high_recall = [
        r for r in union_results
        if r["candidate_recall"] >= 0.95
    ]

    if high_recall:
        selected = min(
            high_recall,
            key=lambda r: (
                r["avg_candidates_per_s1"],
                -r["candidate_recall"]
            )
        )
    else:
        selected = max(
            union_results,
            key=lambda r: (
                r["candidate_recall"],
                -r["avg_candidates_per_s1"]
            )
        )

    lines.append(
        f"The measured trade-off to investigate further is "
        f"**{selected['experiment_id']} ({selected['blocking_rule']})**."
    )

    lines.append("")
    lines.append(
        f"In this 50,000-S1 development sample it produced "
        f"{selected['candidate_pairs']:,} candidate pairs, "
        f"{fmt_recall(selected['candidate_recall'])} measured candidate recall, "
        f"and {selected['avg_candidates_per_s1']:.2f} average candidates/S1."
    )

    lines.append("")
    lines.append(
        "This is a measured development-sample trade-off, not a final "
        "production conclusion. It should be validated on a larger sample "
        "before integration."
    )

    lines.append("")
    lines.append("## Leakage Check")
    lines.append("")
    lines.append(
        "Ground truth was not used to create B1, B2, B3, or B4. "
        "Ground truth was used only after candidate generation to calculate recall."
    )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "The experiments use the training TSV files and a deterministic "
        "50,000-row S1 development sample. Original business fields are "
        "preserved in the source views; normalized fields are derived fields."
    )
    lines.append("")

    ANALYSIS_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2 BLOCKING EXPERIMENTS COMPLETE")
    print("=" * 70)

    print("\nBenchmark:")
    for row in results:
        print(
            f"{row['experiment_id']:12s} "
            f"pairs={row['candidate_pairs']:,} "
            f"recall={row['candidate_recall']:.6f} "
            f"avg/S1={row['avg_candidates_per_s1']:.2f}"
        )

    print("\nFiles created:")
    print(BENCHMARK_FILE)
    print(ANALYSIS_FILE)

    print("\nConfiguration to investigate further:")
    print(
        selected["experiment_id"],
        "-",
        selected["blocking_rule"]
    )

    con.close()


if __name__ == "__main__":
    main()