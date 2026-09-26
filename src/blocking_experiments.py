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
    """
    return f"""
        lower(
            trim(
                regexp_replace(
                    regexp_replace(
                        coalesce(CAST({column} AS VARCHAR), ''),
                        '[^\\\\w\\\\s]',
                        ' ',
                        'g'
                    ),
                    '\\\\s+',
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
                '\\\\d+'
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
                '\\\\d+'
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
    # BUILD A DETERMINISTIC "SIGNIFICANT" NAME TOKEN
    # --------------------------------------------------------
    #
    # Instead of a global token-frequency table (which can collapse to
    # almost no usable tokens on this dataset), choose one stable token
    # per business name: the longest normalized token meeting the minimum
    # length. This is still a significant normalized-name token and does
    # not use ground truth.

    print("\n[5/8] Building significant-name-token fields...")

    for table_name in ["s1_sample", "s2_norm", "s3_norm"]:
        con.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS name_token VARCHAR
            """
        )

        con.execute(
            f"""
            UPDATE {table_name}
            SET name_token = (
                SELECT token
                FROM unnest(string_split({table_name}.name_norm, ' ')) AS t(token)
                WHERE length(token) >= {TOKEN_MIN_LEN}
                  AND token <> ''
                ORDER BY length(token) DESC, token
                LIMIT 1
            )
            """
        )

    print("Token rule: longest normalized name token with length >=", TOKEN_MIN_LEN)

    # --------------------------------------------------------
    # COMPACT BLOCKING KEY TABLES
    # --------------------------------------------------------

    print("\nBuilding compact blocking-key tables...")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE s1_keys AS
        SELECT
            country_norm,
            name_norm,
            name_token,
            address_number,
            address_signature,
            COUNT(*) AS row_count
        FROM s1_sample
        GROUP BY
            country_norm,
            name_norm,
            name_token,
            address_number,
            address_signature
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE s2_keys AS
        SELECT
            country_norm,
            name_norm,
            name_token,
            address_number,
            address_signature,
            COUNT(*) AS row_count
        FROM s2_norm
        GROUP BY
            country_norm,
            name_norm,
            name_token,
            address_number,
            address_signature
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_keys AS
        SELECT
            country_norm,
            name_norm,
            name_token,
            address_number,
            address_signature,
            COUNT(*) AS row_count
        FROM s3_norm
        GROUP BY
            country_norm,
            name_norm,
            name_token,
            address_number,
            address_signature
    """)

    # --------------------------------------------------------
    # EXPERIMENT HELPERS
    # --------------------------------------------------------

    results = []

    def rule_condition(alias1, alias2, rule):
        if rule == "B1":
            return (
                f"{alias1}.country_norm = {alias2}.country_norm "
                f"AND {alias1}.name_norm <> '' "
                f"AND {alias1}.name_norm = {alias2}.name_norm"
            )
        if rule == "B2":
            return (
                f"{alias1}.country_norm = {alias2}.country_norm "
                f"AND {alias1}.name_token <> '' "
                f"AND {alias1}.name_token = {alias2}.name_token"
            )
        if rule == "B3":
            return (
                f"{alias1}.country_norm = {alias2}.country_norm "
                f"AND {alias1}.address_number <> '' "
                f"AND {alias1}.name_token <> '' "
                f"AND {alias1}.address_number = {alias2}.address_number "
                f"AND {alias1}.name_token = {alias2}.name_token"
            )
        if rule == "B4":
            return (
                f"{alias1}.country_norm = {alias2}.country_norm "
                f"AND {alias1}.address_signature <> '' "
                f"AND {alias1}.address_signature = {alias2}.address_signature"
            )
        raise ValueError(rule)

    def union_condition(alias1, alias2, rules):
        return " OR ".join(
            f"({rule_condition(alias1, alias2, r)})" for r in rules
        )

    def count_rule_pairs(source_table, rule):
        condition = rule_condition("a", "b", rule)
        return con.execute(
            f"""
            SELECT COALESCE(SUM(a.row_count * b.row_count), 0)
            FROM s1_keys a
            INNER JOIN {source_table} b
                ON {condition}
            """
        ).fetchone()[0] or 0

    def count_union_pairs(source_table, rules):
        condition = union_condition("a", "b", rules)
        return con.execute(
            f"""
            SELECT COALESCE(SUM(a.row_count * b.row_count), 0)
            FROM s1_keys a
            INNER JOIN {source_table} b
                ON {condition}
            """
        ).fetchone()[0] or 0

    def recall_for_rule(rule):
        if rule == "B1":
            cond2 = rule_condition("s1", "s2", "B1")
            cond3 = rule_condition("s1", "s3", "B1")
        elif rule == "B2":
            cond2 = rule_condition("s1", "s2", "B2")
            cond3 = rule_condition("s1", "s3", "B2")
        elif rule == "B3":
            cond2 = rule_condition("s1", "s2", "B3")
            cond3 = rule_condition("s1", "s3", "B3")
        else:
            cond2 = rule_condition("s1", "s2", "B4")
            cond3 = rule_condition("s1", "s3", "B4")

        return con.execute(
            f"""
            SELECT COUNT(*)
            FROM gt_pairs g
            INNER JOIN s1_sample s1
                ON s1.entity_id = g.s1_id
            LEFT JOIN s2_norm s2
                ON s2.entity_id = g.candidate_id
            LEFT JOIN s3_norm s3
                ON s3.entity_id = g.candidate_id
            WHERE
                (s2.entity_id IS NOT NULL AND {cond2})
                OR
                (s3.entity_id IS NOT NULL AND {cond3})
            """
        ).fetchone()[0] or 0

    def recall_for_union(rules):
        # Ground-truth recall is checked directly against the original rows.
        conditions2 = union_condition("s1", "s2", rules)
        conditions3 = union_condition("s1", "s3", rules)

        return con.execute(
            f"""
            SELECT COUNT(*)
            FROM gt_pairs g
            INNER JOIN s1_sample s1
                ON s1.entity_id = g.s1_id
            LEFT JOIN s2_norm s2
                ON s2.entity_id = g.candidate_id
            LEFT JOIN s3_norm s3
                ON s3.entity_id = g.candidate_id
            WHERE
                (s2.entity_id IS NOT NULL AND ({conditions2}))
                OR
                (s3.entity_id IS NOT NULL AND ({conditions3}))
            """
        ).fetchone()[0] or 0

    def add_result(experiment_id, blocking_rule, candidate_pairs, matched_pairs, notes):
        recall = (
            matched_pairs / total_true_matches
            if total_true_matches > 0 else 0.0
        )
        avg_candidates = candidate_pairs / SAMPLE_SIZE

        print()
        print("=" * 70)
        print(f"{experiment_id} - {blocking_rule}")
        print("=" * 70)
        print("Candidate pairs:", f"{candidate_pairs:,}")
        print("True IDs recovered:", f"{matched_pairs:,}")
        print("Missing true IDs:", f"{total_true_matches - matched_pairs:,}")
        print("Candidate recall:", f"{recall:.4%}")
        print("Average candidates/S1:", f"{avg_candidates:.2f}")

        results.append({
            "experiment_id": experiment_id,
            "blocking_rule": blocking_rule,
            "s1_rows": SAMPLE_SIZE,
            "candidate_pairs": candidate_pairs,
            "candidate_recall": recall,
            "avg_candidates_per_s1": avg_candidates,
            "notes": notes,
        })

    # --------------------------------------------------------
    # INDIVIDUAL BLOCKING PASSES
    # --------------------------------------------------------

    print("\n[6/8] Running blocking benchmarks...")

    individual_rules = {
        "B1": (
            "country + exact normalized business name",
            "Exact country + normalized name equality."
        ),
        "B2": (
            "country + significant normalized name token",
            f"Country + longest normalized name token of length >= {TOKEN_MIN_LEN}."
        ),
        "B3": (
            "country + address number + name token",
            "Country + extracted address number + significant normalized name token."
        ),
        "B4": (
            "country + address signature",
            "Country + first three normalized address tokens."
        ),
    }

    for rule, (label, notes) in individual_rules.items():
        candidate_pairs = (
            count_rule_pairs("s2_keys", rule)
            + count_rule_pairs("s3_keys", rule)
        )
        matched_pairs = recall_for_rule(rule)
        add_result(rule, label, candidate_pairs, matched_pairs, notes)

    # --------------------------------------------------------
    # UNION BENCHMARKS
    # --------------------------------------------------------
    #
    # These are calculated directly against compact grouped key tables,
    # not entity-level candidate tables. This avoids materializing billions
    # of candidate pairs.

    unions = [
        ("B1+B2", ["B1", "B2"], "B1 UNION B2"),
        ("B1+B2+B3", ["B1", "B2", "B3"], "B1 UNION B2 UNION B3"),
        ("B1+B2+B3+B4", ["B1", "B2", "B3", "B4"],
         "B1 UNION B2 UNION B3 UNION B4"),
    ]

    for experiment_id, rules, label in unions:
        candidate_pairs = (
            count_union_pairs("s2_keys", rules)
            + count_union_pairs("s3_keys", rules)
        )
        matched_pairs = recall_for_union(rules)
        add_result(
            experiment_id,
            label,
            candidate_pairs,
            matched_pairs,
            "Union calculated from compact blocking-key groups."
        )

    # --------------------------------------------------------
    # WRITE BENCHMARK CSV
    # --------------------------------------------------------

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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                key: row.get(key, "")
                for key in fieldnames
            })

    # --------------------------------------------------------
    # CREATE ANALYSIS
    # --------------------------------------------------------

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
    lines.append("Ground truth was used only for evaluation.")
    lines.append("It was not used to create blocking keys.")
    lines.append("")

    lines.append("## Blocking Passes Tested")
    lines.append("")
    lines.append("### B1")
    lines.append("Country + exact normalized business name.")
    lines.append("")
    lines.append("### B2")
    lines.append(
        f"Country + longest normalized name token with length >= {TOKEN_MIN_LEN}."
    )
    lines.append("")
    lines.append("### B3")
    lines.append(
        "Country + extracted address number + significant normalized name token."
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
    lines.append("| Experiment | Candidate pairs | Recall | Avg candidates/S1 |")
    lines.append("|---|---:|---:|---:|")
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
    for row in results[:4]:
        lines.append(
            f"- **{row['experiment_id']}** produced "
            f"{row['candidate_pairs']:,} candidate pairs with "
            f"{fmt_recall(row['candidate_recall'])} recall and "
            f"{row['avg_candidates_per_s1']:.2f} average candidates/S1."
        )

    lines.append("")
    lines.append("## Union Results")
    lines.append("")
    for row in results[4:]:
        lines.append(
            f"- **{row['experiment_id']}** produced "
            f"{row['candidate_pairs']:,} candidate pairs with "
            f"{fmt_recall(row['candidate_recall'])} recall and "
            f"{row['avg_candidates_per_s1']:.2f} average candidates/S1."
        )

    lines.append("")
    lines.append("## Configuration to Investigate Further")
    lines.append("")

    union_results = [r for r in results if "+" in r["experiment_id"]]
    high_recall = [
        r for r in union_results if r["candidate_recall"] >= 0.95
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
        f"In this {SAMPLE_SIZE:,}-S1 development sample it produced "
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
        "Ground truth was used only to calculate recall."
    )
    lines.append("")

    ANALYSIS_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print("PHASE 2 BLOCKING EXPERIMENTS COMPLETE")
    print("=" * 70)

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
    print(selected["experiment_id"], "-", selected["blocking_rule"])

    con.close()


if __name__ == "__main__":
    main()
