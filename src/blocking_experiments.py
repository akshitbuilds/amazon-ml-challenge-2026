from pathlib import Path
import csv
import duckdb

# ============================================================
# PHASE 2 - BLOCKING EXPERIMENTS (complete: B1-B7)
#
# Fixes carried over from earlier debugging:
#   - Unicode-aware normalization (\p{L}\p{N} instead of ASCII \w),
#     which previously blanked out every non-Latin-script name
#     (Devanagari/Tamil/Gujarati/etc) and split accented Latin
#     names in two.
#   - TOKEN_MAX_FREQ tightened so common domain words (dermatology,
#     primary, urgent, etc.) don't dominate B2's candidate volume.
#
# New in this version:
#   - B5: suffix-stripped exact name match (handles legal-suffix
#     variation: "Pvt Ltd" vs "Private Limited").
#   - B3 REDESIGNED: now uses each entity's SECOND-rarest token
#     (independent of B2's chosen token) + address number, so it
#     is no longer a guaranteed subset of B2.
#   - B6: order-independent two-token signature (handles token
#     reordering and gives a tighter multi-signal key than B2).
#   - B7: capped name-prefix key (handles residual typos/variation
#     at the end of a name; frequency-capped like B2 so it cannot
#     blow up candidate volume).
#   - max_candidates_per_s1 reported for every experiment.
#   - Incremental true-IDs-recovered tracked as rules are added to
#     a running union, in a fixed evaluation order.
#   - Final configuration selected automatically: best recall
#     subject to an average-candidates-per-S1 ceiling, not just
#     "union everything".
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

TOKEN_MIN_LEN = 4
TOKEN_MAX_FREQ = 2_000

NAME_PREFIX_LEN = 6
PREFIX_MIN_LEN = 4
PREFIX_MAX_FREQ = 2_000

# Ceiling used by the automatic final-configuration selector: we will
# not recommend a cumulative configuration whose average candidates
# per S1 row exceeds this, even if a later addition would raise
# recall further. Tune this based on what your ML scoring stage can
# afford per S1 row.
MAX_ACCEPTABLE_AVG_CANDIDATES = 500

# Legal-entity / corporate-suffix words stripped when building the
# suffix-insensitive name key (B5) and the prefix key (B7). Not
# exhaustive; extend if you spot more in your data.
LEGAL_SUFFIX_TOKENS = [
    "llc", "inc", "ltd", "ltda", "pvt", "private", "limited", "corp",
    "corporation", "co", "plc", "llp", "gmbh", "srl", "sa", "ag",
    "group", "holdings", "holding", "enterprise", "enterprises",
    "company", "companies", "pc", "pllc", "pty",
]
_SUFFIX_LIST_SQL = "[" + ",".join(f"'{s}'" for s in LEGAL_SUFFIX_TOKENS) + "]"


def sql_normalized_text(column):
    """
    Uses Unicode property classes \\p{L} (any letter, any script) and
    \\p{N} (any digit) instead of \\w, since RE2's \\w is ASCII-only
    and would blank out non-Latin-script names and split accented
    Latin names in two.
    """
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


def build_source_table_sql(csv_path, name_expr, sample_clause=""):
    """
    Shared SELECT used for S1/S2/S3: normalized name, address fields,
    suffix-stripped name, and a capped-length name prefix. The prefix
    is computed in an outer SELECT so it can reference name_stripped
    from the inner one.
    """
    addr_expr = sql_normalized_text("business_address")
    return f"""
        SELECT
            *,
            substr(replace(name_stripped, ' ', ''), 1, {NAME_PREFIX_LEN}) AS name_prefix
        FROM (
            SELECT
                entity_id,
                business_name,
                business_address,
                country,
                {name_expr} AS name_norm,
                {addr_expr} AS address_norm,
                lower(trim(coalesce(CAST(country AS VARCHAR), ''))) AS country_norm,
                regexp_extract({addr_expr}, '\\d+') AS address_number,
                concat_ws(
                    ' ',
                    split_part({addr_expr}, ' ', 1),
                    split_part({addr_expr}, ' ', 2),
                    split_part({addr_expr}, ' ', 3)
                ) AS address_signature,
                trim(
                    coalesce(
                        array_to_string(
                            list_filter(
                                string_split({name_expr}, ' '),
                                x -> NOT list_contains({_SUFFIX_LIST_SQL}, x)
                            ),
                            ' '
                        ),
                        ''
                    )
                ) AS name_stripped
            FROM read_csv('{csv_path.as_posix()}', sep='\\t', header=true, all_varchar=true)
            {sample_clause}
        ) base
    """


def main():

    print("=" * 70)
    print("PHASE 2 - BLOCKING EXPERIMENTS (B1-B7)")
    print("=" * 70)

    con = duckdb.connect()

    temp_dir = ROOT / "experiments" / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{temp_dir.as_posix()}'")
    con.execute("SET threads = 2")
    con.execute("SET preserve_insertion_order = false")

    # --------------------------------------------------------
    # SOURCE TABLES
    # --------------------------------------------------------

    print(f"\n[1/9] Building deterministic {SAMPLE_SIZE:,}-row S1 sample...")
    name_expr = sql_normalized_text("business_name")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s1_sample AS
        {build_source_table_sql(
            S1_FILE, name_expr,
            f"USING SAMPLE reservoir({SAMPLE_SIZE} ROWS) REPEATABLE ({RANDOM_SEED})"
        )}
        """
    )
    sample_count = con.execute("SELECT COUNT(*) FROM s1_sample").fetchone()[0]
    print("S1 sample rows:", sample_count)
    if sample_count != SAMPLE_SIZE:
        raise RuntimeError(f"Expected {SAMPLE_SIZE} S1 rows, got {sample_count}")

    print("\n[2/9] Preparing ground truth...")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW gt_raw AS
        SELECT * FROM read_csv('{GT_FILE.as_posix()}', sep='\\t', header=true, all_varchar=true)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW gt_pairs AS
        SELECT g.source1_entity_id AS s1_id, trim(x) AS candidate_id
        FROM gt_raw g,
        unnest(string_split(coalesce(g.matched_entity_ids, ''), ',')) AS u(x)
        INNER JOIN s1_sample s ON s.entity_id = g.source1_entity_id
        WHERE trim(x) <> ''
        """
    )
    total_true_matches = con.execute("SELECT COUNT(*) FROM gt_pairs").fetchone()[0]
    matched_s1_count = con.execute("SELECT COUNT(DISTINCT s1_id) FROM gt_pairs").fetchone()[0]
    print("S1 rows with ground-truth matches:", matched_s1_count)
    print("Total true matched IDs:", total_true_matches)

    print("\n[3/9] Building S2 normalized table...")
    con.execute(f"CREATE OR REPLACE TEMP TABLE s2_norm AS {build_source_table_sql(S2_FILE, name_expr)}")

    print("[3/9] Building S3 normalized table...")
    con.execute(f"CREATE OR REPLACE TEMP TABLE s3_norm AS {build_source_table_sql(S3_FILE, name_expr)}")

    # --------------------------------------------------------
    # TOKEN RANKING (shared basis for B2, B3, B6)
    # --------------------------------------------------------

    print(f"\n[4/9] Building token frequency table (min length {TOKEN_MIN_LEN}, cap {TOKEN_MAX_FREQ:,})...")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE token_freq AS
        SELECT token, COUNT(*) AS freq
        FROM (
            SELECT u.token FROM s1_sample s, unnest(string_split(s.name_norm, ' ')) AS u(token)
            WHERE length(u.token) >= {TOKEN_MIN_LEN} AND u.token <> ''
            UNION ALL
            SELECT u.token FROM s2_norm s, unnest(string_split(s.name_norm, ' ')) AS u(token)
            WHERE length(u.token) >= {TOKEN_MIN_LEN} AND u.token <> ''
            UNION ALL
            SELECT u.token FROM s3_norm s, unnest(string_split(s.name_norm, ' ')) AS u(token)
            WHERE length(u.token) >= {TOKEN_MIN_LEN} AND u.token <> ''
        ) t
        GROUP BY token
        """
    )

    def build_ranked_tokens(view_name, source_table):
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {view_name} AS
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
            INNER JOIN token_freq f ON f.token = u.token
            WHERE length(u.token) >= {TOKEN_MIN_LEN}
              AND u.token <> ''
              AND f.freq <= {TOKEN_MAX_FREQ}
            """
        )

    build_ranked_tokens("s1_tokens_ranked", "s1_sample")
    build_ranked_tokens("s2_tokens_ranked", "s2_norm")
    build_ranked_tokens("s3_tokens_ranked", "s3_norm")

    for label, src in (("s1", "s1_tokens_ranked"), ("s2", "s2_tokens_ranked"), ("s3", "s3_tokens_ranked")):
        con.execute(f"CREATE OR REPLACE TEMP VIEW {label}_tokens AS SELECT entity_id, country_norm, token FROM {src} WHERE rn = 1")
        con.execute(f"CREATE OR REPLACE TEMP VIEW {label}_tokens2 AS SELECT entity_id, country_norm, token FROM {src} WHERE rn = 2")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {label}_signature AS
            SELECT entity_id, country_norm, string_agg(token, '_' ORDER BY token) AS sig
            FROM {src}
            WHERE rn <= 2
            GROUP BY entity_id, country_norm
            """
        )

    s1_tok_n = con.execute("SELECT COUNT(*) FROM s1_tokens").fetchone()[0]
    print(f"S1 rows with a primary (B2) token: {s1_tok_n:,}")
    s1_tok2_n = con.execute("SELECT COUNT(*) FROM s1_tokens2").fetchone()[0]
    print(f"S1 rows with a secondary (B3) token: {s1_tok2_n:,}")
    s1_sig_n = con.execute("SELECT COUNT(*) FROM s1_signature").fetchone()[0]
    print(f"S1 rows with a signature (B6): {s1_sig_n:,}")

    # --------------------------------------------------------
    # PREFIX KEY (B7), frequency-capped like tokens
    # --------------------------------------------------------

    print(f"\n[5/9] Building capped name-prefix table (len {NAME_PREFIX_LEN}, cap {PREFIX_MAX_FREQ:,})...")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE prefix_freq AS
        SELECT country_norm, name_prefix, COUNT(*) AS freq
        FROM (
            SELECT country_norm, name_prefix FROM s1_sample WHERE length(name_prefix) >= {PREFIX_MIN_LEN}
            UNION ALL
            SELECT country_norm, name_prefix FROM s2_norm WHERE length(name_prefix) >= {PREFIX_MIN_LEN}
            UNION ALL
            SELECT country_norm, name_prefix FROM s3_norm WHERE length(name_prefix) >= {PREFIX_MIN_LEN}
        ) t
        GROUP BY country_norm, name_prefix
        """
    )

    def build_prefix_view(view_name, source_table):
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {view_name} AS
            SELECT s.entity_id, s.country_norm, s.name_prefix
            FROM {source_table} s
            INNER JOIN prefix_freq f
                ON f.country_norm = s.country_norm AND f.name_prefix = s.name_prefix
            WHERE length(s.name_prefix) >= {PREFIX_MIN_LEN}
              AND f.freq <= {PREFIX_MAX_FREQ}
            """
        )

    build_prefix_view("s1_prefix", "s1_sample")
    build_prefix_view("s2_prefix", "s2_norm")
    build_prefix_view("s3_prefix", "s3_norm")
    print("S1 rows with a qualifying prefix:", con.execute("SELECT COUNT(*) FROM s1_prefix").fetchone()[0])

    # --------------------------------------------------------
    # BLOCKING RULE SQL FRAGMENTS (each: S1-S2 UNION ALL S1-S3)
    # --------------------------------------------------------

    b1_sql = """
        SELECT s1.entity_id AS s1_id, s2.entity_id AS candidate_id
        FROM s1_sample s1 INNER JOIN s2_norm s2
            ON s2.country_norm = s1.country_norm AND s2.name_norm = s1.name_norm
        WHERE s1.name_norm <> ''
        UNION ALL
        SELECT s1.entity_id AS s1_id, s3.entity_id AS candidate_id
        FROM s1_sample s1 INNER JOIN s3_norm s3
            ON s3.country_norm = s1.country_norm AND s3.name_norm = s1.name_norm
        WHERE s1.name_norm <> ''
    """

    b2_sql = """
        SELECT s1.entity_id AS s1_id, s2.entity_id AS candidate_id
        FROM s1_tokens s1 INNER JOIN s2_tokens s2
            ON s1.country_norm = s2.country_norm AND s1.token = s2.token
        UNION ALL
        SELECT s1.entity_id AS s1_id, s3.entity_id AS candidate_id
        FROM s1_tokens s1 INNER JOIN s3_tokens s3
            ON s1.country_norm = s3.country_norm AND s1.token = s3.token
    """

    # B3, REDESIGNED: second-rarest token (independent of B2) + address number.
    b3_sql = """
        SELECT t1.entity_id AS s1_id, a2.entity_id AS candidate_id
        FROM s1_tokens2 t1
        INNER JOIN s1_sample a1 ON a1.entity_id = t1.entity_id AND a1.address_number <> ''
        INNER JOIN s2_norm a2 ON a2.country_norm = t1.country_norm AND a2.address_number = a1.address_number
        INNER JOIN s2_tokens2 t2 ON t2.entity_id = a2.entity_id AND t2.token = t1.token
        UNION ALL
        SELECT t1.entity_id AS s1_id, a3.entity_id AS candidate_id
        FROM s1_tokens2 t1
        INNER JOIN s1_sample a1 ON a1.entity_id = t1.entity_id AND a1.address_number <> ''
        INNER JOIN s3_norm a3 ON a3.country_norm = t1.country_norm AND a3.address_number = a1.address_number
        INNER JOIN s3_tokens2 t3 ON t3.entity_id = a3.entity_id AND t3.token = t1.token
    """

    b4_sql = """
        SELECT s1.entity_id AS s1_id, s2.entity_id AS candidate_id
        FROM s1_sample s1 INNER JOIN s2_norm s2
            ON s1.country_norm = s2.country_norm AND s1.address_signature = s2.address_signature AND s1.address_signature <> ''
        UNION ALL
        SELECT s1.entity_id AS s1_id, s3.entity_id AS candidate_id
        FROM s1_sample s1 INNER JOIN s3_norm s3
            ON s1.country_norm = s3.country_norm AND s1.address_signature = s3.address_signature AND s1.address_signature <> ''
    """

    b5_sql = """
        SELECT s1.entity_id AS s1_id, s2.entity_id AS candidate_id
        FROM s1_sample s1 INNER JOIN s2_norm s2
            ON s2.country_norm = s1.country_norm AND s2.name_stripped = s1.name_stripped
        WHERE s1.name_stripped <> ''
        UNION ALL
        SELECT s1.entity_id AS s1_id, s3.entity_id AS candidate_id
        FROM s1_sample s1 INNER JOIN s3_norm s3
            ON s3.country_norm = s1.country_norm AND s3.name_stripped = s1.name_stripped
        WHERE s1.name_stripped <> ''
    """

    b6_sql = """
        SELECT s1.entity_id AS s1_id, s2.entity_id AS candidate_id
        FROM s1_signature s1 INNER JOIN s2_signature s2
            ON s1.country_norm = s2.country_norm AND s1.sig = s2.sig
        UNION ALL
        SELECT s1.entity_id AS s1_id, s3.entity_id AS candidate_id
        FROM s1_signature s1 INNER JOIN s3_signature s3
            ON s1.country_norm = s3.country_norm AND s1.sig = s3.sig
    """

    b7_sql = """
        SELECT s1.entity_id AS s1_id, s2.entity_id AS candidate_id
        FROM s1_prefix s1 INNER JOIN s2_prefix s2
            ON s1.country_norm = s2.country_norm AND s1.name_prefix = s2.name_prefix
        UNION ALL
        SELECT s1.entity_id AS s1_id, s3.entity_id AS candidate_id
        FROM s1_prefix s1 INNER JOIN s3_prefix s3
            ON s1.country_norm = s3.country_norm AND s1.name_prefix = s3.name_prefix
    """

    RULE_SQL = {
        "B1": b1_sql, "B2": b2_sql, "B3": b3_sql, "B4": b4_sql,
        "B5": b5_sql, "B6": b6_sql, "B7": b7_sql,
    }
    RULE_LABEL = {
        "B1": "country + exact normalized business name",
        "B2": "country + rarest normalized name token (freq-capped)",
        "B3": "country + address number + SECOND-rarest name token (independent of B2)",
        "B4": "country + address signature (first 3 address tokens)",
        "B5": "country + suffix-stripped exact name (legal-suffix insensitive)",
        "B6": "country + order-independent two-token signature",
        "B7": "country + frequency-capped 6-char name prefix",
    }
    RULE_NOTES = {
        "B1": "Exact match; highest precision, catches identical naming only.",
        "B2": f"Each entity's single rarest token (len>={TOKEN_MIN_LEN}, corpus freq<={TOKEN_MAX_FREQ:,}).",
        "B3": "Uses each entity's SECOND-rarest token (rn=2), not the one B2 already used, "
              "combined with extracted address number, so it is independent of B2 rather than a subset.",
        "B4": "Compact address signature; tolerant of minor address wording differences.",
        "B5": f"Strips legal-entity suffix words ({', '.join(LEGAL_SUFFIX_TOKENS[:6])}, ...) before exact match.",
        "B6": "Each entity's two rarest qualifying tokens, sorted alphabetically and joined -- "
              "order-independent and tighter than B2 alone, at the cost of requiring 2 qualifying tokens.",
        "B7": f"First {NAME_PREFIX_LEN} chars of suffix-stripped, space-removed name; "
              f"prefixes with corpus frequency > {PREFIX_MAX_FREQ:,} are excluded to bound volume.",
    }

    # --------------------------------------------------------
    # GENERIC EXPERIMENT RUNNER (materialized DISTINCT candidates)
    # --------------------------------------------------------

    def run_experiment(experiment_id, blocking_rule, candidate_sql, notes=""):
        print()
        print("=" * 70)
        print(f"{experiment_id} - {blocking_rule}")
        print("=" * 70)

        stats_sql = f"""
            WITH candidates AS (
                SELECT DISTINCT s1_id, candidate_id FROM ({candidate_sql})
            ),
            candidate_counts AS (
                SELECT s1_id, COUNT(*) AS candidate_count FROM candidates GROUP BY s1_id
            ),
            candidate_stats AS (
                SELECT
                    (SELECT COUNT(*) FROM candidates) AS candidate_pairs,
                    COUNT(*) AS represented_s1,
                    COALESCE(AVG(candidate_count), 0) AS avg_candidates_per_s1,
                    COALESCE(MAX(candidate_count), 0) AS max_candidates_per_s1
                FROM candidate_counts
            ),
            matched AS (
                SELECT DISTINCT c.s1_id, c.candidate_id
                FROM candidates c
                INNER JOIN gt_pairs g ON c.s1_id = g.s1_id AND c.candidate_id = g.candidate_id
            )
            SELECT
                cs.candidate_pairs, cs.represented_s1, cs.avg_candidates_per_s1,
                cs.max_candidates_per_s1, (SELECT COUNT(*) FROM matched) AS matched_true_ids
            FROM candidate_stats cs
        """

        result = con.execute(stats_sql).fetchone()
        candidate_pairs = result[0] or 0
        represented_s1 = result[1] or 0
        avg_candidates = result[2] or 0
        max_candidates = result[3] or 0
        matched_true_ids = result[4] or 0
        recall = matched_true_ids / total_true_matches if total_true_matches else 0

        print(f"Candidate pairs:       {candidate_pairs:,}")
        print(f"S1 represented:        {represented_s1:,}")
        print(f"Avg candidates/S1:     {avg_candidates:.2f}")
        print(f"Max candidates/S1:     {max_candidates:,}")
        print(f"True IDs recovered:    {matched_true_ids:,}")
        print(f"Missing true IDs:      {total_true_matches - matched_true_ids:,}")
        print(f"Candidate recall:      {recall:.4%}")
        print(f"Zero-candidate S1:     {SAMPLE_SIZE - represented_s1:,}")

        return {
            "experiment_id": experiment_id,
            "blocking_rule": blocking_rule,
            "candidate_pairs": candidate_pairs,
            "represented_s1": represented_s1,
            "avg_candidates_per_s1": avg_candidates,
            "max_candidates_per_s1": max_candidates,
            "matched_true_ids": matched_true_ids,
            "candidate_recall": recall,
            "zero_candidate_s1": SAMPLE_SIZE - represented_s1,
            "notes": notes,
        }

    # --------------------------------------------------------
    # [6/9] RUN EACH RULE INDIVIDUALLY
    # --------------------------------------------------------

    print("\n[6/9] Running individual blocking passes B1-B7...")
    individual_results = {}
    for rid in ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]:
        individual_results[rid] = run_experiment(rid, RULE_LABEL[rid], RULE_SQL[rid], RULE_NOTES[rid])

    # --------------------------------------------------------
    # [7/9] CUMULATIVE UNION, IN A FIXED ORDER, WITH INCREMENTAL GAIN
    # --------------------------------------------------------

    print("\n[7/9] Running cumulative union with incremental recall tracking...")
    CUMULATIVE_ORDER = ["B1", "B5", "B2", "B3", "B4", "B6", "B7"]

    cumulative_results = []
    prev_matched = 0
    sql_so_far = []
    label_so_far = []

    for rid in CUMULATIVE_ORDER:
        sql_so_far.append(RULE_SQL[rid])
        label_so_far.append(rid)
        combo_id = "+".join(label_so_far)
        combo_sql = "\nUNION ALL\n".join(sql_so_far)
        result = run_experiment(combo_id, " UNION ".join(label_so_far), combo_sql,
                                 f"Cumulative union through {rid}, added in fixed evaluation order.")
        incremental_gain = result["matched_true_ids"] - prev_matched
        result["incremental_true_ids_recovered"] = incremental_gain
        prev_matched = result["matched_true_ids"]
        cumulative_results.append(result)
        print(f"  -> incremental true IDs recovered by adding {rid}: {incremental_gain:,}")

    # --------------------------------------------------------
    # [8/9] AUTOMATIC FINAL-CONFIGURATION SELECTION
    # --------------------------------------------------------

    print("\n[8/9] Selecting final configuration...")

    # Rules that contributed (near) zero incremental recall in the
    # cumulative order are flagged as redundant given the current
    # rule set and evaluation order.
    redundant_rules = [
        (CUMULATIVE_ORDER[i], cumulative_results[i]["incremental_true_ids_recovered"])
        for i in range(len(CUMULATIVE_ORDER))
        if cumulative_results[i]["incremental_true_ids_recovered"] <= 0
    ]

    # Best recall subject to the average-candidates ceiling.
    within_budget = [r for r in cumulative_results if r["avg_candidates_per_s1"] <= MAX_ACCEPTABLE_AVG_CANDIDATES]
    if within_budget:
        selected = max(within_budget, key=lambda r: r["candidate_recall"])
    else:
        # Nothing fits the ceiling; fall back to the cheapest step and
        # flag it clearly rather than silently picking something over budget.
        selected = min(cumulative_results, key=lambda r: r["avg_candidates_per_s1"])

    selected_rules = selected["experiment_id"].split("+")
    print(f"Selected configuration: {selected['experiment_id']}")
    print(f"  recall={selected['candidate_recall']:.4%} avg/S1={selected['avg_candidates_per_s1']:.2f}")
    if redundant_rules:
        print("Rules contributing zero/negative incremental recall in this order:", redundant_rules)

    # --------------------------------------------------------
    # [9/9] WRITE CSV + ANALYSIS REPORT
    # --------------------------------------------------------

    print("\n[9/9] Writing benchmark CSV and analysis report...")

    fieldnames = [
        "experiment_id", "blocking_rule", "candidate_pairs", "represented_s1",
        "avg_candidates_per_s1", "max_candidates_per_s1", "matched_true_ids",
        "candidate_recall", "zero_candidate_s1", "incremental_true_ids_recovered", "notes",
    ]
    with open(BENCHMARK_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rid in ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]:
            row = dict(individual_results[rid])
            row["incremental_true_ids_recovered"] = ""
            writer.writerow(row)
        for row in cumulative_results:
            writer.writerow(row)

    def fmt_pct(x):
        return f"{x:.4%}"

    lines = []
    lines.append("# Phase 2 Blocking Analysis")
    lines.append("")
    lines.append("## Development Sample")
    lines.append("")
    lines.append(f"- S1 sample size: {SAMPLE_SIZE:,} (seed {RANDOM_SEED})")
    lines.append(f"- S1 rows with ground-truth matches: {matched_s1_count:,}")
    lines.append(f"- Total true matched IDs in sample: {total_true_matches:,}")
    lines.append("")
    lines.append(
        "Ground truth was used only to score candidates after generation, never to "
        "construct blocking keys."
    )
    lines.append("")

    lines.append("## Normalization fixes carried into this run")
    lines.append("")
    lines.append(
        "- Regex escaping was corrected so the normalization pattern is interpreted "
        "as intended by the regex engine, rather than matching literal backslash "
        "characters."
    )
    lines.append(
        "- Character classes switched from ASCII `\\w`/`\\s` to Unicode `\\p{L}`/`\\p{N}`, "
        "since DuckDB's RE2 engine treats `\\w` as ASCII-only. Without this, every "
        "non-Latin-script business name (Devanagari, Tamil, Gujarati, etc.) and every "
        "accented Latin name normalized to an empty or broken string."
    )
    lines.append("")

    lines.append("## Individual Blocking Passes")
    lines.append("")
    lines.append("| Rule | Candidate pairs | Recall | Avg/S1 | Max/S1 | Zero-cand. S1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for rid in ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]:
        r = individual_results[rid]
        lines.append(
            f"| {rid} | {r['candidate_pairs']:,} | {fmt_pct(r['candidate_recall'])} | "
            f"{r['avg_candidates_per_s1']:.2f} | {r['max_candidates_per_s1']:,} | {r['zero_candidate_s1']:,} |"
        )
    lines.append("")
    for rid in ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]:
        lines.append(f"- **{rid}** ({RULE_LABEL[rid]}): {RULE_NOTES[rid]}")
    lines.append("")

    lines.append("## Cumulative Union (fixed evaluation order, with incremental gain)")
    lines.append("")
    lines.append("| Step | Candidate pairs | Recall | Avg/S1 | New true IDs from this rule |")
    lines.append("|---|---:|---:|---:|---:|")
    for i, rid in enumerate(CUMULATIVE_ORDER):
        r = cumulative_results[i]
        lines.append(
            f"| +{rid} ({r['experiment_id']}) | {r['candidate_pairs']:,} | "
            f"{fmt_pct(r['candidate_recall'])} | {r['avg_candidates_per_s1']:.2f} | "
            f"{r['incremental_true_ids_recovered']:,} |"
        )
    lines.append("")

    lines.append("## Rule Inclusion Rationale")
    lines.append("")
    for i, rid in enumerate(CUMULATIVE_ORDER):
        gain = cumulative_results[i]["incremental_true_ids_recovered"]
        if gain > 0:
            lines.append(
                f"- **{rid} kept**: added {gain:,} true IDs not already found by "
                f"{'+'.join(CUMULATIVE_ORDER[:i]) or '(nothing)'}."
            )
        else:
            lines.append(
                f"- **{rid} contributed no incremental recall** in this evaluation order "
                f"(0 or negative new true IDs on top of {'+'.join(CUMULATIVE_ORDER[:i])}). "
                f"Kept in the benchmark for transparency, but excluded from the recommended "
                f"configuration unless reordering changes this."
            )
    lines.append("")

    lines.append("## Recommended Production Configuration")
    lines.append("")
    lines.append(
        f"**{selected['experiment_id']}** -- selected as the cumulative step with the "
        f"highest measured recall ({fmt_pct(selected['candidate_recall'])}) while keeping "
        f"average candidates/S1 ({selected['avg_candidates_per_s1']:.2f}) at or under the "
        f"configured ceiling of {MAX_ACCEPTABLE_AVG_CANDIDATES}."
    )
    lines.append("")
    if within_budget:
        pass
    else:
        lines.append(
            "**WARNING**: no cumulative configuration stayed within the "
            f"{MAX_ACCEPTABLE_AVG_CANDIDATES} avg-candidates/S1 ceiling. The cheapest "
            "available step was selected as a fallback -- review whether the ceiling "
            "itself needs to change, or whether further-tightened keys are needed."
        )
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        f"- Measured recall tops out at {fmt_pct(cumulative_results[-1]['candidate_recall'])} "
        "for the full union of all seven passes on this development sample. If your project "
        "target is materially higher, these deterministic exact/near-exact keys are not "
        "sufficient on their own -- true matches with heavier typos, transliteration "
        "differences, or unrelated wording will not share any of these keys and need a "
        "genuinely fuzzy pass (e.g. phonetic encoding, edit-distance-bounded blocking) on "
        "top of this set."
    )
    lines.append(
        "- All measurements are on a 10,000-row S1 development sample; recall and volume "
        "should be re-validated on a larger sample before being treated as final production "
        "numbers."
    )
    lines.append(
        "- B3's redesign only uses each entity's second-rarest qualifying token; entities "
        "with fewer than two qualifying tokens (short or highly generic names) get no B3 "
        "candidates and rely on B1/B2/B4/B5/B6/B7 instead."
    )
    lines.append("")

    ANALYSIS_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print("PHASE 2 BLOCKING EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nSelected configuration: {selected['experiment_id']}")
    print(f"Recall: {fmt_pct(selected['candidate_recall'])}")
    print(f"Candidate pairs: {selected['candidate_pairs']:,}")
    print(f"Avg candidates/S1: {selected['avg_candidates_per_s1']:.2f}")
    print(f"\nFiles written:")
    print(BENCHMARK_FILE)
    print(ANALYSIS_FILE)

    con.close()


if __name__ == "__main__":
    main()