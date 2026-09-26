import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN = ROOT / "data" / "tsv" / "train"
TEST = ROOT / "data" / "tsv" / "test"
REPORT = ROOT / "reports" / "data_audit.txt"

files = [
    TRAIN / "train_source1.tsv",
    TRAIN / "train_source2.tsv",
    TRAIN / "train_source3.tsv",
    TRAIN / "train_ground_truth.tsv",
    TEST / "test_source1.tsv",
    TEST / "test_source2.tsv",
    TEST / "test_source3.tsv",
]

REPORT.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT, "w", encoding="utf-8") as out:

    for path in files:
        out.write("\n" + "=" * 80 + "\n")
        out.write(f"FILE: {path.name}\n")
        out.write("=" * 80 + "\n")

        df = pd.read_csv(path, sep="\t", nrows=10000)

        out.write(f"Columns: {list(df.columns)}\n")
        out.write(f"Sample rows inspected: {len(df)}\n\n")

        out.write("Dtypes:\n")
        out.write(str(df.dtypes))
        out.write("\n\n")

        out.write("Missing values in sample:\n")
        out.write(str(df.isna().sum()))
        out.write("\n\n")

        if "country" in df.columns:
            out.write("Country distribution in sample:\n")
            out.write(str(df["country"].value_counts(dropna=False)))
            out.write("\n\n")

        if "business_name" in df.columns:
            out.write("Business name length statistics:\n")
            lengths = df["business_name"].fillna("").astype(str).str.len()
            out.write(str(lengths.describe()))
            out.write("\n\n")

        if "business_address" in df.columns:
            out.write("Business address length statistics:\n")
            lengths = df["business_address"].fillna("").astype(str).str.len()
            out.write(str(lengths.describe()))
            out.write("\n\n")

        if "entity_id" in df.columns:
            out.write(
                f"Duplicate entity IDs in sample: "
                f"{df['entity_id'].duplicated().sum()}\n"
            )

print(f"Audit written to: {REPORT}")