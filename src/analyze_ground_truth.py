import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

path = ROOT / "dataset" / "train" / "train_ground_truth.tsv"

df = pd.read_csv(path, sep="\t")

print("Rows:", len(df))
print("Columns:", list(df.columns))

match_counts = []
s2_counts = []
s3_counts = []

for value in df["matched_entity_ids"].fillna(""):
    value = str(value).strip()

    if not value:
        ids = []
    else:
        ids = [x.strip() for x in value.split(",") if x.strip()]

    match_counts.append(len(ids))
    s2_counts.append(sum(x.startswith("S2-") for x in ids))
    s3_counts.append(sum(x.startswith("S3-") for x in ids))

df["match_count"] = match_counts
df["s2_count"] = s2_counts
df["s3_count"] = s3_counts

print("\nMATCH COUNT DISTRIBUTION")
print(df["match_count"].value_counts().sort_index())

print("\nS2 MATCH COUNT")
print(df["s2_count"].value_counts().sort_index())

print("\nS3 MATCH COUNT")
print(df["s3_count"].value_counts().sort_index())

print("\nZERO MATCH:", (df["match_count"] == 0).sum())
print("ONE MATCH:", (df["match_count"] == 1).sum())
print("MULTIPLE MATCH:", (df["match_count"] > 1).sum())

print("\nBOTH S2 AND S3:")
print(((df["s2_count"] > 0) & (df["s3_count"] > 0)).sum())

print("\nS2 ONLY:")
print(((df["s2_count"] > 0) & (df["s3_count"] == 0)).sum())

print("\nS3 ONLY:")
print(((df["s2_count"] == 0) & (df["s3_count"] > 0)).sum())