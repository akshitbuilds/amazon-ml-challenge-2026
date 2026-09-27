from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "tsv" / "train"
CANDIDATE_FILE = ROOT / "experiments" / "phase3_training_candidates.tsv"
GT_FILE = DATA_DIR / "train_ground_truth.tsv"
S1_FILE = DATA_DIR / "train_source1.tsv"
S2_FILE = DATA_DIR / "train_source2.tsv"
S3_FILE = DATA_DIR / "train_source3.tsv"


def main():
    c = pd.read_csv(CANDIDATE_FILE, sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False)
    s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(S2_FILE, sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(S3_FILE, sep="\t", dtype=str, keep_default_na=False)

    assert not c.duplicated(["source1_entity_id", "matched_entity_id"]).any()
    assert set(c["label"].unique()).issubset({"0", "1"})
    assert set(c["source1_entity_id"]).issubset(set(s1["entity_id"]))
    assert set(c["matched_entity_id"]).issubset(
        set(s2["entity_id"]) | set(s3["entity_id"])
    )

    gt_pairs = set()
    for row in gt.itertuples(index=False):
        matched = str(row.matched_entity_ids).strip()
        if matched:
            for target in matched.split(","):
                target = target.strip()
                if target:
                    gt_pairs.add((str(row.source1_entity_id), target))

    positives = c[c["label"] == "1"]
    positive_pairs = set(
        zip(positives["source1_entity_id"], positives["matched_entity_id"])
    )
    assert positive_pairs.issubset(gt_pairs)

    print("Phase 3 validation PASSED.")
    print(f"Candidate pairs: {len(c):,}")
    print(f"Positive pairs: {len(positives):,}")
    print(f"Negative pairs: {len(c) - len(positives):,}")


if __name__ == "__main__":
    main()
