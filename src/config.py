from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"
REPORT_DIR = ROOT / "reports"
EXPERIMENT_DIR = ROOT / "experiments"

S1_TRAIN = TRAIN_DIR / "train_source1.tsv"
S2_TRAIN = TRAIN_DIR / "train_source2.tsv"
S3_TRAIN = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

S1_TEST = TEST_DIR / "test_source1.tsv"
S2_TEST = TEST_DIR / "test_source2.tsv"
S3_TEST = TEST_DIR / "test_source3.tsv"