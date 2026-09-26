from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


DEFAULT_CHUNK_SIZE = 100_000


def load_tsv(
    path: Path,
    usecols: Optional[list[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a TSV file while preserving IDs and text fields as strings.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=usecols,
        nrows=nrows,
        keep_default_na=False,
    )


def iter_tsv_chunks(
    path: Path,
    usecols: Optional[list[str]] = None,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """
    Iterate over a TSV file in chunks.

    This is important for the Amazon dataset because the source
    files contain millions of rows and should not always be loaded
    completely into memory.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if chunksize <= 0:
        raise ValueError("chunksize must be greater than 0")

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=usecols,
        chunksize=chunksize,
        keep_default_na=False,
    )

    yield from reader


def load_source1(
    path: Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a Source 1 TSV."""
    return load_tsv(
        path,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=nrows,
    )


def load_source2(
    path: Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a Source 2 TSV."""
    return load_tsv(
        path,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=nrows,
    )


def load_source3(
    path: Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a Source 3 TSV."""
    return load_tsv(
        path,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=nrows,
    )


if __name__ == "__main__":
    from config import S1_TRAIN

    print("Testing io_utils...")
    print(f"Expected Source 1 path: {S1_TRAIN}")

    if S1_TRAIN.exists():
        df = load_source1(S1_TRAIN, nrows=5)

        print("\nLoaded successfully.")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\nSample:")
        print(df.to_string(index=False))
    else:
        print("\nDataset not found locally.")
        print("io_utils.py itself is ready.")