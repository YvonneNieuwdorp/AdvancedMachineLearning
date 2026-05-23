"""
checker.py
----------
Validates the scraped Billboard dataset and prints a summary report.

Can be run standalone:
    python src/checker.py
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BILLBOARD_PATH = os.path.join(BASE_DIR, "data", "billboard_year_end_2003_2023.csv")


def check_billboard_dataset(path: str) -> pd.DataFrame:
    """
    Loads and validates the Billboard Year-End dataset.

    Prints a summary including shape, missing values, year distribution,
    rank statistics, and a sample of unique artists.

    Args:
        path: Path to the Billboard CSV file.

    Returns:
        The loaded DataFrame.
    """
    df = pd.read_csv(path)

    print("\n=== BASIC INFO ===")
    print(df.head())
    print(df.info())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    print("\n=== ARTIST NUMERIC CHECK (should be empty) ===")
    numeric_artists = df[df["artist"].astype(str).str.isnumeric()]
    print(f"Found {len(numeric_artists)} numeric artist rows")

    print("\n=== YEAR DISTRIBUTION ===")
    print(df["year"].value_counts().sort_index())

    print("\n=== RANK CHECK ===")
    print(df["rank"].describe())

    print("\n=== UNIQUE ARTISTS SAMPLE ===")
    print(df["artist"].dropna().unique()[:20])

    print("\n=== DONE ===")
    return df


if __name__ == "__main__":
    check_billboard_dataset(BILLBOARD_PATH)
