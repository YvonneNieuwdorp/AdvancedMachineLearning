"""
cleaning.py
-----------
Downloads the Genius lyrics dataset, filters it, and produces two output files:

1. clean_genre_lyrics.csv  – used for genre classification
   Columns : tag, lyrics
   Range   : 1923–2023, English only, uniform distribution across genres

2. clean_hit_lyrics.csv    – used for hit prediction
   Columns : title, artist, lyrics, rank, top_10
   Range   : 2003–2023, English only, merged with Billboard chart data

Can be run standalone:
    python src/cleaning.py

Outputs:
    data/clean_genre_lyrics.csv
    data/clean_hit_lyrics.csv
"""

import os
import glob
import re
import pandas as pd
import kagglehub

RUN_PREPROCESSING = False  # Set to True to re-run the full pipeline

BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BILLBOARD_PATH     = os.path.join(BASE_DIR, "data", "billboard_year_end_2003_2023.csv")
GENRE_OUTPUT_PATH  = os.path.join(BASE_DIR, "data", "clean_genre_lyrics.csv")
HIT_OUTPUT_PATH    = os.path.join(BASE_DIR, "data", "clean_hit_lyrics.csv")

KAGGLE_DATASET = "carlosgdcj/genius-song-lyrics-with-language-information"

# Year ranges
GENRE_START_YEAR = 1923
GENRE_END_YEAR   = 2023
HIT_START_YEAR   = 2003
HIT_END_YEAR     = 2023


# Normalisation helpers

def clean_artist(artist: str) -> str:
    """
    Normalises an artist name for matching by removing featured artists
    and standardising formatting.

    Splits on 'Featuring', 'with', '&', ',', or a capital-X separator,
    keeps only the primary artist, and returns the result lowercased.

    Args:
        artist: Raw artist string (e.g. 'NellyFeaturingTimbaland').

    Returns:
        Cleaned, lowercased primary artist name (e.g. 'nelly').
    """
    if not isinstance(artist, str):
        return artist
    artist = re.sub(r'(?i)([a-z])([Ff]eaturing)', r'\1 \2', artist)
    artist = re.split(r'(?i)\bfeaturing\b|\bwith\b|&|,|[xX](?=[A-Z])', artist)[0]
    return artist.strip().lower()


def clean_title(title: str) -> str:
    """
    Normalises a song title for matching by removing parenthetical text
    and punctuation.

    Args:
        title: Raw song title (e.g. "Hips Don't Lie (feat. Wyclef Jean)").

    Returns:
        Cleaned, lowercased title (e.g. 'hips dont lie').
    """
    if not isinstance(title, str):
        return title
    title = re.sub(r'\(.*?\)', '', title)
    title = re.sub(r'[^\w\s]', '', title)
    return title.strip().lower()


# Shared download helper

def _download_raw_df(usecols: list[str]) -> pd.DataFrame:
    """
    Downloads the Kaggle dataset and returns a DataFrame with the requested
    columns, filtered to English songs with non-null lyrics.

    Args:
        usecols: Columns to load from the raw CSV.

    Returns:
        DataFrame filtered to English rows with valid lyrics.

    Raises:
        FileNotFoundError: If no CSV is found in the downloaded dataset.
    """
    print("Starting Kaggle dataset download...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Dataset downloaded to cache: {path}")

    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found at {path}")

    raw_path = csv_files[0]
    print(f"CSV file found: {raw_path}")

    df = pd.read_csv(raw_path, usecols=usecols)
    print("Data loaded!")

    df = df[df["language"] == "en"]
    df = df.dropna(subset=["lyrics"])
    print(f"After language + lyrics filter: {df.shape}")

    return df


# Pipeline for genre classification

def build_genre_dataset() -> None:
    """
    Produces clean_genre_lyrics.csv for genre classification.

    Steps:
    - Download Genius dataset (1923–2023)
    - Keep English songs only
    - Drop 'misc' tag and rows without a tag
    - Undersample to a uniform distribution across genres
    - Save columns: tag, lyrics
    """
    print("\n=== GENRE DATASET ===")

    df = _download_raw_df(usecols=["tag", "year", "lyrics", "language"])

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] >= GENRE_START_YEAR) & (df["year"] <= GENRE_END_YEAR)]
    print(f"After year filter ({GENRE_START_YEAR}–{GENRE_END_YEAR}): {df.shape}")

    df = df[df["tag"] != "misc"]
    df = df.dropna(subset=["tag"])
    print(f"After tag filter: {df.shape}")

    # Uniform distribution: undersample each genre to the size of the smallest
    min_count = df["tag"].value_counts().min()
    print(f"\nGenre counts before balancing:\n{df['tag'].value_counts()}")
    df = (
        df.groupby("tag", group_keys=False)
          .apply(lambda g: g.sample(n=min_count, random_state=42))
          .reset_index(drop=True)
    )
    print(f"\nGenre counts after balancing (uniform at {min_count}):\n{df['tag'].value_counts()}")

    df = df[["tag", "lyrics"]]
    df.to_csv(GENRE_OUTPUT_PATH, index=False)
    print(f"\nSaved to {GENRE_OUTPUT_PATH}  ({len(df):,} rows)")


# Pipeline for hit prediction

def build_hit_dataset() -> None:
    """
    Produces clean_hit_lyrics.csv for hit prediction.

    Steps:
    - Download Genius dataset (2003–2023)
    - Keep English songs only
    - Merge with Billboard chart data on normalised title + artist
    - Save columns: title, artist, lyrics, rank, top_10
    """
    print("\n=== HIT PREDICTION DATASET ===")

    df = _download_raw_df(usecols=["title", "artist", "year", "lyrics", "language"])

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] >= HIT_START_YEAR) & (df["year"] <= HIT_END_YEAR)]
    print(f"After year filter ({HIT_START_YEAR}–{HIT_END_YEAR}): {df.shape}")

    df = df.dropna(subset=["title", "artist"])
    print(f"After dropping NaN title/artist: {df.shape}")

    # Normalise for matching
    billboard_df = pd.read_csv(BILLBOARD_PATH)

    df["title_clean"]           = df["title"].apply(clean_title)
    df["artist_clean"]          = df["artist"].apply(clean_artist)
    billboard_df["title_clean"] = billboard_df["title"].apply(clean_title)
    billboard_df["artist_clean"]= billboard_df["artist"].apply(clean_artist)

    n_genius           = len(df)
    n_billboard_unique = billboard_df[["title_clean", "artist_clean"]].drop_duplicates().shape[0]

    df = df.merge(
        billboard_df[["title_clean", "artist_clean", "rank", "top_10"]],
        on=["title_clean", "artist_clean"],
        how="left",
    )

    n_unique_matched = (
        df.loc[df["rank"].notna(), ["title_clean", "artist_clean"]]
          .drop_duplicates()
          .shape[0]
    )

    print("\nMERGE RESULTS")
    print(f"  Genius rows before merge:   {n_genius:,}")
    print(f"  Billboard unique songs:     {n_billboard_unique:,}")
    print(f"  Unique songs matched:       {n_unique_matched:,}")
    print(f"  Match rate:                 {n_unique_matched / n_billboard_unique * 100:.1f}%")

    df = df[["title", "artist", "lyrics", "rank", "top_10"]]
    df.to_csv(HIT_OUTPUT_PATH, index=False)
    print(f"\nSaved to {HIT_OUTPUT_PATH}  ({len(df):,} rows)")


# Entry point

if __name__ == "__main__":
    if RUN_PREPROCESSING:
        build_genre_dataset()
        build_hit_dataset()
    else:
        print("Preprocessing disabled (RUN_PREPROCESSING=False). Set to True to re-run.")