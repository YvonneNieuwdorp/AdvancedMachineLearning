"""
cleaning.py
-----------
Downloads the Genius lyrics dataset, filters it, and merges it with the
Billboard chart data to produce the final training dataset.

Can be run standalone:
    python src/cleaning.py

Output: data/clean_song_lyrics.csv
"""

import os
import glob
import re
import pandas as pd
import kagglehub

RUN_PREPROCESSING = False  # Set to True to re-run the full pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BILLBOARD_PATH     = os.path.join(BASE_DIR, "data", "billboard_year_end_2003_2023.csv")
LYRIC_OUTPUT_PATH  = os.path.join(BASE_DIR, "data", "clean_song_lyrics.csv")
KAGGLE_DATASET = "carlosgdcj/genius-song-lyrics-with-language-information"
START_YEAR = 2003
END_YEAR = 2023


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


def download_and_clean_data() -> None:
    """
    Runs the full data acquisition and cleaning pipeline.

    Downloads the Genius lyrics dataset from Kaggle, applies filtering
    and normalisation, merges with the Billboard chart data, and saves
    the result to LYRIC_OUTPUT_PATH.

    Raises:
        FileNotFoundError: If no CSV is found in the downloaded Kaggle dataset.
    """
    print("Starting Kaggle dataset download...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Dataset downloaded to cache: {path}")

    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found at {path}")

    raw_path = csv_files[0]
    print(f"CSV file found: {raw_path}")

    df = pd.read_csv(raw_path, usecols=["title", "tag", "artist", "year", "lyrics", "language"])
    print("Data loaded!")

    print("\nCleaning data...")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["language"] == "en"]
    print(f"After language filter: {df.shape}")
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]
    print(f"After year filter:     {df.shape}")
    df = df[df["tag"] != "misc"]
    print(f"After genre filter:    {df.shape}")
    df = df.dropna(subset=["year", "lyrics", "title", "artist"])
    print(f"After dropping NaNs:   {df.shape}")

    billboard_df = pd.read_csv(BILLBOARD_PATH)

    df["title_clean"] = df["title"].apply(clean_title)
    df["artist_clean"] = df["artist"].apply(clean_artist)
    billboard_df["title_clean"] = billboard_df["title"].apply(clean_title)
    billboard_df["artist_clean"] = billboard_df["artist"].apply(clean_artist)

    n_genius = len(df)
    n_billboard_unique = billboard_df[["title_clean", "artist_clean"]].drop_duplicates().shape[0]

    df = df.merge(
        billboard_df[["title_clean", "artist_clean", "rank", "top_10"]],
        on=["title_clean", "artist_clean"],
        how="left"
    )

    n_unique_matched = df[["title_clean", "artist_clean"]].drop_duplicates().shape[0]

    print("\n=== MERGE RESULTS ===")
    print(f"Genius rows before merge:      {n_genius}")
    print(f"Billboard unique songs:        {n_billboard_unique}")
    print(f"Unique songs matched:          {n_unique_matched}")
    print(f"Match rate:                    {n_unique_matched / n_billboard_unique * 100:.1f}%")

    df.to_csv(LYRIC_OUTPUT_PATH, index=False)
    print(f"\nSaved to {LYRIC_OUTPUT_PATH}")


if __name__ == "__main__":
    if RUN_PREPROCESSING:
        download_and_clean_data()
    else:
        print("Preprocessing disabled (RUN_PREPROCESSING=False). Set to True to re-run.")
