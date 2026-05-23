"""
preprocessing.py
----------------
Cleans and preprocesses raw song lyrics for use in ML models.

Exposed functions:
    explore_and_clean_data(df)  — deduplication and overview
    preprocess_lyrics(df)       — full NLP preprocessing pipeline
"""

import re
import string

import emoji
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def explore_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prints a summary of the raw dataset and removes duplicate rows.

    Removes exact duplicate rows first, then rows with identical lyrics
    (keeping the first occurrence).

    Args:
        df: Raw lyrics DataFrame loaded from CSV.

    Returns:
        Deduplicated DataFrame.
    """
    print("\n=== DATASET OVERVIEW ===")
    print(f"Shape:          {df.shape}")
    print(f"Columns:        {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["lyrics"])
    print(f"Shape after deduplication: {df.shape}")
    print(f"\nGenre distribution:\n{df['tag'].value_counts()}")
    return df


def preprocess_lyrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies NLP preprocessing to the lyrics column and stores the result
    in a new 'clean_lyrics' column.

    Steps applied in order:
        1. Drop rows with missing lyrics or genre tag
        2. Remove structural markers like [Chorus], [Verse 1], etc.
        3. Convert emoji to text descriptions
        4. Lowercase all text
        5. Remove punctuation
        6. Normalise whitespace
        7. Tokenise by splitting on whitespace
        8. Remove English stopwords
        9. Lemmatise each token
        10. Rejoin tokens into a single string

    Args:
        df: DataFrame containing at least 'lyrics' and 'tag' columns.

    Returns:
        DataFrame with an added 'clean_lyrics' column.
    """
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df = df.dropna(subset=["lyrics", "tag"]).copy()

    df["clean_lyrics"] = df["lyrics"].apply(lambda x: re.sub(r"\[.*?\]", "", x))
    df["clean_lyrics"] = df["clean_lyrics"].apply(emoji.demojize)
    df["clean_lyrics"] = df["clean_lyrics"].str.lower()
    df["clean_lyrics"] = df["clean_lyrics"].apply(
        lambda x: "".join(c for c in x if c not in string.punctuation)
    )
    df["clean_lyrics"] = df["clean_lyrics"].apply(
        lambda x: re.sub(r"\s+", " ", x).strip()
    )
    df["clean_lyrics"] = df["clean_lyrics"].apply(str.split)
    df["clean_lyrics"] = df["clean_lyrics"].apply(
        lambda tokens: [t for t in tokens if t not in stop_words]
    )
    df["clean_lyrics"] = df["clean_lyrics"].apply(
        lambda tokens: [lemmatizer.lemmatize(t) for t in tokens]
    )
    df["clean_lyrics"] = df["clean_lyrics"].apply(" ".join)

    print(f"\nGenre distribution after preprocessing:\n{df['tag'].value_counts()}")
    return df
