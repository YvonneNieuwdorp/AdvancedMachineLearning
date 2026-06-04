"""
feature_engineering.py
-----------------------
Extracts numerical features from raw song lyrics to complement TF-IDF.

Exposed:
    NUMERIC_FEATURES    list of feature column names (import this everywhere)
    add_lyric_features  adds all 5 features to a DataFrame in one call
"""

import re
import pandas as pd
from textblob import TextBlob

# Single source of truth for feature column names —
# imported by hit_prediction.py and models.py
NUMERIC_FEATURES = [
    "vocab_richness",
    "avg_word_len",
    "word_count",
    "sentiment",
    "rhyme_density",
]


def vocab_richness(lyrics: str) -> float:
    """
    Ratio of unique words to total words (vocabulary diversity).

    A higher score means the lyrics use a wider range of vocabulary.
    Typical range: 0.0 – 1.0.

    Args:
        lyrics: Raw or preprocessed lyrics string.

    Returns:
        Float between 0 and 1.
    """
    words = str(lyrics).split()
    return len(set(words)) / max(len(words), 1)


def avg_word_length(lyrics: str) -> float:
    """
    Average number of characters per word.

    Args:
        lyrics: Raw or preprocessed lyrics string.

    Returns:
        Float representing mean word length.
    """
    words = str(lyrics).split()
    return sum(len(w) for w in words) / max(len(words), 1)


def word_count(lyrics: str) -> int:
    """
    Total number of words in the lyrics.

    Args:
        lyrics: Raw or preprocessed lyrics string.

    Returns:
        Integer word count.
    """
    return len(str(lyrics).split())


def sentiment_score(lyrics: str) -> float:
    """
    Sentiment polarity of the lyrics using TextBlob.

    Range: -1.0 (very negative) to +1.0 (very positive).

    Args:
        lyrics: Raw lyrics string.

    Returns:
        Float polarity score.
    """
    return TextBlob(str(lyrics)).sentiment.polarity


def rhyme_density(lyrics: str) -> float:
    """
    Fraction of line-ending words that appear as an ending word on
    at least one other line (a simple proxy for rhyme density).

    Args:
        lyrics: Raw lyrics string, with lines separated by newlines.

    Returns:
        Float between 0.0 and 1.0.
    """
    lines = [l.strip() for l in str(lyrics).split("\n") if l.strip()]
    if len(lines) < 2:
        return 0.0
    endings = [line.split()[-1].lower() for line in lines if line.split()]
    matches = sum(
        1 for i, e in enumerate(endings)
        if e in endings[:i] or e in endings[i + 1:]
    )
    return matches / max(len(endings), 1)


def add_lyric_features(df: pd.DataFrame, lyrics_col: str = "lyrics") -> pd.DataFrame:
    """
    Adds all 5 numerical lyric features to a DataFrame.

    Computes features on the raw (unprocessed) lyrics column so that
    information lost during NLP cleaning (punctuation, line breaks,
    capitalisation) is still available for feature extraction.

    Args:
        df:         DataFrame containing a lyrics column.
        lyrics_col: Name of the column with raw lyrics. Default: 'lyrics'.

    Returns:
        Copy of df with 5 additional columns:
            vocab_richness, avg_word_len, word_count, sentiment, rhyme_density.
    """
    df = df.copy()
    df["vocab_richness"] = df[lyrics_col].apply(vocab_richness)
    df["avg_word_len"]   = df[lyrics_col].apply(avg_word_length)
    df["word_count"]     = df[lyrics_col].apply(word_count)
    df["sentiment"]      = df[lyrics_col].apply(sentiment_score)
    df["rhyme_density"]  = df[lyrics_col].apply(rhyme_density)
    return df
    