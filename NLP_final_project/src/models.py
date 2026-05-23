"""
models.py
---------
Creates and splits data for the three genre classification models.

Exposed functions:
    split_data(df)    — stratified 80/20 train/test split
    create_models()   — returns three fitted-ready sklearn Pipelines
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def split_data(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the preprocessed dataset into train and test sets.

    Uses an 80/20 split stratified by genre label to ensure proportional
    class representation in both sets.

    Args:
        df: Preprocessed DataFrame containing 'clean_lyrics' and 'tag' columns.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as pandas Series.
    """
    X = df["clean_lyrics"]
    y = df["tag"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def create_models() -> tuple[Pipeline, Pipeline, Pipeline]:
    """
    Creates three sklearn Pipelines, each with its own TF-IDF vectorizer
    and a classifier.

    Each pipeline uses:
        - TfidfVectorizer with unigrams + bigrams, max 10,000 features
        - A separate vectorizer instance per pipeline to avoid shared state

    Models:
        - Logistic Regression (balanced class weights, max 1000 iterations)
        - Multinomial Naive Bayes
        - LinearSVC (balanced class weights)

    Returns:
        Tuple of (log_pipeline, nb_pipeline, svm_pipeline).
    """
    def tfidf() -> TfidfVectorizer:
        return TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    log_pipeline = Pipeline([
        ("vectorizer", tfidf()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    nb_pipeline = Pipeline([
        ("vectorizer", tfidf()),
        ("model", MultinomialNB()),
    ])
    svm_pipeline = Pipeline([
        ("vectorizer", tfidf()),
        ("model", LinearSVC(class_weight="balanced")),
    ])
    return log_pipeline, nb_pipeline, svm_pipeline
