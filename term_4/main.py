# WEEK 2
# Billboard Hot 100 scraper
# Scrapes weekly Hot 100 charts from 8 years ago until 3 years ago
# Goal: create dataset for NLP + ML hit prediction project

# pip install -r requirements.txt

import re
import time
import nltk
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from tokenizers import Tokenizer # main object for tokenization
from tokenizers.models import BPE # model implementing Byte Pair Encoding.
from tokenizers.trainers import BpeTrainer # trains the BPE vocabulary
from tokenizers.pre_tokenizers import Whitespace # simple pre-tokenizer splitting on spaces
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import json
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import emoji
import string
from sklearn.model_selection import train_test_split

def preprocess_lyrics(df):
    nltk.download("stopwords")
    stop_words = stopwords.words("english")

    # drop rows with missing lyrics
    df = df.dropna(subset=["lyrics"]) 
    # remove bracketed sections like [Chorus], [Verse 1], etc.
    df["clean_lyrics"] = df["lyrics"].apply(lambda x: re.sub(r'\[.*?\]', '', x)) 
    # convert emojis to descriptive text
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda x: emoji.demojize(x)) 
    # lowercase 
    df["clean_lyrics"] = df["clean_lyrics"].str.lower()
    # verwijderen leestekens
    df['clean_lyrics'] = df['clean_lyrics'].apply(lambda x: "".join([c for c in x if c not in string.punctuation]))
    # whitespace normalization
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    # tokenization
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda x: x.split())
    # remove stopwords
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda tokens: [t for t in tokens if t.lower() not in stop_words])
    
    # check if tags are equally distributed across the dataset
    tag_counts = df["tag"].value_counts()
    print(f"Tag distribution: {tag_counts}")

    return df

def test_train_split_data(df_clean): 
    X = df_clean['clean_lyrics']
    y = df_clean['tag']

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state=42, stratify=y
    )

    print(f'tag train {y_train.value_counts()}')
    print(f'tag test {y_test.value_counts()}')

    return X_train, X_test, y_train, y_test

# Model Creation
def create_models():
    """
    Creates TF-IDF + classifier pipelines for Logistic Regression and Naive Bayes.

    Returns:
        tuple: (log_pipeline, nb_pipeline)
    """
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3)  # unigrams, bigrams, trigrams
    )

    # Logistic Regression pipeline
    log_pipeline = Pipeline([
        ("vectorizer", tfidf),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # Naive Bayes pipeline
    nb_pipeline = Pipeline([
        ("vectorizer", tfidf),
        ("model", MultinomialNB())
    ])

    return log_pipeline, nb_pipeline

# Main pipeline
def main():
    df = pd.read_csv("clean_song_lyrics.csv")
    df_clean = preprocess_lyrics(df)

    X_train, X_test, y_train, y_test = test_train_split_data(df_clean)
    
    log_model, nb_model = create_models()

    # Train models
    log_model.fit(X_train, y_train)
    nb_model.fit(X_train, y_train)


   

if __name__ == "__main__":
    main()
# stratified splitting, want pop is wel wat groter
# f1 score, macro f1 score, confusion matrix, ROC curve, precision-recall curve
# 