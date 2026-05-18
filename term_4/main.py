# WEEK 2
# Billboard Hot 100 scraper
# NLP + ML hit prediction project

# pip install -r requirements.txt

import re
import nltk
import pandas as pd
import numpy as np
import emoji
import string

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

def explore_and_clean_data(df):
    print("dataset")

    # Shape
    print("\nDataset shape:")
    print(df.shape)

    # Columns
    print("\nColumns:")
    print(df.columns)

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Duplicate rows
    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # Remove duplicates
    df = df.drop_duplicates()

    print("\nShape after removing duplicates:")
    print(df.shape)

    # Remove duplicate lyrics
    df = df.drop_duplicates(subset=["lyrics"])

    print("\nShape after removing duplicate lyrics:")
    print(df.shape)

    # Genre distribution
    print("\nGenre distribution:")
    print(df["tag"].value_counts())

    return df

def preprocess_lyrics(df):

    nltk.download("stopwords")

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # Remove missing lyrics
    df = df.dropna(subset=["lyrics", "tag"])
    # Remove [Chorus], [Verse], etc.
    df["clean_lyrics"] = df["lyrics"].apply(lambda x: re.sub(r"\[.*?\]", "", x))
    # Emoji to text
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda x: emoji.demojize(x))
    # Lowercase
    df["clean_lyrics"] = df["clean_lyrics"].str.lower()
    # Remove punctuation
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda x: "".join([c for c in x if c not in string.punctuation]))
    # Normalize whitespace
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    # Tokenization
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda x: x.split())
    # Remove stopwords
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda tokens: [t for t in tokens if t not in stop_words])
    # Lemmatization
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda tokens: [lemmatizer.lemmatize(t) for t in tokens])
    # Back to string
    df["clean_lyrics"] = df["clean_lyrics"].apply(lambda tokens: " ".join(tokens))

    print("\nTag distribution:")
    print(df["tag"].value_counts())

    return df

def test_train_split_data(df_clean):
    X = df_clean["clean_lyrics"]
    y = df_clean["tag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def create_models():

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    # Logistic Regression
    log_pipeline = Pipeline([
        ("vectorizer", tfidf),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    # Naive Bayes
    nb_pipeline = Pipeline([
        ("vectorizer", tfidf),
        ("model", MultinomialNB())
    ])

    svm_pipline = Pipeline([
        ("vectorizer", tfidf), 
        ("model", LinearSVC(class_weight="balanced"))
    ])

    return log_pipeline, nb_pipeline, svm_pipline

def evaluation(X_test, y_test, log_model, nb_model, svm_model):
    log_pred = log_model.predict(X_test)

    print("Logistic Regressson")
    print("Accuracy:")
    print(accuracy_score(y_test, log_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, log_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, log_pred))


    nb_pred = nb_model.predict(X_test)

    print("Naive Bayes")
    print("Accuracy:")
    print(accuracy_score(y_test, nb_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, nb_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, nb_pred))

    svm_pred = svm_model.predict(X_test)
    print("LinearSVC")
    print("Accuracy:", accuracy_score(y_test, svm_pred))
    print("Macro F1:", f1_score(y_test, svm_pred, average="macro"))

    print("\nClassification Report:")
    print(classification_report(y_test, svm_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, svm_pred))


def show_top_words(model, n=3): 
    feature_names = model.named_steps["vectorizer"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_
    classes = model.named_steps["model"].classes_

    for i, genre in enumerate(classes): 
        top_indices = coefficients[i].argsort()[-n:]
        print(f"Top words for {genre}")
        for idx in reversed(top_indices): 
            print(feature_names[idx])


def main():
    df = pd.read_csv("clean_song_lyrics.csv")

    explore_and_clean_data(df)

    df_clean = preprocess_lyrics(df)

    X_train, X_test, y_train, y_test = test_train_split_data(df_clean)

    log_model, nb_model, svm_model = create_models()
    log_model.fit(X_train, y_train)
    nb_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)


    # Evaluate
    evaluation(X_test, y_test, log_model, nb_model, svm_model)

    show_top_words(svm_model)


if __name__ == "__main__":
    main()