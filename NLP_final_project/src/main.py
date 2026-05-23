"""
main.py
-
Entry point for the genre classification pipeline.

Run this file to execute the full pipeline end-to-end:
    python src/main.py

Pipeline steps:
    1. (Optional) Scrape Billboard chart data       -> scraper.py
    2. (Optional) Download and clean lyrics data    -> cleaning.py
    3. Load and deduplicate dataset                 -> preprocessing.py
    4. Preprocess lyrics (NLP)                      -> preprocessing.py
    5. Train/test split                             -> models.py
    6. Train three classifiers                      -> models.py
    7. Evaluate and save visualisations             -> evaluation.py
    8. Print top words per genre                    -> evaluation.py

Steps 1 and 2 are skipped by default to avoid re-running long operations.
Set RUN_SCRAPER or RUN_PREPROCESSING to True in their respective modules,
or use the flags below to trigger them from here.
"""

import os
import pandas as pd

from scraper import scrape_chart_data, RUN_SCRAPER, BILLBOARD_OUTPUT_PATH
from checker import check_billboard_dataset
from cleaning import download_and_clean_data, RUN_PREPROCESSING, LYRIC_OUTPUT_PATH
from preprocessing import explore_and_clean_data, preprocess_lyrics
from models import split_data, create_models
from evaluation import evaluation, show_top_words, predict_genre

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Override flags here if you want to re-run data acquisition from main.py

FORCE_SCRAPE         = False   # Set True to re-scrape Billboard data
FORCE_PREPROCESSING  = False   # Set True to re-download and clean lyrics
VALIDATE_BILLBOARD   = False   # Set True to print Billboard dataset checks


def main() -> None:
    """
    Runs the full genre classification pipeline end-to-end.
    """

    # Step 1: Billboard scraper (skipped if CSV already exists)
    if FORCE_SCRAPE or (RUN_SCRAPER and not os.path.exists(BILLBOARD_OUTPUT_PATH)):
        print("\n Step 1: Scraping Billboard data")
        scrape_chart_data()
    else:
        print(f"\n Step 1: Skipped (using existing {BILLBOARD_OUTPUT_PATH})")

    if VALIDATE_BILLBOARD:
        print("\n Step 1b: Validating Billboard dataset")
        check_billboard_dataset(BILLBOARD_OUTPUT_PATH)

    # Step 2: Lyrics cleaning (skipped if CSV already exists)
    if FORCE_PREPROCESSING or (RUN_PREPROCESSING and not os.path.exists(LYRIC_OUTPUT_PATH)):
        print("\n Step 2: Downloading and cleaning lyrics")
        download_and_clean_data()
    else:
        print(f"\n Step 2: Skipped (using existing {LYRIC_OUTPUT_PATH})")

    # Step 3: Load and deduplicate
    print("\n Step 3: Loading dataset")
    df = pd.read_csv(LYRIC_OUTPUT_PATH)
    df = explore_and_clean_data(df)

    # Step 4: NLP preprocessing
    print("\n Step 4: Preprocessing lyrics")
    df_clean = preprocess_lyrics(df)

    # Step 5: Train/test split
    print("\n Step 5: Splitting data")
    X_train, X_test, y_train, y_test = split_data(df_clean)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Step 6: Train models
    print("\n Step 6: Training models")
    log_model, nb_model, svm_model = create_models()
    log_model.fit(X_train, y_train)
    print("  Logistic Regression trained")
    nb_model.fit(X_train, y_train)
    print("  Naive Bayes trained")
    svm_model.fit(X_train, y_train)
    print("  LinearSVC trained")

    # Step 7: Evaluate
    print("\n Step 7: Evaluating models")
    evaluation(X_test, y_test, log_model, nb_model, svm_model)

    # Step 8: Interpretability
    print("\n Step 8: Top words per genre")
    show_top_words(svm_model)

    print("\nPipeline complete. Outputs saved to outputs/")
    
    # Step 9: Prototype demo: predict genre for new lyrics
    print("\n Step 9: Prototype demo")
    examples = [
        ("I used to pray for times like this, to rhyme like this", "rap"),
        ("We found love in a hopeless place, shining in the dark", "pop"),
        ("Pour some sugar on me, in the name of love",             "rock"),
    ]
    for lyrics, true_label in examples:
        prediction = predict_genre(lyrics, svm_model)
        print(f"  Lyrics : \"{lyrics[:50]}...\"")
        print(f"  Predicted: {prediction} | Actual: {true_label}\n")


if __name__ == "__main__":
    main()
