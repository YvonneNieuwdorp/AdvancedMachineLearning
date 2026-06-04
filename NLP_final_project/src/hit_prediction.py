"""
hit_prediction.py
-----------------
Predicts how well a song charted on the Billboard Year-End Hot 100
based on its lyrics alone, as a 3-class classification problem.

Labels assigned via a left join between the Genius lyrics dataset and
the scraped Billboard chart data (2003-2023):
    "top_10" : song appeared in the Top 10 at least once
    "top_100": song appeared in the chart but never reached the Top 10
    "none"   : song not found in the Billboard dataset at all

Pipeline steps:
    1. Left-join lyrics CSV with Billboard CSV on title_clean + artist_clean
    2. Assign 3-class chart_label from the merged top_10 column
    3. Reuse NLP preprocessing pipeline from preprocessing.py
    4. Stratified 80/20 train/test split
    5. Train Logistic Regression, Naive Bayes, and LinearSVC
    6. Evaluate with per-class precision/recall/F1 and macro F1
    7. Save confusion matrices + model comparison chart
    8. Print top discriminating words per class

Can be run standalone:
    python src/hit_prediction.py

Or called from main.py via run_hit_prediction_pipeline().
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import explore_and_clean_data, preprocess_lyrics

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BILLBOARD_PATH = os.path.join(BASE_DIR, "data", "billboard_year_end_2003_2023.csv")
LYRIC_PATH     = os.path.join(BASE_DIR, "data", "clean_song_lyrics.csv")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")

# Ordered from least to most successful: used consistently for all outputs
CLASS_ORDER = ["none", "top_100", "top_10"]


# Step 1: Merge and label

def build_hit_dataset(
    lyrics_path: str = LYRIC_PATH,
    billboard_path: str = BILLBOARD_PATH,
) -> pd.DataFrame:
    """
    Merges the lyrics CSV with the Billboard CSV to attach a 3-class
    chart_label to every song.

    Matching is performed on title_clean + artist_clean. The Billboard CSV
    is first deduplicated so each unique song contributes one row (taking the
    maximum top_10 value across years: a song counts as top_10 if it ever
    reached that threshold). A left join is then used so that songs present
    in the lyrics dataset but absent from Billboard receive the label "none",
    representing songs that never charted in the Hot 100 at all.

    Label assignment:
        "top_10" : matched in Billboard and top_10 == 1
        "top_100": matched in Billboard and top_10 == 0
        "none"   : no Billboard match found

    If title_clean / artist_clean columns are absent from the Billboard CSV
    (i.e. the raw scraped file is used rather than the cleaned one), they are
    derived on the fly using the same normalisation functions from cleaning.py.

    Args:
        lyrics_path:    Path to clean_song_lyrics.csv.
        billboard_path: Path to billboard_year_end_2003_2023.csv.

    Returns:
        DataFrame containing all lyrics columns plus a new 'chart_label'
        column with values in {"none", "top_100", "top_10"}.
    """
    lyrics_df = pd.read_csv(lyrics_path)
    billboard_df = pd.read_csv(billboard_path)

    print(f"Lyrics CSV: {lyrics_df.shape[0]} rows")
    print(f"Billboard CSV: {billboard_df.shape[0]} rows")

    # Derive normalised join keys if the raw scraped CSV is used
    if "title_clean" not in billboard_df.columns:
        from cleaning import clean_artist, clean_title
        billboard_df["title_clean"]  = billboard_df["title"].apply(clean_title)
        billboard_df["artist_clean"] = billboard_df["artist"].apply(clean_artist)

    # One row per unique song; top_10 = 1 if it ever reached the Top 10
    billboard_dedup = (
        billboard_df
        .groupby(["title_clean", "artist_clean"], as_index=False)["top_10"]
        .max()
    )
    print(f"Unique Billboard songs (after deduplication): {len(billboard_dedup)}")

    # Left join: unmatched lyrics rows receive None in the top_10 column
    merged = lyrics_df.merge(
        billboard_dedup[["title_clean", "artist_clean", "top_10"]],
        on=["title_clean", "artist_clean"],
        how="left",
    )

    # Assign the 3-class label
    def _assign_label(top_10_val: float) -> str:
        if pd.isna(top_10_val):
            return "none"
        return "top_10" if int(top_10_val) == 1 else "top_100"

    merged["chart_label"] = merged["top_10"].apply(_assign_label)

    print(f"\nMERGE RESULTS")
    print(f"Total rows: {len(merged)}")
    print(f"\nClass distribution:\n{merged['chart_label'].value_counts()}")
    print(
        f"\nClass balance (%):\n"
        f"{merged['chart_label'].value_counts(normalize=True).mul(100).round(1)}"
    )

    return merged


# Step 2: Train / test split

def split_hit_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Performs a stratified 80/20 train/test split on the chart_label column.

    Stratification ensures all three classes are proportionally represented
    in both sets, which is especially important given the natural imbalance
    between "top_10" (~10% of charted songs) and the other classes.

    Args:
        df:           Preprocessed DataFrame with 'clean_lyrics' and
                      'chart_label' columns.
        test_size:    Fraction of data reserved for testing. Defaults to 0.2.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as pandas Series.
    """
    X = df["clean_lyrics"]
    y = df["chart_label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# Step 3: Model definitions

def create_hit_models() -> tuple[Pipeline, Pipeline, Pipeline]:
    """
    Creates three multi-class classification pipelines, each combining a
    TF-IDF vectoriser with a different classifier.

    Each pipeline uses unigrams and bigrams with a vocabulary capped at
    10,000 features. class_weight='balanced' is applied where the classifier
    supports it, compensating for the unequal class frequencies without
    requiring manual resampling.

    Models:
        - Logistic Regression (balanced class weights, max 1000 iterations)
        - Multinomial Naive Bayes (no native class_weight; compensated via
          sample_weight at fit time inside evaluate_hit_models)
        - LinearSVC (balanced class weights, max 2000 iterations)

    Returns:
        Tuple of (log_pipeline, nb_pipeline, svm_pipeline), all unfitted.
    """
    def tfidf() -> TfidfVectorizer:
        return TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    log_pipeline = Pipeline([
        ("vectorizer", tfidf()),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])

    nb_pipeline = Pipeline([
        ("vectorizer", tfidf()),
        # MultinomialNB has no class_weight parameter; imbalance is handled
        # via per-sample weights passed at fit time in evaluate_hit_models().
        ("model", MultinomialNB()),
    ])

    svm_pipeline = Pipeline([
        ("vectorizer", tfidf()),
        ("model", LinearSVC(
            class_weight="balanced", random_state=42, max_iter=2000
        )),
    ])

    return log_pipeline, nb_pipeline, svm_pipeline


# Step 4: Helpers

def _compute_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Computes per-sample weights inversely proportional to class frequency.

    Used to compensate for class imbalance in Multinomial Naive Bayes, which
    does not expose a class_weight parameter. The weight for each sample is:
        total_samples / (n_classes * samples_in_class)

    Args:
        y: Series of class labels for the training set.

    Returns:
        NumPy array of float weights, one per sample.
    """
    class_counts = y.value_counts()
    n_classes    = len(class_counts)
    total        = len(y)
    return y.map(lambda lbl: total / (n_classes * class_counts[lbl])).values


def _save_confusion_matrix(
    y_test: pd.Series,
    pred: np.ndarray,
    model_name: str,
    output_dir: str,
) -> None:
    """
    Saves a labelled confusion matrix heatmap as a PNG file.

    The matrix rows and columns follow CLASS_ORDER so the layout is
    consistent across all three models.

    Args:
        y_test:      True chart_label values for the test set.
        pred:        Predicted chart_label values.
        model_name:  Human-readable model name used in the title and filename.
        output_dir:  Directory to save the PNG.
    """
    cm  = confusion_matrix(y_test, pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {model_name}\n(Chart Position Prediction)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    fname    = model_name.lower().replace(" ", "_")
    out_path = os.path.join(output_dir, f"hit_confusion_matrix_{fname}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {out_path}")


def _plot_hit_model_comparison(
    results: dict[str, dict[str, float]],
    output_dir: str,
) -> None:
    """
    Saves a grouped bar chart comparing macro F1 and per-class F1 scores
    across all three models.

    Args:
        results:    Dict of {model_name: {metric_name: float}} as returned
                    by evaluate_hit_models().
        output_dir: Directory to save the PNG.
    """
    names        = list(results.keys())
    macro_f1     = [results[n]["macro_f1"]    for n in names]
    f1_top10     = [results[n]["f1_top_10"]   for n in names]
    f1_top100    = [results[n]["f1_top_100"]  for n in names]
    f1_none      = [results[n]["f1_none"]     for n in names]

    x     = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - 1.5 * width, macro_f1,  width, label="Macro F1",        color="#4C8BE2")
    b2 = ax.bar(x - 0.5 * width, f1_top10,  width, label="F1 (top_10)",     color="#E2714C")
    b3 = ax.bar(x + 0.5 * width, f1_top100, width, label="F1 (top_100)",    color="#4CBE82")
    b4 = ax.bar(x + 1.5 * width, f1_none,   width, label="F1 (none)",       color="#BE4C82")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Chart Position Prediction: Model Comparison")
    ax.legend()
    for bars in (b1, b2, b3, b4):
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "hit_model_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Model comparison chart saved -> {out_path}")


# Step 5: Evaluation

def evaluate_hit_models(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    log_model: Pipeline,
    nb_model: Pipeline,
    svm_model: Pipeline,
    output_dir: str = OUTPUT_DIR,
) -> dict[str, dict[str, float]]:
    """
    Fits all three models, evaluates them on the test set, and saves
    visualisations to output_dir.

    Naive Bayes is fitted with per-sample weights (derived from training
    label frequencies) to compensate for its lack of a class_weight
    parameter. Logistic Regression and LinearSVC use class_weight='balanced'
    set at construction time.

    For each model the following are printed to stdout:
        - Per-class precision, recall, and F1
        - Macro F1 across all three classes
        - Full sklearn classification report

    Saved to output_dir:
        - hit_confusion_matrix_{model}.png  (one per model)
        - hit_model_comparison.png          (grouped bar chart)

    Args:
        X_train:    Training lyrics (preprocessed strings).
        X_test:     Test lyrics (preprocessed strings).
        y_train:    Training chart_label values.
        y_test:     Test chart_label values.
        log_model:  Unfitted Logistic Regression pipeline.
        nb_model:   Unfitted Naive Bayes pipeline.
        svm_model:  Unfitted LinearSVC pipeline.
        output_dir: Directory to save PNG outputs.

    Returns:
        Dict of {model_name: {"macro_f1": float, "f1_top_10": float,
        "f1_top_100": float, "f1_none": float, "precision_top_10": float,
        "recall_top_10": float}}.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Fit models: Naive Bayes needs explicit sample weights
    sample_weights = _compute_sample_weights(y_train)
    log_model.fit(X_train, y_train)
    nb_model.fit(X_train, y_train, model__sample_weight=sample_weights)
    svm_model.fit(X_train, y_train)

    models = {
        "Logistic Regression": log_model,
        "Naive Bayes":         nb_model,
        "LinearSVC":           svm_model,
    }

    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        pred   = model.predict(X_test)
        report = classification_report(
            y_test, pred,
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        )

        macro_f1     = f1_score(y_test, pred, average="macro",    zero_division=0)
        f1_top10     = report["top_10"]["f1-score"]
        f1_top100    = report["top_100"]["f1-score"]
        f1_none      = report["none"]["f1-score"]
        prec_top10   = report["top_10"]["precision"]
        recall_top10 = report["top_10"]["recall"]

        results[name] = {
            "macro_f1":        macro_f1,
            "f1_top_10":       f1_top10,
            "f1_top_100":      f1_top100,
            "f1_none":         f1_none,
            "precision_top_10": prec_top10,
            "recall_top_10":   recall_top10,
        }

        print(f"\n{'=' * 45}")
        print(f"  {name}")
        print(f"{'=' * 45}")
        print(f"  Macro F1:                    {macro_f1:.4f}")
        print(f"  F1 : top_10:               {f1_top10:.4f}")
        print(f"  F1 : top_100:              {f1_top100:.4f}")
        print(f"  F1 : none:                 {f1_none:.4f}")
        print(f"  Precision (top_10 class):   {prec_top10:.4f}")
        print(f"  Recall    (top_10 class):   {recall_top10:.4f}")
        print(f"\nFull classification report:")
        print(classification_report(
            y_test, pred,
            labels=CLASS_ORDER,
            zero_division=0,
        ))

        _save_confusion_matrix(y_test, pred, name, output_dir)

    _plot_hit_model_comparison(results, output_dir)

    return results


# Step 6: Interpretability

def show_top_hit_words(model: Pipeline, n: int = 15) -> None:
    """
    Prints the n words most strongly associated with each chart class for
    a LinearSVC or Logistic Regression model.

    For multi-class LinearSVC, coef_ has shape (n_classes, n_features) with
    one coefficient vector per class. The top-n features by coefficient
    magnitude are the words most predictive of that class.

    Args:
        model: Fitted sklearn Pipeline containing 'vectorizer' and 'model'
               steps. The model step must expose a coef_ attribute
               (LinearSVC or Logistic Regression).
        n:     Number of top words to display per class. Defaults to 15.
    """
    feature_names = model.named_steps["vectorizer"].get_feature_names_out()
    coef          = model.named_steps["model"].coef_
    classes       = model.named_steps["model"].classes_

    print("\nTOP WORDS PER CHART CLASS")
    for i, label in enumerate(classes):
        print(f"\n {label.upper()} ")
        top_idx = coef[i].argsort()[-n:][::-1]
        for idx in top_idx:
            print(f"  {feature_names[idx]:<25} coef={coef[i][idx]:.4f}")



# Step 7: Single-song prediction
 
def predict_hit(lyrics: str, model: Pipeline) -> tuple[str, dict[str, float] | None]:
    """
    Predicts the chart class for a single song given its raw lyrics.

    Wraps the vectorisation and prediction steps so callers do not need to
    interact with the pipeline directly.

    Args:
        lyrics: Raw, unprocessed song lyrics as a plain string.
        model:  A fitted sklearn Pipeline (vectorizer + classifier).

    Returns:
        Tuple of (predicted_label, class_probabilities).
        predicted_label is one of "none", "top_100", or "top_10".
        class_probabilities is a dict mapping each class to its probability,
        or None for models that do not support predict_proba (e.g. LinearSVC).
    """
    label = str(model.predict([lyrics])[0])
    probs = None
    if hasattr(model, "predict_proba"):
        raw   = model.predict_proba([lyrics])[0]
        probs = dict(zip(model.classes_, raw))
    return label, probs

 
# Full standalone pipeline
 
def run_hit_prediction_pipeline() -> None:
    """
    Runs the full 3-class chart position prediction pipeline end-to-end.

    Executes all steps in order: dataset construction, preprocessing,
    splitting, model training, evaluation, interpretability analysis, and
    a short demo. Outputs (confusion matrices, comparison chart) are saved
    to the outputs/ directory.

    Can be called from main.py or run standalone via __main__.
    """

    # Step 1: Build labelled dataset via left join
    print("\n Step 1: Merging lyrics with Billboard labels")
    df = build_hit_dataset()

    # Step 2: Deduplicate
    print("\n Step 2: Deduplicating")
    df = explore_and_clean_data(df)

    # Step 3: NLP preprocessing
    print("\n Step 3: Preprocessing lyrics")
    df = preprocess_lyrics(df)

    # Step 4: Train/test split
    print("\n Step 4: Splitting data")
    X_train, X_test, y_train, y_test = split_hit_data(df)
    print(f"  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")
    print(f"\n  Train class distribution:\n{y_train.value_counts().to_string()}")
    print(f"\n  Test  class distribution:\n{y_test.value_counts().to_string()}")

    # Step 5: Create models
    print("\n Step 5: Creating models")
    log_model, nb_model, svm_model = create_hit_models()

    # Step 6: Train and evaluate
    print("\n Step 6: Training and evaluating models")
    evaluate_hit_models(
        X_train, X_test, y_train, y_test,
        log_model, nb_model, svm_model,
    )

    # Step 7: Interpretability
    print("\n Step 7: Top discriminating words per class (LinearSVC)")
    show_top_hit_words(svm_model)

    # Step 8: Demo predictions
    print("\n Step 8: Demo predictions (LinearSVC)")
    demo_songs = [
        ("I used to pray for times like this, to rhyme like this, so long ago", "top_10"),
        ("We found love in a hopeless place, shining in the dark",              "top_10"),
        ("Hey mama, I know I act a fool but got your name tattooed on my arm",  "top_100"),
        ("This is an obscure deep cut that never got radio play at all",        "none"),
    ]
    for lyrics, true_label in demo_songs:
        label, probs = predict_hit(lyrics, svm_model)
        probs_str = (
            "  |  ".join(f"{k}: {v:.2f}" for k, v in probs.items())
            if probs else "N/A"
        )
        print(f"  Lyrics   : \"{lyrics[:55]}...\"")
        print(f"  Predicted: {label:<8}  |  Actual: {true_label}")
        if probs:
            print(f"  Probs    : {probs_str}")
        print()

    print("Chart position prediction pipeline complete. Outputs saved to outputs/")


if __name__ == "__main__":
    run_hit_prediction_pipeline()