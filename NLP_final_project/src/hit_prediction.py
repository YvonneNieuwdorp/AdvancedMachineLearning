"""
hit_prediction.py
-----------------
Predicts how well a song charted on the Billboard Year-End Hot 100
based on its lyrics alone, as a 2-class classification problem.

Labels assigned via a left join between the Genius lyrics dataset and
the scraped Billboard chart data (2003-2023):
    "top_10" : song appeared in the Top 10 at least once
    "top_100": song appeared in the chart but never reached the Top 10

The "none" class (songs not on Billboard) is excluded because
clean_song_lyrics.csv was already built via an inner join on Billboard,
meaning there are no genuine non-charting songs in the dataset.

Features used:
    - TF-IDF on preprocessed lyrics (unigrams + bigrams, 10k features)
    - 5 numerical features: vocab_richness, avg_word_len, word_count,
      sentiment, rhyme_density

Pipeline steps:
    1. Left-join lyrics CSV with Billboard CSV on title_clean + artist_clean
    2. Assign 2-class chart_label from the merged top_10 column
    3. Add numerical lyric features via feature_engineering.py
    4. Reuse NLP preprocessing pipeline from preprocessing.py
    5. Stratified 80/20 train/test split
    6. Train Logistic Regression, Naive Bayes, and LinearSVC
    7. Evaluate with per-class precision/recall/F1 and macro F1
    8. Save confusion matrices + model comparison chart
    9. Print top discriminating words per class

Can be run standalone:
    python src/hit_prediction.py

Or called from main.py via run_hit_prediction_pipeline().
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from feature_engineering import add_lyric_features, NUMERIC_FEATURES
from preprocessing import explore_and_clean_data, preprocess_lyrics

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BILLBOARD_PATH = os.path.join(BASE_DIR, "data", "billboard_year_end_2003_2023.csv")
LYRIC_PATH     = os.path.join(BASE_DIR, "data", "clean_song_lyrics.csv")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")

# 2-class setup: top_10 vs top_100
CLASS_ORDER = ["top_100", "top_10"]


# ── Sklearn transformer helpers ───────────────────────────────────────────────

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a single column from a DataFrame and returns it as a Series."""

    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column]


class NumericFeatureSelector(BaseEstimator, TransformerMixin):
    """Selects numeric feature columns from a DataFrame as a numpy array."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].values


# ── Step 1: Merge and label ───────────────────────────────────────────────────

def build_hit_dataset(
    lyrics_path: str = LYRIC_PATH,
    billboard_path: str = BILLBOARD_PATH,
) -> pd.DataFrame:
    """
    Merges the lyrics CSV with the Billboard CSV to attach a 2-class
    chart_label to every song, then adds numerical lyric features.

    Matching is performed on title_clean + artist_clean. The Billboard CSV
    is deduplicated so each unique song contributes one row (taking the
    maximum top_10 value across years). A left join is used, and songs
    with no Billboard match (label "none") are dropped because the lyrics
    CSV was originally built via an inner join on Billboard — there are no
    genuine non-charting songs in the dataset.

    Label assignment:
        "top_10" : matched in Billboard and top_10 == 1
        "top_100": matched in Billboard and top_10 == 0

    Numerical features added (from feature_engineering.py):
        vocab_richness, avg_word_len, word_count, sentiment, rhyme_density

    Args:
        lyrics_path:    Path to clean_song_lyrics.csv.
        billboard_path: Path to billboard_year_end_2003_2023.csv.

    Returns:
        DataFrame containing all lyrics columns, a 'chart_label' column,
        and 5 numerical feature columns.
    """
    lyrics_df    = pd.read_csv(lyrics_path)
    billboard_df = pd.read_csv(billboard_path)

    print(f"Lyrics CSV:    {lyrics_df.shape[0]} rows")
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

    # Left join: unmatched rows receive NaN in top_10
    merged = lyrics_df.merge(
        billboard_dedup[["title_clean", "artist_clean", "top_10"]],
        on=["title_clean", "artist_clean"],
        how="left",
    )

    # Assign labels, then drop "none" (no genuine non-charting songs exist)
    def _assign_label(top_10_val: float) -> str:
        if pd.isna(top_10_val):
            return "none"
        return "top_10" if int(top_10_val) == 1 else "top_100"

    merged["chart_label"] = merged["top_10"].apply(_assign_label)

    before = len(merged)
    merged = merged[merged["chart_label"] != "none"].copy()
    print(f"\nDropped {before - len(merged)} 'none' rows (no genuine non-charting songs)")

    print(f"\nMERGE RESULTS")
    print(f"Total rows: {len(merged)}")
    print(f"\nClass distribution:\n{merged['chart_label'].value_counts()}")
    print(
        f"\nClass balance (%):\n"
        f"{merged['chart_label'].value_counts(normalize=True).mul(100).round(1)}"
    )

    # Add numerical lyric features (computed on raw lyrics before NLP cleaning)
    print("\nAdding numerical lyric features...")
    merged = add_lyric_features(merged, lyrics_col="lyrics")
    print(f"Features added: {NUMERIC_FEATURES}")

    return merged


# ── Step 2: Train / test split ────────────────────────────────────────────────

def split_hit_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs a stratified 80/20 train/test split on the chart_label column.

    X is a DataFrame containing both 'clean_lyrics' (for TF-IDF) and the
    5 numerical feature columns (for the numeric branch of the pipeline).
    Naive Bayes only uses the 'clean_lyrics' column; LogReg and SVM use all.

    Args:
        df:           Preprocessed DataFrame with 'clean_lyrics', the 5
                      numerical feature columns, and 'chart_label'.
        test_size:    Fraction of data reserved for testing. Defaults to 0.2.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
        X_train / X_test are DataFrames; y_train / y_test are Series.
    """
    feature_cols = ["clean_lyrics"] + NUMERIC_FEATURES
    X = df[feature_cols]
    y = df["chart_label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ── Step 3: Model definitions ─────────────────────────────────────────────────

def create_hit_models() -> tuple[Pipeline, Pipeline, Pipeline]:
    """
    Creates three multi-class classification pipelines.

    Logistic Regression and LinearSVC use a FeatureUnion that combines:
        - TF-IDF on 'clean_lyrics' (sparse, 10k features, unigrams+bigrams)
        - StandardScaled numerical features (vocab_richness, avg_word_len,
          word_count, sentiment, rhyme_density)

    Multinomial Naive Bayes only uses TF-IDF on 'clean_lyrics' because
    StandardScaler can produce negative values, which MultinomialNB does
    not support.

    class_weight='balanced' is applied to LogReg and LinearSVC to compensate
    for the ~10% / ~90% split between top_10 and top_100. Naive Bayes uses
    per-sample weights passed at fit time in evaluate_hit_models().

    Returns:
        Tuple of (log_pipeline, nb_pipeline, svm_pipeline), all unfitted.
    """
    def combined_features() -> FeatureUnion:
        return FeatureUnion([
            ("tfidf", Pipeline([
                ("selector",   ColumnSelector("clean_lyrics")),
                ("vectorizer", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ])),
            ("numeric", Pipeline([
                ("selector", NumericFeatureSelector(NUMERIC_FEATURES)),
                ("scaler",   StandardScaler()),
            ])),
        ])

    log_pipeline = Pipeline([
        ("features", combined_features()),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])

    # Naive Bayes: TF-IDF only (no negative values allowed)
    nb_pipeline = Pipeline([
        ("selector",   ColumnSelector("clean_lyrics")),
        ("vectorizer", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("model",      MultinomialNB()),
    ])

    svm_pipeline = Pipeline([
        ("features", combined_features()),
        ("model", LinearSVC(
            class_weight="balanced", random_state=42, max_iter=2000
        )),
    ])

    return log_pipeline, nb_pipeline, svm_pipeline


# ── Step 4: Helpers ───────────────────────────────────────────────────────────

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

    Args:
        y_test:      True chart_label values for the test set.
        pred:        Predicted chart_label values.
        model_name:  Human-readable model name used in the title and filename.
        output_dir:  Directory to save the PNG.
    """
    cm  = confusion_matrix(y_test, pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
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
        results:    Dict of {model_name: {metric_name: float}}.
        output_dir: Directory to save the PNG.
    """
    names     = list(results.keys())
    macro_f1  = [results[n]["macro_f1"]   for n in names]
    f1_top10  = [results[n]["f1_top_10"]  for n in names]
    f1_top100 = [results[n]["f1_top_100"] for n in names]

    x     = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width, macro_f1,  width, label="Macro F1",     color="#4C8BE2")
    b2 = ax.bar(x,         f1_top10,  width, label="F1 (top_10)",  color="#E2714C")
    b3 = ax.bar(x + width, f1_top100, width, label="F1 (top_100)", color="#4CBE82")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Chart Position Prediction: Model Comparison")
    ax.legend()
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "hit_model_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Model comparison chart saved -> {out_path}")


# ── Step 5: Evaluation ────────────────────────────────────────────────────────

def evaluate_hit_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
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

    Naive Bayes receives only the 'clean_lyrics' column and is fitted with
    per-sample weights to compensate for its lack of class_weight support.
    Logistic Regression and LinearSVC receive the full feature DataFrame
    (TF-IDF + numerical features via FeatureUnion).

    For each model the following are printed to stdout:
        - Macro F1 and per-class F1, precision, recall
        - Full sklearn classification report

    Saved to output_dir:
        - hit_confusion_matrix_{model}.png  (one per model)
        - hit_model_comparison.png          (grouped bar chart)

    Args:
        X_train:    Training feature DataFrame (clean_lyrics + numeric cols).
        X_test:     Test feature DataFrame.
        y_train:    Training chart_label values.
        y_test:     Test chart_label values.
        log_model:  Unfitted Logistic Regression pipeline.
        nb_model:   Unfitted Naive Bayes pipeline.
        svm_model:  Unfitted LinearSVC pipeline.
        output_dir: Directory to save PNG outputs.

    Returns:
        Dict of {model_name: {"macro_f1": float, "f1_top_10": float,
        "f1_top_100": float, "precision_top_10": float,
        "recall_top_10": float}}.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Naive Bayes uses only clean_lyrics; LogReg + SVM use the full DataFrame
    sample_weights = _compute_sample_weights(y_train)
    log_model.fit(X_train, y_train)
    nb_model.fit(X_train[["clean_lyrics"]], y_train, model__sample_weight=sample_weights)
    svm_model.fit(X_train, y_train)

    models = {
        "Logistic Regression": (log_model, X_test),
        "Naive Bayes":         (nb_model,  X_test[["clean_lyrics"]]),
        "LinearSVC":           (svm_model, X_test),
    }

    results: dict[str, dict[str, float]] = {}

    for name, (model, X_input) in models.items():
        pred   = model.predict(X_input)
        report = classification_report(
            y_test, pred,
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        )

        macro_f1     = f1_score(y_test, pred, average="macro", zero_division=0)
        f1_top10     = report["top_10"]["f1-score"]
        f1_top100    = report["top_100"]["f1-score"]
        prec_top10   = report["top_10"]["precision"]
        recall_top10 = report["top_10"]["recall"]

        results[name] = {
            "macro_f1":         macro_f1,
            "f1_top_10":        f1_top10,
            "f1_top_100":       f1_top100,
            "precision_top_10": prec_top10,
            "recall_top_10":    recall_top10,
        }

        print(f"\n{'=' * 45}")
        print(f"  {name}")
        print(f"{'=' * 45}")
        print(f"  Macro F1:                    {macro_f1:.4f}")
        print(f"  F1  (top_10):                {f1_top10:.4f}")
        print(f"  F1  (top_100):               {f1_top100:.4f}")
        print(f"  Precision (top_10):          {prec_top10:.4f}")
        print(f"  Recall    (top_10):          {recall_top10:.4f}")
        print(f"\nFull classification report:")
        print(classification_report(
            y_test, pred,
            labels=CLASS_ORDER,
            zero_division=0,
        ))

        _save_confusion_matrix(y_test, pred, name, output_dir)

    _plot_hit_model_comparison(results, output_dir)
    return results


# ── Step 6: Interpretability ──────────────────────────────────────────────────
def show_top_hit_words(model: Pipeline, n: int = 15) -> None:
    feature_union  = model.named_steps["features"]
    tfidf_pipeline = dict(feature_union.transformer_list)["tfidf"]
    vectorizer     = tfidf_pipeline.named_steps["vectorizer"]

    tfidf_names   = vectorizer.get_feature_names_out()
    numeric_names = np.array(NUMERIC_FEATURES)
    feature_names = np.concatenate([tfidf_names, numeric_names])

    coef    = model.named_steps["model"].coef_
    classes = model.named_steps["model"].classes_

    print("\nTOP WORDS PER CHART CLASS (LinearSVC)")

    # Binaire classificatie: coef_ heeft maar 1 rij
    # Positieve waarden -> classes[1] (top_10)
    # Negatieve waarden -> classes[0] (top_100)
    if coef.shape[0] == 1:
        coef_row = coef[0]

        print(f"\n  {classes[1].upper()} (hoogste positieve coëfficiënten)")
        top_idx = coef_row.argsort()[-n:][::-1]
        for idx in top_idx:
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"    {name:<30} coef={coef_row[idx]:.4f}")

        print(f"\n  {classes[0].upper()} (hoogste negatieve coëfficiënten)")
        bot_idx = coef_row.argsort()[:n]
        for idx in bot_idx:
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"    {name:<30} coef={coef_row[idx]:.4f}")

    else:
        # Multiclass: originele logica
        for i, label in enumerate(classes):
            print(f"\n  {label.upper()}")
            top_idx = coef[i].argsort()[-n:][::-1]
            for idx in top_idx:
                name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                print(f"    {name:<30} coef={coef[i][idx]:.4f}")

# ── Step 7: Single-song prediction ───────────────────────────────────────────

def predict_hit(lyrics: str, model: Pipeline) -> tuple[str, dict[str, float] | None]:
    """
    Predicts the chart class for a single song given its raw lyrics.

    Constructs a one-row DataFrame with placeholder numerical features set
    to 0.0 so the FeatureUnion pipeline receives the expected input shape.
    For production use, pass real numerical features via add_lyric_features().

    Args:
        lyrics: Raw, unprocessed song lyrics as a plain string.
        model:  A fitted sklearn Pipeline (FeatureUnion + classifier, or
                plain vectorizer + classifier for Naive Bayes).

    Returns:
        Tuple of (predicted_label, class_probabilities).
        predicted_label is one of "top_100" or "top_10".
        class_probabilities is a dict mapping each class to its probability,
        or None for models that do not support predict_proba (e.g. LinearSVC).
    """
    from feature_engineering import add_lyric_features

    # Build a one-row DataFrame with real numerical features
    row = pd.DataFrame([{"lyrics": lyrics, "clean_lyrics": lyrics}])
    row = add_lyric_features(row, lyrics_col="lyrics")

    # Ensure all expected columns are present
    for col in NUMERIC_FEATURES:
        if col not in row.columns:
            row[col] = 0.0

    label = str(model.predict(row)[0])
    probs = None
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(row)[0]
        classes   = model.named_steps["model"].classes_
        probs     = dict(zip(classes, raw_probs))
    return label, probs


# ── Full standalone pipeline ──────────────────────────────────────────────────

def run_hit_prediction_pipeline() -> None:
    """
    Runs the full 2-class chart position prediction pipeline end-to-end.

    Executes all steps: dataset construction (with feature engineering),
    NLP preprocessing, splitting, model training, evaluation,
    interpretability analysis, and a short demo.

    Outputs (confusion matrices, comparison chart) are saved to outputs/.
    """

    # Step 1: Build labelled dataset with numerical features
    print("\n Step 1: Merging lyrics with Billboard labels + adding features")
    df = build_hit_dataset()

    # Step 2: Deduplicate
    print("\n Step 2: Deduplicating")
    df = explore_and_clean_data(df)

    # Step 3: NLP preprocessing (adds 'clean_lyrics' column)
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
        ("Baby you light up my world like nobody else",                         "top_10"),
    ]
    for lyrics, true_label in demo_songs:
        label, probs = predict_hit(lyrics, svm_model)
        print(f"  Lyrics   : \"{lyrics[:55]}...\"")
        print(f"  Predicted: {label:<8}  |  Actual: {true_label}")
        if probs:
            probs_str = "  |  ".join(f"{k}: {v:.2f}" for k, v in probs.items())
            print(f"  Probs    : {probs_str}")
        print()

    print("Chart position prediction pipeline complete. Outputs saved to outputs/")


if __name__ == "__main__":
    run_hit_prediction_pipeline()