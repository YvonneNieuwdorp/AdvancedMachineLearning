"""
hit_prediction.py
-----------------
Predicts chart position (top_25 vs top_100) using lyrics features.

Features:
    - TF-IDF on preprocessed lyrics (unigrams + bigrams, 10k features)
    - 5 numerical features: vocab_richness, avg_word_len, word_count,
      sentiment, rhyme_density

Results saved to:
    outputs/hit_results.json
    outputs/hit_confusion_matrix_*.png
    outputs/hit_model_comparison.png

Can be run standalone:
    python src/hit_prediction.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from cleaning import build_hit_dataset, RUN_HIT_CLEANING, HIT_OUTPUT_PATH
from feature_engineering import NUMERIC_FEATURES, add_lyric_features
from preprocessing import explore_and_clean_data, preprocess_lyrics

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")

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


# ── Step 1: Load and label ────────────────────────────────────────────────────

def load_hit_dataset(path: str = HIT_OUTPUT_PATH) -> pd.DataFrame:
    """
    Loads clean_hit_lyrics.csv, assigns a 2-class chart_label
    (top_25 / top_100), and adds numerical lyric features.

    The CSV was produced by cleaning.build_hit_dataset() which already
    merged Genius lyrics with Billboard ranks. Here we re-threshold to
    top-25 for a better class balance.

    Args:
        path: Path to clean_hit_lyrics.csv.

    Returns:
        DataFrame with chart_label and 5 numerical feature columns added.
    """
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")

    # Re-threshold: top_10 instead of top_10 for better balance
    df["chart_label"] = df["rank"].apply(
        lambda r: "top_10" if pd.notna(r) and int(r) <= 10 else "top_100"
    )

    # Drop rows with no rank (no Billboard match)
    before = len(df)
    df = df[df["rank"].notna()].copy()
    print(f"Dropped {before - len(df)} rows with no Billboard match")
    print(f"\nClass distribution:\n{df['chart_label'].value_counts()}")
    print(f"\nClass balance (%):\n"
          f"{df['chart_label'].value_counts(normalize=True).mul(100).round(1)}")

    print("\nAdding numerical lyric features...")
    df = add_lyric_features(df, lyrics_col="lyrics")
    return df


# ── Step 2: Train/test split ──────────────────────────────────────────────────

def split_hit_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split. X is a DataFrame with clean_lyrics + numeric cols."""
    feature_cols = ["clean_lyrics"] + NUMERIC_FEATURES
    X = df[feature_cols]
    y = df["chart_label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ── Step 3: Model definitions ─────────────────────────────────────────────────

def create_hit_models() -> tuple[Pipeline, Pipeline, Pipeline]:
    """
    Creates three pipelines using FeatureUnion (TF-IDF + scaled numeric features).

    Returns:
        Tuple of (log_pipeline, rf_pipeline, svm_pipeline), all unfitted.
    """
    def combined_features() -> FeatureUnion:
        return FeatureUnion([
            ("tfidf", Pipeline([
                ("selector",   ColumnSelector("clean_lyrics")),
                ("vectorizer", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
            ])),
            ("numeric", Pipeline([
                ("selector", NumericFeatureSelector(NUMERIC_FEATURES)),
                ("scaler",   StandardScaler()),
            ])),
        ])

    log_pipeline = Pipeline([
        ("features", combined_features()),
        ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])
    svm_pipeline = Pipeline([
        ("features", combined_features()),
        ("model", LinearSVC(random_state=42, max_iter=2000)),
    ])
    rf_pipeline = Pipeline([
        ("features", combined_features()),
        ("model", RandomForestClassifier(
            random_state=42, n_estimators=200, class_weight="balanced", n_jobs=-1
        )),
    ])
    return log_pipeline, rf_pipeline, svm_pipeline


# ── Step 4: Helpers ───────────────────────────────────────────────────────────

def _save_confusion_matrix(
    y_test: pd.Series,
    pred: np.ndarray,
    model_name: str,
    output_dir: str,
) -> None:
    cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax)
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
    names     = list(results.keys())
    macro_f1  = [results[n]["macro_f1"]    for n in names]
    f1_top10  = [results[n]["f1_top_10"]   for n in names]
    f1_top100 = [results[n]["f1_top_100"]  for n in names]

    x     = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width, macro_f1,  width, label="Macro F1",    color="#4C8BE2")
    b2 = ax.bar(x,         f1_top10,  width, label="F1 (top_10)", color="#E2714C")
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
    rf_model: Pipeline,
    svm_model: Pipeline,
    output_dir: str = OUTPUT_DIR,
) -> dict[str, dict[str, float]]:
    """
    Fits and evaluates all three models. Saves confusion matrices,
    comparison chart, and hit_results.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    models = {
        "Logistic Regression": log_model,
        "Random Forest":       rf_model,
        "LinearSVC":           svm_model,
    }

    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        pred   = model.predict(X_test)
        report = classification_report(
            y_test, pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
        )
        macro_f1    = f1_score(y_test, pred, average="macro", zero_division=0)
        f1_top10    = report["top_10"]["f1-score"]
        f1_top100   = report["top_100"]["f1-score"]
        prec_top10  = report["top_10"]["precision"]
        rec_top10   = report["top_10"]["recall"]

        results[name] = {
            "macro_f1":        macro_f1,
            "f1_top_10":       f1_top10,
            "f1_top_100":      f1_top100,
            "precision_top_10": prec_top10,
            "recall_top_10":   rec_top10,
        }

        print(f"\n{'=' * 45}")
        print(f"  {name}")
        print(f"  Macro F1: {macro_f1:.4f}  |  F1 top_10: {f1_top10:.4f}  |  F1 top_100: {f1_top100:.4f}")
        print(classification_report(y_test, pred, labels=CLASS_ORDER, zero_division=0))

        _save_confusion_matrix(y_test, pred, name, output_dir)

    _plot_hit_model_comparison(results, output_dir)

    results_path = os.path.join(output_dir, "hit_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {results_path}")

    return results


# ── Step 6: Interpretability ──────────────────────────────────────────────────

def show_top_hit_words(model: Pipeline, n: int = 15) -> None:
    """Prints top discriminating words/features per class for LinearSVC."""
    feature_union  = model.named_steps["features"]
    tfidf_pipeline = dict(feature_union.transformer_list)["tfidf"]
    vectorizer     = tfidf_pipeline.named_steps["vectorizer"]

    tfidf_names   = vectorizer.get_feature_names_out()
    numeric_names = np.array(NUMERIC_FEATURES)
    feature_names = np.concatenate([tfidf_names, numeric_names])

    coef    = model.named_steps["model"].coef_
    classes = model.named_steps["model"].classes_

    print("\nTOP FEATURES PER CHART CLASS (LinearSVC)")

    if coef.shape[0] == 1:
        coef_row = coef[0]
        print(f"\n  {classes[1].upper()} (highest positive coefficients)")
        for idx in coef_row.argsort()[-n:][::-1]:
            feat = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"    {feat:<30} coef={coef_row[idx]:.4f}")
        print(f"\n  {classes[0].upper()} (highest negative coefficients)")
        for idx in coef_row.argsort()[:n]:
            feat = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"    {feat:<30} coef={coef_row[idx]:.4f}")
    else:
        for i, label in enumerate(classes):
            print(f"\n  {label.upper()}")
            for idx in coef[i].argsort()[-n:][::-1]:
                feat = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                print(f"    {feat:<30} coef={coef[i][idx]:.4f}")


# ── Step 7: Single-song prediction ───────────────────────────────────────────

def predict_hit(lyrics: str, model: Pipeline) -> tuple[str, dict[str, float] | None]:
    """Predicts chart class for a single song's raw lyrics."""
    row = pd.DataFrame([{"lyrics": lyrics, "clean_lyrics": lyrics}])
    row = add_lyric_features(row, lyrics_col="lyrics")
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


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_hit_prediction_pipeline() -> None:
    """
    Runs the full chart position prediction pipeline end-to-end.

    Steps:
        1. (Optional) Build hit dataset from Kaggle + Billboard
        2. Load, label, and add lyric features
        3. NLP preprocessing
        4. Train/test split
        5. Train + evaluate models
        6. 5-fold cross-validation
        7. Top discriminating features (LinearSVC)
        8. Demo predictions
    """

    # Step 1: Build dataset if needed
    if RUN_HIT_CLEANING or not os.path.exists(HIT_OUTPUT_PATH):
        print("\n Step 1: Building hit dataset")
        build_hit_dataset()
    else:
        print(f"\n Step 1: Skipped (using existing {HIT_OUTPUT_PATH})")

    # Step 2: Load + label
    print("\n Step 2: Loading dataset")
    df = load_hit_dataset()
    df = df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)
    

    # Step 3: NLP preprocessing
    df["tag"] = "unknown"
    df = preprocess_lyrics(df)
    
    # Step 4: Split
    print("\n Step 4: Splitting data")
    X_train, X_test, y_train, y_test = split_hit_data(df)
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"\n  Train distribution:\n{y_train.value_counts().to_string()}")
    print(f"\n  Test  distribution:\n{y_test.value_counts().to_string()}")

    # Step 5: Train + evaluate
    print("\n Step 5: Training and evaluating models")
    log_model, rf_model, svm_model = create_hit_models()
    evaluate_hit_models(X_train, X_test, y_train, y_test, log_model, rf_model, svm_model)

    # Step 6: Cross-validation
    print("\n Step 6: 5-fold cross-validation")
    cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_cols = ["clean_lyrics"] + NUMERIC_FEATURES

    for name, model in [
        ("Logistic Regression", log_model),
        ("Random Forest",       rf_model),
        ("LinearSVC",           svm_model),
    ]:
        scores = cross_val_score(
            model, df[feature_cols], df["chart_label"], cv=cv, scoring="f1_macro"
        )
        print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")

    # Step 7: Top features
    print("\n Step 7: Top discriminating features (LinearSVC)")
    show_top_hit_words(svm_model)

    # Step 8: Demo
    print("\n Step 8: Demo predictions (LinearSVC)")
    demo_songs = [
        ("I used to pray for times like this, to rhyme like this, so long ago", "top_25"),
        ("We found love in a hopeless place, shining in the dark",              "top_25"),
        ("Hey mama, I know I act a fool but got your name tattooed on my arm",  "top_100"),
        ("Baby you light up my world like nobody else",                         "top_25"),
    ]
    for lyrics, true_label in demo_songs:
        label, probs = predict_hit(lyrics, svm_model)
        print(f"  Lyrics   : \"{lyrics[:55]}...\"")
        print(f"  Predicted: {label:<8}  |  Actual: {true_label}")
        if probs:
            print(f"  Probs    : {' | '.join(f'{k}: {v:.2f}' for k, v in probs.items())}")
        print()

    print("Hit prediction pipeline complete. Outputs saved to outputs/")


if __name__ == "__main__":
    run_hit_prediction_pipeline()