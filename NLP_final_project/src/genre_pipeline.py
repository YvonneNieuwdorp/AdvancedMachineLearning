"""
genre_pipeline.py
-----------------
End-to-end genre classification pipeline.

Models trained:
    1. Logistic Regression  (TF-IDF)
    2. Naive Bayes          (TF-IDF)
    3. LinearSVC            (TF-IDF)
    4. DistilBERT           (HuggingFace fine-tune, optional)

Results saved to:
    outputs/genre_results.json
    outputs/confusion_matrix_*.png
    outputs/genre_model_comparison.png

Can be run standalone:
    python src/genre_pipeline.py
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,f1_score,)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion

from cleaning import build_genre_dataset, RUN_GENRE_CLEANING, GENRE_OUTPUT_PATH
from preprocessing import explore_and_clean_data, preprocess_lyrics

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
GENRE_CLEAN_PATH = os.path.join(BASE_DIR, "data", "clean_genre_lyrics_preprocessed.csv")
RUN_HF_MODEL = False  # Set True to fine-tune DistilBERT (requires GPU recommended)
REMOVE_GENRE = 'pop'

# ── Data ──────────────────────────────────────────────────────────────────────

def load_genre_data(path: str = GENRE_OUTPUT_PATH) -> pd.DataFrame:
    if os.path.exists(GENRE_CLEAN_PATH):
        print("Loading cached preprocessed data...")
        return pd.read_csv(GENRE_CLEAN_PATH)
    df = pd.read_csv(path)
    df = explore_and_clean_data(df)
    df = preprocess_lyrics(df)
    df.to_csv(GENRE_CLEAN_PATH, index=False)
    return df

def sample_dataset(df: pd.DataFrame, max_samples: int = 50000, random_state: int = 42):
    """
    Reduce dataset size for faster training.
    Keeps class balance using stratified sampling.
    """
    return (
        df.groupby("tag", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), max_samples // 5), random_state=random_state))
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def split_genre_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified 80/20 split on genre tag."""
    X = df["clean_lyrics"]
    y = df["tag"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ── Sklearn models ────────────────────────────────────────────────────────────
def create_genre_models():
    def word_tfidf():
        return TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            stop_words="english",
            min_df=10,
            sublinear_tf=True
        )
    def char_tfidf():
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2,4),
            max_features=8000
        )
    vectorizer = FeatureUnion([
        ("word", word_tfidf()),
        ("char", char_tfidf())
    ])

    log_pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("model", LogisticRegression(max_iter=1500, class_weight="balanced", n_jobs=-1))
    ])
    nb_pipeline = Pipeline([
        ("vectorizer", word_tfidf()),
        ("model", MultinomialNB())
    ])
    svm_pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("model", LinearSVC(class_weight="balanced", max_iter=1000))
    ])

    return log_pipeline, nb_pipeline, svm_pipeline


def _save_confusion_matrix(
    y_test: pd.Series,
    pred,
    model_name: str,
    labels: list[str],
    output_dir: str,
) -> None:
    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    fname    = model_name.lower().replace(" ", "_")
    out_path = os.path.join(output_dir, f"genre_confusion_matrix_{fname}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {out_path}")


def _plot_model_comparison(
    results: dict[str, dict[str, float]],
    output_dir: str,
) -> None:
    names    = list(results.keys())
    accuracy = [results[n]["accuracy"] for n in names]
    macro_f1 = [results[n]["macro_f1"] for n in names]

    x     = range(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar([i - width / 2 for i in x], accuracy, width, label="Accuracy", color="#4C8BE2")
    b2 = ax.bar([i + width / 2 for i in x], macro_f1, width, label="Macro F1", color="#E2714C")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Genre Classification: Model Comparison")
    ax.legend()
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "genre_model_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Model comparison chart saved -> {out_path}")


def evaluate_genre_models(
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
    Fits and evaluates all three sklearn genre models.
    Saves confusion matrices, a comparison chart, and genre_results.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = sorted(y_test.unique().tolist())

    models = {
        "Logistic Regression": log_model,
        "Naive Bayes":         nb_model,
        "LinearSVC":           svm_model,
    }

    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc  = accuracy_score(y_test, pred)
        f1   = f1_score(y_test, pred, average="macro", zero_division=0)
        results[name] = {"accuracy": acc, "macro_f1": f1}

        print(f"\n{'=' * 45}")
        print(f"  {name}")
        print(f"  Accuracy : {acc:.4f}  |  Macro F1 : {f1:.4f}")
        print(classification_report(y_test, pred, zero_division=0))

        _save_confusion_matrix(y_test, pred, name, labels, output_dir)

    _plot_model_comparison(results, output_dir)

    results_path = os.path.join(output_dir, "genre_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {results_path}")

    return results

def show_top_genre_words(model: Pipeline, n: int = 10) -> None:
    """Prints the top n discriminating words per genre for a LinearSVC pipeline."""
    feature_names = model.named_steps["vectorizer"].get_feature_names_out()
    coef          = model.named_steps["model"].coef_
    classes       = model.named_steps["model"].classes_

    print("\nTOP WORDS PER GENRE (LinearSVC)")
    for i, genre in enumerate(classes):
        top_idx = coef[i].argsort()[-n:][::-1]
        print(f"\n  {genre}:")
        for idx in top_idx:
            print(f"    {feature_names[idx]}")


def train_hf_genre_model(
    df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
) -> dict[str, float]:
    """
    Fine-tunes a DistilBERT model on the genre classification task.

    Uses HuggingFace Trainer API with:
        - Label encoding via sklearn LabelEncoder
        - Tokenisation with DistilBertTokenizerFast (max_length=256, truncation)
        - Stratified 80/20 split (same seed as sklearn models)
        - Macro F1 as evaluation metric

    Args:
        df:         Preprocessed DataFrame with 'lyrics' (raw) and 'tag' columns.
                    Uses raw lyrics (not clean_lyrics) so BERT sees full text.
        output_dir: Directory to save the fine-tuned model and results.
        model_name: HuggingFace model ID to fine-tune. Defaults to distilbert-base-uncased.
        epochs:     Number of training epochs. Defaults to 3.
        batch_size: Per-device batch size. Defaults to 16.

    Returns:
        Dict with {"accuracy": float, "macro_f1": float} on the test split.

    Notes:
        - Requires: transformers, datasets, torch (pip install transformers datasets torch)
        - GPU strongly recommended; CPU training is very slow on a large dataset.
        - The fine-tuned model is saved to outputs/hf_genre_model/.
    """
    try:
        import torch
        from datasets import Dataset
        from sklearn.preprocessing import LabelEncoder
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            Trainer,
            TrainingArguments,
            EvalPrediction,
        )
        import numpy as np
    except ImportError as e:
        print(f"\n[HF] Missing dependency: {e}")
        print("[HF] Install with: pip install transformers datasets torch")
        return {}

    print("\n=== HUGGINGFACE DISTILBERT GENRE MODEL ===")
    print(f"  Base model : {model_name}")
    print(f"  Epochs     : {epochs}  |  Batch size: {batch_size}")

    # Encode labels
    le     = LabelEncoder()

    # Use raw lyrics for BERT (contains more signal than cleaned text)
    df = df.sample(n=5000, random_state=42)
    texts = df["lyrics"].fillna("").tolist()
    labels = le.fit_transform(df["tag"].values)
    n_labels = len(le.classes_)
    print(f"  Labels ({n_labels}): {list(le.classes_)}")

    # Stratified split
    from sklearn.model_selection import train_test_split
    idx = list(range(len(texts)))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    def make_dataset(indices):
        return Dataset.from_dict({
            "text":  [texts[i] for i in indices],
            "label": [int(labels[i]) for i in indices],
        }).map(tokenize, batched=True)

    print("  Tokenising train split...")
    train_ds = make_dataset(train_idx)
    print("  Tokenising test split...")
    test_ds  = make_dataset(test_idx)

    hf_model_dir = os.path.join(output_dir, "hf_genre_model")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels
    )

    def compute_metrics(p: EvalPrediction) -> dict:
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro", zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=hf_model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir=os.path.join(hf_model_dir, "logs"),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("  Training...")
    trainer.train()

    print("  Evaluating...")
    eval_results = trainer.evaluate()
    hf_results = {
        "accuracy": eval_results.get("eval_accuracy", 0.0),
        "macro_f1": eval_results.get("eval_macro_f1", 0.0),
    }
    print(f"  Accuracy : {hf_results['accuracy']:.4f}  |  Macro F1 : {hf_results['macro_f1']:.4f}")

    # Save model + label mapping
    trainer.save_model(hf_model_dir)
    tokenizer.save_pretrained(hf_model_dir)
    label_map_path = os.path.join(hf_model_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({str(i): label for i, label in enumerate(le.classes_)}, f, indent=2)
    print(f"  Model saved -> {hf_model_dir}")

    # Add to genre_results.json
    results_path = os.path.join(output_dir, "genre_results.json")
    try:
        with open(results_path) as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}
    all_results["DistilBERT"] = hf_results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results appended -> {results_path}")

    return hf_results


def predict_genre(lyrics: str, model: Pipeline) -> str:
    """Predicts the genre of a single song given its raw lyrics."""
    return model.predict([lyrics])[0]

def run_genre_pipeline() -> None:
    """
    Runs the full genre classification pipeline end-to-end.

    Steps:
        1. (Optional) Build genre dataset from Kaggle
        2. Load and preprocess genre lyrics
        3. Train/test split
        4. Train + evaluate Logistic Regression, Naive Bayes, LinearSVC
        5. Print top discriminating words (LinearSVC)
        6. (Optional) Fine-tune DistilBERT
        7. Demo predictions
    """

    # Step 1: Build dataset if needed
    if RUN_GENRE_CLEANING or not os.path.exists(GENRE_OUTPUT_PATH):
        print("\n Step 1: Building genre dataset")
        build_genre_dataset()
    else:
        print(f"\n Step 1: Skipped (using existing {GENRE_OUTPUT_PATH})")

    # Step 2: Load + preprocess
    print("\n Step 2: Loading and preprocessing")
    df = load_genre_data()
    df = df[df["tag"] != REMOVE_GENRE].reset_index(drop=True)
    df = sample_dataset(df, max_samples=10000)  # Reduce size for faster training; comment out to use full dataset

    # Step 3: Split
    print("\n Step 3: Splitting data")
    X_train, X_test, y_train, y_test = split_genre_data(df)
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Step 4: Train + evaluate sklearn models
    print("\n Step 4: Training and evaluating sklearn models")
    log_model, nb_model, svm_model = create_genre_models()
    evaluate_genre_models(X_train, X_test, y_train, y_test,
                          log_model, nb_model, svm_model)

    # Step 5: Top words
    print("\n Step 5: Top discriminating words")
    show_top_genre_words(svm_model)

    # Step 6: HuggingFace DistilBERT
    if RUN_HF_MODEL:
        print("\n Step 6: Fine-tuning DistilBERT")
        train_hf_genre_model(df)
    else:
        print("\n Step 6: DistilBERT skipped (RUN_HF_MODEL=False)")

    # Step 7: Demo
    print("\n Step 7: Demo predictions (LinearSVC)")
    examples = [
        ("I used to pray for times like this, to rhyme like this",        "rap"),
        ("Pour some sugar on me, in the name of love",                    "rock"),
    ]
    for lyrics, true_label in examples:
        pred = predict_genre(lyrics, svm_model)
        print(f"  Lyrics   : \"{lyrics[:55]}\"")
        print(f"  Predicted: {pred}  |  Actual: {true_label}\n")

    print("Genre pipeline complete. Outputs saved to outputs/")


if __name__ == "__main__":
    run_genre_pipeline()