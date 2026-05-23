"""
evaluation.py
-------------
Evaluates trained models and saves visualisations.

Functions:
    evaluation(...)             confusion matrices + metric report per model
    plot_model_comparison(...)  grouped bar chart across models
    show_top_words(...)         top discriminating words per genre
    predict_genre(...)          predicts the genre of a single song given its raw lyrics.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def plot_model_comparison(results: dict[str, dict[str, float]], output_dir: str = OUTPUT_DIR) -> None:
    """
    Saves a grouped bar chart comparing accuracy and macro F1 across models.

    Args:
        results:    Dict of {model_name: {"accuracy": float, "macro_f1": float}}.
        output_dir: Directory to save the chart PNG.
    """
    names    = list(results.keys())
    accuracy = [results[n]["accuracy"] for n in names]
    macro_f1 = [results[n]["macro_f1"] for n in names]

    x = range(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar([i - width / 2 for i in x], accuracy, width, label="Accuracy", color="#4C8BE2")
    bars2 = ax.bar([i + width / 2 for i in x], macro_f1, width, label="Macro F1", color="#E2714C")

    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Accuracy vs Macro F1")
    ax.legend()
    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Model comparison chart saved -> {out_path}")


def evaluation(
    X_test: pd.Series,
    y_test: pd.Series,
    log_model: Pipeline,
    nb_model: Pipeline,
    svm_model: Pipeline,
    output_dir: str = OUTPUT_DIR,
) -> dict[str, dict[str, float]]:
    """
    Evaluates all three models on the test set and saves visualisations.

    For each model, prints accuracy, macro F1, and a full classification
    report, then saves a confusion matrix heatmap. After all models are
    evaluated, saves a grouped bar chart comparing their scores.

    Args:
        X_test:     Test lyrics (preprocessed strings).
        y_test:     True genre labels for the test set.
        log_model:  Fitted Logistic Regression pipeline.
        nb_model:   Fitted Naive Bayes pipeline.
        svm_model:  Fitted LinearSVC pipeline.
        output_dir: Directory to save PNG outputs.

    Returns:
        Dict of {model_name: {"accuracy": float, "macro_f1": float}}.
    """
    os.makedirs(output_dir, exist_ok=True)

    models = {
        "Logistic Regression": log_model,
        "Naive Bayes":         nb_model,
        "LinearSVC":           svm_model,
    }
    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        pred = model.predict(X_test)
        acc  = accuracy_score(y_test, pred)
        f1   = f1_score(y_test, pred, average="macro")
        results[name] = {"accuracy": acc, "macro_f1": f1}

        print(f"\n{'=' * 40}")
        print(f"{name}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Macro F1 : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, pred, zero_division=0))

        cm     = confusion_matrix(y_test, pred)
        labels = model.classes_

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()

        filename = name.lower().replace(" ", "_")
        out_path = os.path.join(output_dir, f"confusion_matrix_{filename}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved -> {out_path}")

    plot_model_comparison(results, output_dir)
    return results


def show_top_words(model: Pipeline, n: int = 10) -> None:
    """
    Prints the top n most discriminating words per genre for a LinearSVC model.

    Args:
        model: Fitted sklearn Pipeline with 'vectorizer' and 'model' steps.
               The model step must expose a coef_ attribute (LinearSVC or
               Logistic Regression).
        n:     Number of top words to display per genre. Defaults to 10.
    """
    feature_names = model.named_steps["vectorizer"].get_feature_names_out()
    coefficients  = model.named_steps["model"].coef_
    classes       = model.named_steps["model"].classes_

    print("\nTOP WORDS PER GENRE (LinearSVC)")
    for i, genre in enumerate(classes):
        top_indices = coefficients[i].argsort()[-n:]
        print(f"\n{genre}:")
        for idx in reversed(top_indices):
            print(f"  {feature_names[idx]}")


def predict_genre(lyrics: str, model: Pipeline) -> str:
    """
    Predicts the genre of a single song given its raw lyrics.

    Wraps the full preprocessing and prediction pipeline so end-users
    can classify new songs without interacting with the training code.

    Args:
        lyrics: Raw, unprocessed song lyrics as a plain string.
        model:  A fitted sklearn Pipeline (vectorizer + classifier).

    Returns:
        Predicted genre label as a string (e.g. 'rap', 'pop').
    """
    return model.predict([lyrics])[0]
