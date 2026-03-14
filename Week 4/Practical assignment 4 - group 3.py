# pip install datasets in terminal 
# pip install emoji

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset
import pandas as pd
import string
import emoji

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# Data Loading
def load_data(authors=None, top_n=None):
    """
    Load tweets from the CelebrityTweets dataset.
    
    Args:
        authors (list[str], optional): List of authors to filter. Defaults to None.
        top_n (int, optional): If specified, selects top N authors by tweet count. Defaults to None.

    Returns:
        pd.DataFrame: Filtered dataset containing tweets from the selected authors.
    """
    dataset_twitter = load_dataset("Jacobvs/CelebrityTweets")
    data = dataset_twitter["train"].to_pandas()

    if top_n is not None:
        # Select top N authors by number of tweets
        authors = data["username"].value_counts().head(top_n).index
        data = data[data["username"].isin(authors)]

    elif authors is not None:
        # Filter dataset by specified authors
        data = data[data["username"].isin(authors)]

    return data


# Text Cleaning
def clean_text(text):
    """
    Cleans the text by removing hashtags, URLs, mentions, punctuation, 
    and converting emojis to descriptive text. Converts to lowercase.
    
    Args:
        text (pd.Series): A pandas Series containing the text to be cleaned.

    Returns:
        pd.Series: Cleaned text.
    """
    # Remove hashtags
    text = text.str.replace(r'#[^\s]+', '', regex=True)
    # Remove URLs
    text = text.str.replace(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
        ' ', 
        regex=True
    )
    # Remove mentions
    text = text.str.replace(r'@[^\s]+', '', regex=True)
    # Remove punctuation
    text = text.apply(lambda x: "".join([c for c in x if c not in string.punctuation]))
    # Convert emojis to descriptive text
    text = text.apply(lambda x: emoji.demojize(x))
    # Drop missing values
    text = text.dropna()
    # Lowercase
    text = text.str.lower()

    return text


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


# Model Evaluation
def evaluate_model(name, model, X_test, y_test):
    """
    Evaluate a model and print accuracy, classification report, and confusion matrix.

    Args:
        name (str): Name of the model.
        model: Trained model pipeline.
        X_test (pd.Series): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        np.ndarray: Predicted labels.
    """
    preds = model.predict(X_test)

    print(f"\n{name}:")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return preds


# Example Predictions
def predict_examples(models):
    """
    Predict authors for example sentences using multiple models.
    
    Args:
        models (dict): Dictionary of model_name: model_pipeline.
    """
    sentences = [
        "Can't wait for my new album to drop!",
        "Starship is ready for launch!"
    ]

    print("\nExample sentence predictions")
    for sentence in sentences:
        print(f"\nSentence: {sentence}")
        for name, model in models.items():
            pred = model.predict([sentence])[0]
            print(f"{name}: {pred}")


# Detailed Predictions
def print_predictions(model, X_test, y_test, test_data, limit=5):
    """
    Print a few test samples with predicted probabilities for each author.

    Args:
        model: Trained model pipeline.
        X_test (pd.Series): Test features.
        y_test (pd.Series): Test labels.
        test_data (pd.DataFrame): Original test data (for tweet text).
        limit (int): Maximum number of samples to display.
    """
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    authors = model.classes_

    print("\nSample predictions")
    for i in range(min(limit, len(test_data))):
        tweet = test_data.iloc[i]["text"]
        true_author = y_test.iloc[i]
        predicted_author = preds[i]

        print(f"\nTweet: {tweet}")
        print(f"True: {true_author}")
        print(f"Predicted: {predicted_author}")
        # Print probability for each author
        for j, author in enumerate(authors):
            print(f"{author} prob: {probs[i][j]:.3f}")


# Misclassified Tweets
def print_misclassified(preds, y_test, test_data, limit=5):
    """
    Print misclassified tweets up to a limit.

    Args:
        preds (np.ndarray): Predicted labels.
        y_test (pd.Series): True labels.
        test_data (pd.DataFrame): Original test data (for tweet text).
        limit (int): Maximum number of misclassified tweets to display.
    """
    print("\nMisclassified tweets")
    count = 0
    for i in range(len(test_data)):
        if preds[i] != y_test.iloc[i]:
            print(f"\nTweet: {test_data.iloc[i]['text']}")
            print(f"True: {y_test.iloc[i]}")
            print(f"Predicted: {preds[i]}")
            count += 1
            if count >= limit:
                break


# Main Pipeline
def main():
    """
    Main function to load data, clean it, train models, and print evaluations.
    """
    # Uncomment to test specific authors
    # authors = ["justinbieber", "elonmusk"]
    # data = load_data(authors)
    
    # Extra: Load top N authors
    data = load_data(top_n=5)

    # Clean the tweet texts
    data["cleaned_text"] = clean_text(data["text"])

    # Split into train/test sets
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["username"]
    )

    X_train = train_data["cleaned_text"]
    X_test = test_data["cleaned_text"]
    y_train = train_data["username"]
    y_test = test_data["username"]

    # Create pipelines
    log_model, nb_model = create_models()

    # Train models
    log_model.fit(X_train, y_train)
    nb_model.fit(X_train, y_train)

    # Evaluate models
    log_preds = evaluate_model("Logistic Regression", log_model, X_test, y_test)
    nb_preds = evaluate_model("Naive Bayes", nb_model, X_test, y_test)

    # Example predictions
    predict_examples({
        "Logistic Regression": log_model,
        "Naive Bayes": nb_model
    })

    # Detailed prediction probabilities
    print_predictions(log_model, X_test, y_test, test_data)

    # Print misclassified tweets
    print_misclassified(log_preds, y_test, test_data)


if __name__ == "__main__":
    main()