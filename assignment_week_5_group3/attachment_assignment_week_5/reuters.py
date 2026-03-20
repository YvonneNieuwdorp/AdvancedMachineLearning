import re
from pathlib import Path
from html import unescape
from typing import Generator
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

__all__ = [
    "iter_sgm_files",
]

SRC_FOLDER = Path(__file__).parent / 'reuters'


def extract_attributes(tag: str) -> dict:
    """Extract attributes from a tag into a lowercase-key dictionary.

    Args:
        tag: Raw attribute string or full tag text containing key="value" pairs.

    Returns:
        A dictionary mapping lowercase attribute names to their string values.
    """
    return {
        k.lower(): v
        for k, v in re.findall(r'([A-Za-z_:][A-Za-z0-9_:\-.]*)="([^"]*)"', tag)
    }


def extract_tag_text(block: str, tag_name: str) -> str | None:
    """Extract and unescape the inner text of the first matching subtag.

    Args:
        block: Text block that may contain the target tag.
        tag_name: Tag name to extract, for example ``TOPICS`` or ``BODY``.

    Returns:
        The stripped inner text for the first match, or ``None`` if missing.
    """
    m = re.search(rf"<{tag_name}[^>]*>(.*?)</{tag_name}>", block, flags=re.DOTALL)
    return unescape(m.group(1)).strip() if m else None


def extract_topics_list(topics_text: str | None) -> list[str]:
    """Parse Reuters topic labels from TOPICS inner XML content.

    Args:
        topics_text: TOPICS content, typically containing repeated ``<D>...</D>`` nodes.

    Returns:
        A list of topic labels. Returns an empty list when input is empty.
    """
    if not topics_text:
        return []
    return re.findall(r"<D>(.*?)</D>", topics_text, flags=re.DOTALL)


def iter_reuters_records(filename: Path) -> Generator[dict, None, None]:
    """Yield filtered Reuters records from one SGML file.

    The function keeps only records where the REUTERS attributes satisfy
    ``TOPICS == "YES"`` and ``LEWISSPLIT in {"TEST", "TRAIN"}``.

    Args:
        filename: Path to a Reuters ``.sgm`` file.

    Yields:
        Dictionaries containing selected metadata and extracted text fields.
    """
    text = Path(filename).read_text(encoding="latin-1", errors="ignore")
    for m in re.finditer(r"<REUTERS\b([^>]*)>(.*?)</REUTERS>", text, flags=re.DOTALL):
        attributes_text, reuters_tag_content = m.group(1), m.group(2)
        attributes = extract_attributes(attributes_text)
        if attributes.get("topics") == "YES" and attributes.get("lewissplit") in ("TEST", "TRAIN"):
            yield {
                "topics_present": attributes.get("topics"),
                "lewissplit": attributes.get("lewissplit"),
                "topics": ','.join(
                    extract_topics_list(extract_tag_text(reuters_tag_content, "TOPICS"))
                ),
                "body_text": extract_tag_text(reuters_tag_content, "BODY"),
            }


def iter_sgm_files(folder: Path) -> Generator[dict, None, None]:
    """Iterate over Reuters SGML files and yield enriched record dictionaries.

    Example of usage:
    df = pd.DataFrame(iter_sgm_files(SRC_FOLDER))

    Args:
        folder: Folder containing Reuters ``.sgm`` files.

    Yields:
        Record dictionaries from ``iter_reuters_records`` with added filename and
        per-file record index.
    """
    for filename in sorted(folder.glob('*.sgm')):
        # print(f"Processing {filename}...")
        for ix, record in enumerate(iter_reuters_records(filename), 1):
            record["filename"] = filename.name
            record["record_ix"] = ix
            yield record

print(list(SRC_FOLDER.glob("*.sgm")))

# DATA LOADING
df = pd.DataFrame(iter_sgm_files(SRC_FOLDER))

# Check properties
print("Columns:", df.columns)
print("Number of rows:", len(df))

# Drop empty text
df = df.dropna(subset=["body_text"])

# split topics
df['topics_list'] = df['topics'].apply(lambda x: x.split(',') if x else [])

# flatten all topics
all_topics = [topic for sublist in df['topics_list'] for topic in sublist]

# top 10 topics
top_10 = [t for t, _ in Counter(all_topics).most_common(10)]
print("Top 10 topics:", top_10)


# TRAIN / TEST SPLIT
# Split the dataset according to the predefined ModApte split.
train_df = df[df['lewissplit'] == 'TRAIN'].copy()
test_df = df[df['lewissplit'] == 'TEST'].copy()


# CREATE BINARY LABELS
# For each of the top 10 topics, create a binary classification problem.
# Each classifier predicts whether a document belongs to a given topic.
for topic in top_10:
    train_df[topic] = train_df['topics_list'].apply(lambda x: int(topic in x))
    test_df[topic] = test_df['topics_list'].apply(lambda x: int(topic in x))


# FEATURE ENGINEERING
# Convert text into numerical features using CountVectorizer.
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(train_df['body_text'])
X_test = vectorizer.transform(test_df['body_text'])

# For Bernoulli NB: binaire features (word present yes/no)
X_train_bin = (X_train > 0).astype(int)
X_test_bin = (X_test > 0).astype(int)


# TRAINING & EVALUATION
# Train both Multinomial and Bernoulli Naïve Bayes models for each topic
# and evaluate using precision, recall, and F1-score.
results = []

for topic in top_10:
    y_train = train_df[topic]
    y_test = test_df[topic]

    # Multinomial NB
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)
    pred_mnb = mnb.predict(X_test)

    # Bernoulli NB
    bnb = BernoulliNB(alpha=1.0)
    bnb.fit(X_train_bin, y_train)
    pred_bnb = bnb.predict(X_test_bin)

    # Calulate metrics 
    metrics = {
        "topic": topic,

        "mnb_precision": precision_score(y_test, pred_mnb, zero_division=0),
        "mnb_recall": recall_score(y_test, pred_mnb, zero_division=0),
        "mnb_f1": f1_score(y_test, pred_mnb, zero_division=0),

        "bnb_precision": precision_score(y_test, pred_bnb, zero_division=0),
        "bnb_recall": recall_score(y_test, pred_bnb, zero_division=0),
        "bnb_f1": f1_score(y_test, pred_bnb, zero_division=0),
    }

    results.append(metrics)


# RESULTS OVERVIEW
# Convert results into a DataFrame for easier analysis and comparison.
results_df = pd.DataFrame(results).round(3)

print("\nResults per topic:")
print(results_df)

# Calculate average performance
avg_results = results_df.mean(numeric_only=True).round(3)

print("\nAverage performance:")
print(avg_results)

# Print a comparison between the two models based on average F1-score.
print("\nModel comparison (based on F1-score):")
print(f"Multinomial NB avg F1: {avg_results['mnb_f1']:.4f}")
print(f"Bernoulli NB avg F1: {avg_results['bnb_f1']:.4f}")