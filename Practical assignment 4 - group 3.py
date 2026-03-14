# pip install datasets in terminal 
from datasets import load_dataset
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the data in a pandas dataframe (or format of your choice)
dataset_twitter = load_dataset("Jacobvs/CelebrityTweets")
data = dataset_twitter["train"].to_pandas()

# The dataset contains 90 authors. Filter for Justin Bieber and Elon Musk
data = data[(data['username'] == 'justinbieber') | (data['username'] == 'elonmusk')]
print(data.columns)

# Clean the text in the tweets (e.g. remove the attached URL, remove hashtags, remove empty tweets, …)
def clean_text(text):
    """Cleans the text by removing hashtags, URLs, mentions, punctuation, and emojis, and converting to lowercase.
    Args:        text (pd.Series): A pandas Series containing the text to be cleaned.
    Returns:        pd.Series: A cleaned version of the input text."""
    text = text.str.replace('#[^\s]+','', regex=True)  # Remove hashtags
    text = text.str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', regex=True)  # Remove URLs
    text = text.str.replace('@[^\s]+','', regex=True)  # Remove mentions
    text = text.apply(lambda x: "".join([char for char in x if char not in string.punctuation]))  # Remove punctuation
    text = text.dropna()  # Remove empty tweets
    text = text.astype(str).apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))  # Remove emojis
    text = text.str.lower()  # Lowercase
    return text

data['cleaned_text'] = clean_text(data['text'])    

# Do a train-test split with a test ratio of 20%
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Vectorize your data with N-grams up to length 3
vectorizer = TfidfVectorizer(
    max_features=50_000,      # limit vocabulary
    ngram_range=(1, 3),       # unigrams + bigrams + trigrams
)

# Train a Logistic Regression and Naïve Bayes model and report the accuracy and other relevant metrics to evaluate the performance of the model. Do you observe relevant differences between the two models?

# Use your models to classify the sentences:
# o	    “Can't wait for my new album to drop!”
# o	    “Starship is ready for launch!”

# -	For the Logistic Regression model, print every tweet in the test set, the predicted and true author and the probability of each author. What special things do you observe? Would you have been able to classify the wrongly classified tweets yourself? Find examples where your answer is yes and no.

#Extras:
#-	Try out your models for more than two authors and report meaningful statistics
#-	How do you properly deal with emojis in the text processing?
#-	Do you see further improvements if you include longer N-grams than trigrams?


