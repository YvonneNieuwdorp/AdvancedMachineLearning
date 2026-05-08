"""Advanced Machine Learning – Natural Language Processing

In NLP, a core problem is how to represent words such that computers and algorithms can work with them. If we use simple 
count-based methods like a CountVectorizer, we fail to capture the meaning of words. For example, the sentences “oil price 
goes up” and “crude costs increase” have no words in common, even though they express the same message. Embeddings solve 
this problem by representing words, sentences, or documents as vectors, where words with similar meanings are mapped close
together. This is, for example, a common problem in search engines, where queries and documents with similar meaning need 
to be matched. 

In this assignment, we will build a Google News–style search engine. You will provide a search term, for example “Oil 
price increase” and the model will return a list of articles that are closest in meaning to your query. The goal is to 
use embeddings to measure semantic similarity, so that even if the exact words in the articles differ from the query, 
relevant articles can still be found. 

We will work with the Reuters news article dataset from Assignment 5 for this problem.
Find below a list of instructions that help you to approach this assignment.
"""
# Imports
import os
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from numpy.linalg import norm

# Only run these lines once
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('punkt_tab')

# Load the Reuters dataset from the previous assignment in a pandas dataframe
# We do not focus on the topic of the article in this exercise. Therefore, modify the function iter_sgm_files to load 
# all articles, not only the ones that have a topic provided
DATA_DIR = Path(__file__).parent / 'reuters'

def iter_sgm_files(data_dir):
    """
    Generator that yields all .sgm files in the data_dit folder.
    """
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.sgm'):
            file_path = data_dir / filename
            with open(file_path, 'r', encoding='latin1') as f:
                sgm_content = f.read()
                soup = BeautifulSoup(sgm_content, 'html.parser')
                for reuters in soup.find_all('reuters'):
                    text = ''
                    if reuters.body:
                        text = reuters.body.get_text()
                    topics = [d.get_text() for d in reuters.topics.find_all('d')] if reuters.topics else []
                    yield {
                        'id': reuters.get('newid'),
                        'text': text,
                        'topics': topics
                    }

def load_reuters(data_dir):
    """
    Load all articles into a pandas DataFrame.
    """
    articles = list(iter_sgm_files(data_dir))
    df = pd.DataFrame(articles)
    print(f'Number of articles loaded: {len(df)}')
    return df
    
# Preprocess and tokenize the text in the articles. You might want to make the text lowercase, remove punctuation and 
# stopwords.
def preprocess_text(text):
    """
    Makes the text lowercase, removes punctuation, tokenizes and removes stopwords.
    """
    text = text.lower()     # Make text lowercase
    text = re.sub(r'[^a-z\s]', '', text)    # remove punctuation
    tokens = word_tokenize(text)    # tokenize the text

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens


# Train a Word2Vec embedding model on your preprocessed text. To improve the quality of your model, you might want to 
# try out different vector sizes and window sizes. For your given data size, is a skip-gram or CBOW type model more
# appropriate?
def train_word2vec(tokenized_texts):
    """
    Trains a Word2Vec model with skip-gram (sg=1).
    The dataset is relatively small, skip-gram works better for semantics for smaller datasets.
    """

    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=100,   # dimension of the embedding
        window=5,          # context window (how many words left/right)
        min_count=2,       # ignore words that are infrequent
        workers=4,
        sg=1               # 1 = skip-gram, 0 = CBOW
    )

    return model


# Embed each of the articles with your model. Recall from the lecture that you can either average the embedding of each 
# word in the article or do a weighted average based on the TF-IDF vectorizer. Which versions gives you better results?
def document_embedding(tokens, model):
    valid_tokens = [t for t in tokens if t in model.wv]

    if not valid_tokens:
        return np.zeros(model.vector_size)

    vectors = [model.wv[t] for t in valid_tokens]
    return np.mean(vectors, axis=0)

# Finally write a function search(query, top_k), where query is your search string, and return a list of the closest 
# top_k articles to your query. As a measure for similarity, we use the cosine similarity.
def cosine_similarity(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def search(query, top_k, model, df):
    query_tokens = preprocess_text(query)
    query_vec = document_embedding(query_tokens, model)

    df['similarity'] = df['embedding'].apply(
        lambda x: cosine_similarity(query_vec, x)
    )

    results = df.sort_values(by='similarity', ascending=False).head(top_k)
    return results

# Report the results for search(‘oil price increase’, 5). Print the text of these 5 articles and judge if they indeed 
# cover the topic you requested. Report articles for two more queries of your choice.
def print_results(query, results):
    for i, row in results.iterrows():
        print(f"ID: {row['id']}")
        print(f"Similarity: {row['similarity']:.4f}")
        print(f"Text: {row['text'][:300]}...\n")

# Main
def main():

    # Load the dataset
    df = load_reuters(DATA_DIR)

    # Perform preprocessing on all articles 
    df['tokens'] = df['text'].apply(preprocess_text)

    # Train Word2Vec model 
    model = train_word2vec(df['tokens'])

    # Make document into vectors
    df['embedding'] = df['tokens'].apply(lambda t: document_embedding(t, model))

    # Report results
    q1 = "oil price increase"
    res1 = search(q1, 5, model, df)
    print_results(q1, res1)

    q2 = "stock market crash"
    res2 = search(q2, 5, model, df)
    print_results(q2, res2)

    q3 = "interest rate rise"
    res3 = search(q3, 5, model, df)
    print_results(q3, res3)

if __name__ == '__main__':
    main()
    