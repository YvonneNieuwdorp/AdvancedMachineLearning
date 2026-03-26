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


# Load the Reuters dataset from the previous assignment in a pandas dataframe
# We do not focus on the topic of the article in this exercise. Therefore, modify the function iter_sgm_files to load 
# all articles, not only the ones that have a topic provided


# Preprocess and tokenize the text in the articles. You might want to make the text lowercase, remove punctuation and 
# stopwords.


# Train a Word2Vec embedding model on your preprocessed text. To improve the quality of your model, you might want to 
# try out different vector sizes and window sizes. For your given data size, is a skip-gram or CBOW type model more
# appropriate?


# Embed each of the articles with your model. Recall from the lecture that you can either average the embedding of each 
# word in the article or do a weighted average based on the TF-IDF vectorizer. Which versions gives you better results?


# Finally write a function search(query, top_k), where query is your search string, and return a list of the closest 
# top_k articles to your query. As a measure for similarity, we use the cosine similarity.


# Report the results for search(‘oil price increase’, 5). Print the text of these 5 articles and judge if they indeed 
# cover the topic you requested. Report articles for two more queries of your choice.

