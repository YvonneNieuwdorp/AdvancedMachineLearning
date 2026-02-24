# Feedback week 1:
# Werk in .py bestand
# Werk met functies
# Code moet repeatable zijn

""" Build a python script that
•	removes repetitive text and structure from the downloaded txt files
•	tokenizes the text with a BPE tokenizer
•	lemmatizes the text with a Lemmatizer
•	stems the text with a Stemmer

You will use the stored texts and properties in subsequent assignments in this course.
"""
import re
import pandas as pd
from scrape_books import load_books

from tokenizers import Tokenizer # main object for tokenization
from tokenizers.models import BPE # model implementing Byte Pair Encoding.
from tokenizers.trainers import BpeTrainer # trains the BPE vocabulary
from tokenizers.pre_tokenizers import Whitespace # simple pre-tokenizer splitting on spaces
from tokenizers.decoders import BPEDecoder

import json

from transformers import AutoTokenizer

import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from nltk.stem import PorterStemmer

