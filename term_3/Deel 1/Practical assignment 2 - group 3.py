import re
import pandas as pd
from pathlib import Path

from tokenizers import Tokenizer # main object for tokenization
from tokenizers.models import BPE # model implementing Byte Pair Encoding.
from tokenizers.trainers import BpeTrainer # trains the BPE vocabulary
from tokenizers.pre_tokenizers import Whitespace # simple pre-tokenizer splitting on spaces
import json

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Constants
TXT_PATTERN = '*.[tT][xX][tT]'

# Pad naar downloaded_books
path_y = r'C:\Users\ynieu\OneDrive\Documenten\Toegepaste wiskunde\Data Science\Advanced Machine Learning\AdvancedMachineLearning\downloaded_books'
path_j = r'C:\Users\Jessi\OneDrive\Bureaublad\Fontys\AML\AdvancedMachineLearning\downloaded_books'

input_dir = Path(path_y)
project_dir = input_dir.parent # Pakt de bovenliggende map = AdvancedMachineLearning 
output_dir = project_dir / 'cleaned_books' 
output_dir.mkdir(parents=True, exist_ok=True) 

model_dir = project_dir / 'tokenizer_model'


# Cleaning Gutenberg books
def clean_books(input_dir, output_dir):
    """
    Removes Gutenberg boilerplate and table of contents
    from all txt files in input_dir and saves cleaned
    versions in output_dir.
    """
    
    book_paths = list(input_dir.glob(TXT_PATTERN))

    for book_path in book_paths:
        with open(book_path, 'r', encoding='utf-8') as file:
            text = file.read()

        begin_marker = '*** START OF THE PROJECT GUTENBERG EBOOK'
        begin_index = text.find(begin_marker)
        if begin_index != -1: 
            text = text[begin_index + (len(begin_marker)):].rstrip()

        lines = text.splitlines()
        content_start = 0
        for i, line in enumerate(lines): 
            if 'contents' in line.lower() or 'table of contents' in line.lower(): 
                content_start = i
                break

        content_end = content_start
        for i in range(content_start + 1, len(lines)):
            line = lines[i]
            if len(line.strip()) == 0: 
                content_end = i
                break
            if len(line.strip()) > 100: 
                content_end = i
                break
        
        if content_start > 0: 
            lines = lines[:content_start] + lines[content_end+1:]

        text = '\n'.join(lines)

        end_marker = '*** END OF THE PROJECT GUTENBERG EBOOK'
        end_index = text.find(end_marker)
        if end_index != -1: 
            text = text[:end_index].rstrip() 

        output_path = output_dir / book_path.name
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f'{book_path.name} cleaned and saved.')

# Train BPE Tokenizer
def train_and_tokenize(clean_dir, model_output_dir, vocab_size=10000):
    """
    Trains a BPE tokenizer on the cleaned books
    and stores tokenizer + tokenized corpus.
    """

    # Verzamel alle teksten
    book_paths = list(clean_dir.glob(TXT_PATTERN))
    corpus = [path.read_text(encoding='utf-8') for path in book_paths]

    print(f'{len(corpus)} books loaded for tokenizer training.')

    # Initialiseer BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
    )

    # Train tokenizer
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    print('Tokenizer training complete.')

    # Opslaan tokenizer
    model_output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = model_output_dir / 'bpe_tokenizer.json'
    tokenizer.save(str(tokenizer_path))

    print(f'Tokenizer saved to {tokenizer_path}')

    # Tokenize complete corpus
    full_text = ' '.join(corpus)
    encoding = tokenizer.encode(full_text)

    token_ids = encoding.ids
    tokens = encoding.tokens

    # Save tokenized output
    with open(model_output_dir / 'tokenized_ids.json', 'w') as f:
        json.dump(token_ids, f)

    with open(model_output_dir / 'tokenized_tokens.json', 'w') as f:
        json.dump(tokens, f)

    print('Tokenized output saved.')

    print(f'Total tokens: {len(tokens)}')
    print(f'Vocabulary size: {tokenizer.get_vocab_size()}')

# Shared Token Loader
def load_tokens(books):
    """
    Loads all cleaned books and returns a list
    of lowercase word tokens.
    """
    
    tokens = []
    for file in books.glob(TXT_PATTERN):
        text = file.read_text(encoding='utf-8').lower()
        words = re.findall(r'\b\w+\b', text)
        tokens.extend(words)
        
    return tokens


# Lemmatization        
def lemma_corpus(tokens):
    
    print('Counting lemmas...')
    lemmatizer = WordNetLemmatizer()
    
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    
    freq = pd.Series(lemmatized).value_counts()
    unique_lemmas = len(set(lemmatized))
    print('Unique lemmas:', unique_lemmas)
    print('Top 20:')
    print(freq.head(20))

    return unique_lemmas

# Stemming
def stem_corpus(tokens):
    
    print('Counting stems...')
    stemmer = PorterStemmer()
    
    stemmed = [stemmer.stem(w) for w in tokens]
    
    freq = pd.Series(stemmed).value_counts()
    unique_stems = len(set(stemmed))
    print('Unique stems:', unique_stems)
    print('Top 20:')
    print(freq.head(20))
    
    return unique_stems

# Main pipeline
def main():
    
    print('Starting NLP preprocessing pipeline...')

    input_dir = Path(path_y)
    project_dir = input_dir.parent
    output_dir = project_dir / 'cleaned_books'
    model_dir = project_dir / 'tokenizer_model'

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    clean_books(input_dir, output_dir)
    train_and_tokenize(output_dir, model_dir)
    tokens = load_tokens(output_dir)
    lemma_count = lemma_corpus(tokens)
    stem_count = stem_corpus(tokens)
    
    # Are there notable differences to the lemmatized tokens? 
    difference = stem_count - lemma_count
    
    print(f'There are {lemma_count} lemmas and {stem_count} stems.')
    if difference > 0:
        print(f'There are {difference} more stems than lemmas.')
    elif difference < 0:
        print(f'There are {abs(difference)} more lemmas than stems.')
    else:
        print(f'There is an equal amount of lemmas and stems.')
 
if __name__ == '__main__':
    main()
