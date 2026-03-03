import re
from pathlib import Path
import re
from collections import defaultdict, Counter

# Preprocessing
def preprocess_corpus(clean_dir):
    """
    Loads cleaned books and converts them into a list of tokens.

    Returns
    tokens : list[str]
        List of word tokens including <EOS>.
    """

    tokens = []

    for file in clean_dir.glob('*.[tT][xX][tT]'):
        text = file.read_text(encoding='utf-8').lower()

        # Split sentences
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            if words:
                tokens.extend(words)
                tokens.append('<EOS>')  # sentence boundary

    print(f'Total tokens collected: {len(tokens)}')
    return tokens

# Build N-gram count models
def build_ngram_counts(tokens, n):
    """
    Builds n-gram and context counts.

    Parameters
    tokens : list[str]
    n : int
        n-gram size (2 = bigram, 3 = trigram)

    Returns
    ngram_counts : dict
    context_counts : dict
    """

    ngram_counts = Counter()
    context_counts = Counter()

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        context = tuple(tokens[i:i+n-1])

        ngram_counts[ngram] += 1
        context_counts[context] += 1

    print(f'{n}-gram types:', len(ngram_counts))
    return ngram_counts, context_counts

# No smoothing
def build_ngram_model(ngram_counts, context_counts):
    """
    Converts counts into an unsmoothed probability model.

    Returns
    model : dict
        model[context][next_word] = probability
    """

    model = defaultdict(dict)

    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        next_word = ngram[-1]

        probability = count / context_counts[context]
        model[context][next_word] = probability

    print('Unsmoothed language model created.')
    return dict(model)

# Laplace smoothing
def build_ngram_model_laplace(ngram_counts, context_counts, vocabulary):
    """
    Builds Laplace-smoothed n-gram model in a memory-efficient way.

    Only stores observed n-grams.
    Unseen words are handled dynamically during prediction.
    """

    model = defaultdict(dict)
    V = len(vocabulary)

    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        next_word = ngram[-1]

        smoothed_prob = (count + 1) / (context_counts[context] + V)
        model[context][next_word] = smoothed_prob

    print("Laplace-smoothed language model created (sparse).")

    return dict(model), V

# Nog te doen
# Next-Word Prediction (Unsmoothed Only)
# Sentence Generation
# Compare Models
# Reflection Section 

# Nodige functies:
# - preprocess_corpus()
# - build_ngram_counts()
# - build_ngram_model()
# - build_ngram_model_laplace()
# - predict_next_word()
# - generate_sentence()
# - main()

# Vul main aan met functies

def main():
    
    clean_dir = Path('cleaned_books')

    # preprocessing
    tokens = preprocess_corpus(clean_dir)
    vocabulary = set(tokens)

    # Bigram
    bi_counts, bi_context = build_ngram_counts(tokens, n=2)
    bigram_model = build_ngram_model(bi_counts, bi_context)
    bigram_model_laplace, V = build_ngram_model_laplace(
        bi_counts,
        bi_context,
        vocabulary
    )

    # Trigram
    tri_counts, tri_context = build_ngram_counts(tokens, n=3)
    trigram_model = build_ngram_model(tri_counts, tri_context)
    trigram_model_laplace, V = build_ngram_model_laplace(
        tri_counts,
        tri_context,
        vocabulary
    )

    return {
        'bigram': bigram_model,
        'bigram_laplace': bigram_model_laplace,
        'trigram': trigram_model,
        'trigram_laplace': trigram_model_laplace,
        'vocabulary': vocabulary
    }
    
if __name__ == '__main__':
    main()