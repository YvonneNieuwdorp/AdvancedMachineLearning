import re
from pathlib import Path
import re
from collections import defaultdict, Counter
import random

# Nodige functies:
# - preprocess_corpus()
# - build_ngram_counts()
# - build_ngram_model()
# - build_ngram_model_laplace()
# - predict_next_word()
# - generate_sentence()
# - main()

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

def predict_next_word(model, context, top_k=1):
    """
    Predicts the next word based on the given context.

    Parameters
    model : dict
        n-gram language model
    context : tuple
        previous word(s)
    top_k : int
        number of predictions to return

    Returns
    list of tuples (word, probability)
    """

    if context not in model:
        return []

    next_words = model[context]

    # sort by probability
    sorted_words = sorted(
        next_words.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_words[:top_k]

def generate_sentence(model, start_context, max_length=20):
    """
    Generates a sentence using an n-gram model.

    Parameters
    model : dict
    start_context : tuple
    max_length : int

    Returns: str
    """

    context = start_context
    sentence = list(context)

    for _ in range(max_length):

        if context not in model:
            break

        next_words = model[context]

        words = list(next_words.keys())
        probabilities = list(next_words.values())

        next_word = random.choices(words, probabilities)[0]

        if next_word == '<EOS>':
            break

        sentence.append(next_word)

        # update context
        context = tuple(sentence[-len(context):])

    return " ".join(sentence)


def compare_models(bigram_model, trigram_model):
    print("\nVoorbeeld voorspellingen:\n")

    contexts = [
        ('the',),
        ('in',),
        ('of',)
    ]

    for context in contexts:

        print("Context:", context)

        bigram_pred = predict_next_word(bigram_model, context, top_k=3)
        print("Bigram:", bigram_pred)

        trigram_context = context
        if len(context) == 1:
            trigram_context = ('the', context[0])

        trigram_pred = predict_next_word(trigram_model, trigram_context, top_k=3)
        print("Trigram:", trigram_pred)

        print()


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

    # Example prediction
    print("\nNext word prediction\n")

    context = ('the',)
    predictions = predict_next_word(bigram_model, context, top_k=5)

    for word, prob in predictions:
        print(word, prob)

    # Sentence generation
    print("\nGenerated sentence\n")

    sentence = generate_sentence(bigram_model, ('the',))
    print(sentence)

    # Compare models
    compare_models(bigram_model, trigram_model)

    return {
        'bigram': bigram_model,
        'bigram_laplace': bigram_model_laplace,
        'trigram': trigram_model,
        'trigram_laplace': trigram_model_laplace,
        'vocabulary': vocabulary
    }
    
if __name__ == '__main__':
    main()
