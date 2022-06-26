import spam_parser
import preprocessing

import multiprocessing
import random
import heapq
import json
from math import log2
from collections import Counter

TQDM_AVAILABLE = True
try:
    from tqdm import tqdm
    tqdm.set_lock(multiprocessing.Lock())
except ImportError:
    TQDM_AVAILABLE = False

random.seed("Hello world!")


def word_embedding(tokenized_corpus, a, n=1):
    """
    Computes the occurrences of other tokens (from the corpus) in all contexts of size `n` of the token `a`.
    """
    v = Counter()
    for tokens in tokenized_corpus:
        for index, token in enumerate(tokens):
            if token == a:
                context = tokens[max(index - n, 0):index + n + 1]
                v.update(context)
    del v[a]
    return v


def pmi_from_embeddings(embeddings, a, b):
    """
    Computes the pointwise mutual information between token `a` and `b` from the given vector-representations.
    """
    # PMI(a, b) = log2(P(a, b) / P(a) * Q(b))
    # P(x)      = "# of x in all contexts"            / "# of unigrams across all contexts"
    # Q(x)      = "# of unigrams in the context of x" / "# of unigrams across all contexts"
    # P(x, y)   = "# of x in the context of y"        / "# of unigrams across all contexts"
    context_sum = sum(sum(vector.values()) for vector in embeddings.values())
    p_a_b = embeddings[a][b] / context_sum
    p_a = sum(embeddings[a].values()) / context_sum
    q_b = sum(vector[b] for vector in embeddings.values()) / context_sum
    return log2(p_a_b / (p_a * q_b))


def compute_term_embeddings(tokenized_corpus, n=1, laplace_smoothing=0):
    """
    Computes the word-embeddings with context-size n for every element in the set of tokens in the corpus.
    """
    embeddings = {}
    vocab = set(token for sentence in tokenized_corpus for token in sentence)
    for token in vocab:
        embeddings[token] = word_embedding(tokenized_corpus, token, n=n)
        for t in vocab:
            embeddings[token][t] += laplace_smoothing
    return embeddings


def find_most_similar(embeddings, word, k=10, iterator_wrapper=None):
    """
    Find the most similar k words using PMI and the given embeddings.
    """
    if not iterator_wrapper:
        def iterator_wrapper(x):
            return x
    # Maintain a heap of the most similar tokens.
    heap = []
    for token in iterator_wrapper(embeddings.keys()):
        # (pmi_from_embeddings is symmetric, order of word and token does not matter.)
        score = pmi_from_embeddings(embeddings, word, token)
        # Add scored token to the heap for this word.
        # The current length of the heap serves as a tie-breaker.
        # heapq implements a min heap, so we need to invert the score to find the best matches later.
        heapq.heappush(heap, (-score, len(heap), token))

    result = {}
    for i in range(k):
        # Popping a value from this heap will give us the element with the lowest inverted score,
        # aka. the most similar token still on the heap.
        inv_score, _, token = heapq.heappop(heap)
        result[token] = -inv_score
    return result


def test_pmi():
    tokenized_corpus = [["text", "is", "a", "complex", "human", "language", "representation"],
                        ["natural", "human", "language", "is", "complex", "and", "also", "is", "diverse"]]
    embeddings = compute_term_embeddings(tokenized_corpus, n=2)
    assert round(pmi_from_embeddings(embeddings, "human", "is"), 3) == \
           round(log2(1 / 52 / (7 / 52 * 10 / 52)), 3)
    assert round(pmi_from_embeddings(embeddings, "human", "natural"), 3) == \
           round(log2(1 / 52 / (7 / 52 * 2 / 52)), 3)
    embeddings = compute_term_embeddings(tokenized_corpus, n=2, laplace_smoothing=1)
    assert round(pmi_from_embeddings(embeddings, "human", "is"), 3) == \
           round(log2(2 / 173 / (18 / 173 * 21 / 173)), 3)
    assert round(pmi_from_embeddings(embeddings, "human", "natural"), 3) == \
           round(log2(2 / 173 / (18 / 173 * 13 / 173)), 3)


if __name__ == "__main__":
    test_pmi()
    spam, ham = spam_parser.parse_from_path(spam_parser.DEFAULT_LOCATION)

    print("Building corpus...")
    corpus = spam + ham
    corpus = [preprocessing.preprocess(sms).casefold() for sms in corpus]
    print("Tokenizing corpus...")
    tokenized_corpus = [preprocessing.tokenize(sms) for sms in corpus]
    print("Removing stop words...")
    tokenized_corpus = [list(filter(lambda x: x not in preprocessing.SMS_STOP_WORDS, sms)) for sms in tokenized_corpus]
    print("Choosing words...")
    chosen_words = random.choices([token for sentence in tokenized_corpus for token in sentence], k=10)
    print("Computing word-embeddings...")
    embeddings = compute_term_embeddings(tokenized_corpus, laplace_smoothing=1)


    def partial_most_similar(index):
        word = chosen_words[index]
        progress_update = None
        if TQDM_AVAILABLE:
            def progress_update(x):
                tqdm.set_lock(tqdm.get_lock())
                # Using any other position than 0 results in jumping progress bars.
                return tqdm(x, position=0, desc=f"Computing similarities for {word}")
        return find_most_similar(embeddings, word, iterator_wrapper=progress_update)


    print("Computing most similar words...")
    # Computation of PMI is slow, so on multicore systems this will speed up the processing!
    with multiprocessing.Pool(10) as pool:
        most_similar = pool.map(partial_most_similar, range(len(chosen_words)))
    most_similar = {k: v for k, v in zip(chosen_words, most_similar)}
    with open("most_similar_words.json", "w") as f:
        json.dump(most_similar, f, indent=4)
