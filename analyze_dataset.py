import re
import statistics

import spam_parser

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="white", font_scale=1.5, font="Times New Roman")

WORD_PATTERN = re.compile("\w+")


def tokenize(text):
    return WORD_PATTERN.findall(text)


def build_vocabulary(sentences):
    return set(word for sentence in sentences for word in tokenize(sentence.lower()))


def vocabulary_overlap(vocab_a, vocab_b):
    return len(vocab_a.intersection(vocab_b)) / len(vocab_a.union(vocab_b))


if __name__ == "__main__":
    spam, ham = spam_parser.parse_from_path(spam_parser.DEFAULT_LOCATION)

    spam_lengths = list(map(len, spam))
    ham_lengths = list(map(len, ham))

    print("Spam SMS lengths: {:.3f} ±{:.3f}".format(statistics.median(spam_lengths), statistics.stdev(spam_lengths)))
    print("Ham SMS lengths:  {:.3f} ±{:.3f}".format(statistics.median(ham_lengths), statistics.stdev(ham_lengths)))

    ax = sns.violinplot(data=[spam_lengths, ham_lengths])
    ax.set_xticklabels(["spam", "ham"])
    plt.show()

    spam_lengths_by_word = list(map(lambda x: len(tokenize(x)), spam))
    ham_lengths_by_word = list(map(lambda x: len(tokenize(x)), ham))

    print("Median words per Spam SMS: {:.3f} ±{:.3f}".format(statistics.median(spam_lengths_by_word), statistics.stdev(spam_lengths_by_word)))
    print("Median words per Ham SMS:  {:.3f} ±{:.3f}".format(statistics.median(ham_lengths_by_word), statistics.stdev(ham_lengths_by_word)))

    ax = sns.violinplot(data=[spam_lengths_by_word, ham_lengths_by_word])
    ax.set_xticklabels(["spam", "ham"])
    plt.show()

    spam_vocab = build_vocabulary(spam)
    ham_vocab = build_vocabulary(ham)

    print("Spam vocabulary length:", len(spam_vocab))
    print("Ham vocabulary length: ", len(ham_vocab))
    print("Vocabulary overlap: {:.3%}".format(vocabulary_overlap(spam_vocab, ham_vocab)))
