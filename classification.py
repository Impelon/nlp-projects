import spam_parser

import numpy as np
from sklearn import model_selection, naive_bayes, neural_network

if __name__ == "__main__":
    spam, ham = spam_parser.parse_from_path(spam_parser.DEFAULT_LOCATION)

    # TODO: preprocessing

    spam = np.array(spam)
    ham = np.array(ham)

    # Splitting the dataset into 5 parts results in 20% test data.
    # To ensure equal distribution of spam and ham in test-data we split each category individually.
    kf = model_selection.KFold(n_splits=5)
    for spam_split, ham_split in zip(kf.split(spam), kf.split(ham)):
        # The splits contain the indicies for each array as a tuple like (train_indices, test_indeces).
        spam_train = spam[spam_split[0]]
        spam_test = spam[spam_split[1]]
        ham_train = ham[ham_split[0]]
        ham_test = ham[ham_split[1]]

        print(len(spam_train), len(spam_test), len(ham_train), len(ham_test))
        train = np.concatenate([spam_train, ham_train])
        test = np.concatenate([spam_test, ham_test])
        print(len(train), len(test))
        print(test)
