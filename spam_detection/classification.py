import spam_parser

import numpy as np
from sklearn import metrics, model_selection, naive_bayes, neural_network
from sklearn.base import clone as model_clone
import sklearn.feature_extraction.text as text_extraction

VECTORIZERS = {"count": text_extraction.CountVectorizer(), "tf-idf": text_extraction.TfidfVectorizer(), }
MODELS = {"bayes": naive_bayes.MultinomialNB(), "ffn": neural_network.MLPClassifier([10])}

if __name__ == "__main__":
    spam, ham = spam_parser.parse_from_path(spam_parser.DEFAULT_LOCATION)

    # TODO: preprocessing

    spam = np.array(spam)
    ham = np.array(ham)

    # We keep a record of all predictions from every split in this dictionary.
    results = {model_name: {vectorizer_name: [] for vectorizer_name in VECTORIZERS} for model_name in MODELS}

    # Splitting the dataset into 5 parts results in 20% test data.
    # To ensure equal distribution of spam and ham in test-data we split each category individually.
    kf = model_selection.KFold(n_splits=5)
    for i, (spam_split, ham_split) in enumerate(zip(kf.split(spam), kf.split(ham))):
        # The splits contain the indices for each array as a tuple like (train_indices, test_indices).
        spam_train = spam[spam_split[0]]
        spam_test = spam[spam_split[1]]
        ham_train = ham[ham_split[0]]
        ham_test = ham[ham_split[1]]

        train = np.concatenate([spam_train, ham_train])
        test = np.concatenate([spam_test, ham_test])

        train_labels = ["spam"] * len(spam_train) + ["ham"] * len(ham_train)
        test_labels = ["spam"] * len(spam_test) + ["ham"] * len(ham_test)

        # TODO: shuffle

        for vectorizer_name, vectorizer in VECTORIZERS.items():
            # Build vocab for the whole set to avoid out-of-vocabulary problems.
            vectorizer.fit(train)
            vectorizer.fit(test)

            train_embeddings = vectorizer.transform(train).toarray()
            test_embeddings = vectorizer.transform(test).toarray()

            for model_name, model in MODELS.items():
                # Create a fresh instance of the model architecture.
                model = model_clone(model)
                print(f"Running {model_name} with {vectorizer_name} on split {i+1} of {kf.get_n_splits()}.")
                model.fit(train_embeddings, train_labels)
                predictions = model.predict(test_embeddings)
                results[model_name][vectorizer_name].append((test_labels, predictions))

    for model_name, model_results in results.items():
        for vectorizer_name, all_splits in model_results.items():
            true_labels = []
            predicted_labels = []
            for labels, preds in all_splits:
                true_labels.extend(labels)
                predicted_labels.extend(preds)
            print(f"Results of {model_name} with {vectorizer_name}:")
            print(metrics.classification_report(true_labels, predicted_labels))

    # TODO Maybe visualize




