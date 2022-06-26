import spam_parser
import preprocessing as pre

import json
import re
import numpy as np
from sklearn import metrics, model_selection, naive_bayes, neural_network, decomposition
from sklearn.base import clone as model_clone
import sklearn.feature_extraction.text as text_extraction
import matplotlib.pyplot as plt

VECTORIZERS = {
    "count": text_extraction.CountVectorizer(stop_words=pre.SMS_STOP_WORDS),
    "tf-idf": text_extraction.TfidfVectorizer(stop_words=pre.SMS_STOP_WORDS),
}
MODELS = {"bayes": naive_bayes.MultinomialNB(), "ffn": neural_network.MLPClassifier([10])}


def plot_pca(name, embeddings, labels):
    # Support arbitrary labels.
    reverse_map = {hash(label) % 100000: label for label in labels}
    colors = [hash(label) % 100000 for label in labels]

    reduction = decomposition.PCA(n_components=2)
    reduced = reduction.fit_transform(embeddings)
    plt.title(f"{name} - PCA significance of shown dimensions:"
              f"{sum(reduction.explained_variance_ratio_):.3%}")
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, cmap="coolwarm")

    handles, labels = scatter.legend_elements()
    # labels are returned as strings with the hash, convert them back to the original labels.
    labels = [reverse_map[int(re.findall(r"\d+", label)[0])] for label in labels]
    plt.legend(handles, labels)
    plt.show()


if __name__ == "__main__":
    spam, ham = spam_parser.parse_from_path(spam_parser.DEFAULT_LOCATION)

    spam = [pre.preprocess(line) for line in spam]
    ham = [pre.preprocess(line) for line in ham]

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

        # Shuffle the data!
        rng = np.random.default_rng(42)
        rng.shuffle(train)
        rng.shuffle(test)
        # By resetting the random number generator,
        # the labels are shuffled in the same way as their corresponding embeddings.
        rng = np.random.default_rng(42)
        rng.shuffle(train_labels)
        rng.shuffle(test_labels)

        for vectorizer_name, vectorizer in VECTORIZERS.items():
            # Create a fresh instance of the vectorizer.
            vectorizer = model_clone(vectorizer)
            # Build vocab for the whole set to avoid out-of-vocabulary problems.
            # Implicit preprocessing by the vectorizer:
            # - tokenize by only selecting sequences word-characters with a minimum length of 2
            # - lowercase all tokens
            vectorizer.fit(train)
            vectorizer.fit(test)

            train_embeddings = vectorizer.transform(train).toarray()
            test_embeddings = vectorizer.transform(test).toarray()

            for model_name, model in MODELS.items():
                # Create a fresh instance of the model architecture.
                model = model_clone(model)
                print(f"Running {model_name} with {vectorizer_name} on split {i + 1} of {kf.get_n_splits()}.")
                model.fit(train_embeddings, train_labels)
                predictions = model.predict(test_embeddings)
                results[model_name][vectorizer_name].append((test_labels, predictions))

    reports = {}
    for model_name, model_results in results.items():
        for vectorizer_name, all_splits in model_results.items():
            true_labels = []
            predicted_labels = []
            for labels, preds in all_splits:
                true_labels.extend(labels)
                predicted_labels.extend(preds)
            print(f"Results of {model_name} with {vectorizer_name}:")
            print(metrics.classification_report(true_labels, predicted_labels, digits=5))
            reports.setdefault(model_name, {})
            reports[model_name][vectorizer_name] = metrics.classification_report(true_labels, predicted_labels,
                                                                                 output_dict=True)

    with open("classification_report.json", "w") as f:
        json.dump(reports, f, indent=4)

    # PCA visualization of vectorization methods.
    labels = ["spam"] * len(spam) + ["ham"] * len(ham)
    for vectorizer_name, vectorizer in VECTORIZERS.items():
        embeddings = vectorizer.fit_transform(np.concatenate([spam, ham])).toarray()
        plot_pca(vectorizer_name, embeddings, labels)

    # Features from the features analyzed in analyze_dataset.py
    # features = [(len(sms),
    #              len(pre.tokenize(sms)),
    #              len(pre.URL_PATTERN.findall(sms)),
    #              len(pre.TELEPHONE_PATTERN.findall(sms)),
    #              len(pre.MONEY_PATTERN.findall(sms)),
    #              len(pre.POBOX_PATTERN.findall(sms))) for sms in np.concatenate([spam, ham])]
    # plot_pca("custom_features", features, labels)
