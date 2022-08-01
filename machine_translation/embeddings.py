import abc
from pathlib import Path
from collections import Counter, OrderedDict

import gensim
import keras

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

__all__ = ["Word2VecEmbeddings", "KeyedVectorsEmbeddings", "EmbeddingsExtension"]


class OrderedCounter(Counter, OrderedDict):
    pass


class EpochPrinter(gensim.models.callbacks.CallbackAny2Vec):

    def __init__(self, show_loss=True):
        self.epoch = 0
        self.show_loss = show_loss

    def on_epoch_end(self, model):
        status = f"Epoch #{self.epoch} done."
        if self.show_loss:
            status += f" loss: {model.get_latest_training_loss()}"
        print(status)
        self.epoch += 1


class EmbeddingsExtension(abc.ABC):

    """
    An abstract class for extensions to add methods for training and loading.
    We use this to provide a uniform interface to Gensim's vectorizers,
    regardless if they require training or not.

    To use the embeddings, one should initialize it and call `try_train` in case training is required.
    Afterwards it is guaranteed to be loaded, except if an exception is raised.
    """

    @property
    @abc.abstractmethod
    def wv(self):
        """
        Return the underlying KeyedVectors instance.
        """
        return None

    def keras_embeddings_initializer(self):
        return keras.initializers.Constant(self.wv.vectors)

    @property
    def is_loaded(self):
        return True

    def try_load(self):
        """
        Load the embeddings, if it is not already loaded.

        Returns:
            A boolean whether an action was performed.
        """
        if self.is_loaded:
            return False
        return self._load()

    def _load(self):
        return False

    def try_train(self, tokenizer, corpus):
        """
        Train the embeddings, if it is not already loaded.

        Args:
            tokenizer: A loaded tokenizer that is used to tokenize the corpus.
                The resulting embeddings will have the same vocabulary as the tokenizer.
            corpus: A sequence of strings to train on.

        Returns:
            A boolean whether an action was performed.
        """
        if self.is_loaded:
            return False
        return self._train(tokenizer, corpus)

    def chunked_id_to_token(self, tokenizer, id_corpus, chunksize=8192):
        """
        Apply tokenizer.id_to_token chunkwise and yield the result.
        Limits maximum memory consumption.
        """
        for i in range(0, id_corpus.get_shape()[0], chunksize):
            tokenized_chunk = tokenizer.id_to_token(id_corpus[i:i+chunksize])
            yield from tokenized_chunk
            del tokenized_chunk


    def count_token_frequencies(self, vocabulary, tokenized_corpus):
        """
        Count and return the number of occurences of each token in the corpus.

        The order of the vocabulary is preserved, however,
        every word from the vocabulary is given a minimum count of 1.
        """
        frequencies = OrderedCounter(vocabulary)
        for tokens in tqdm(tokenized_corpus, desc="Counting tokens"):
            frequencies.update(tokens)
        return frequencies

    def train(self, vocabulary, corpus):
        return False


class KeyedVectorsEmbeddings(EmbeddingsExtension, abc.ABC):

    def __init__(self, path):
        self.path = Path(path)
        self._keyedvectors = None

    @property
    def wv(self):
        return self._keyedvectors

    @property
    def is_loaded(self):
        return bool(self._keyedvectors)

    def _load(self):
        if not self.path.exists():
            return False
        self._keyedvectors = gensim.models.KeyedVectors.load(str(self.path))
        return True


class Word2VecEmbeddings(KeyedVectorsEmbeddings):

    default_trainer_options = {
        "compute_loss": True,
    }

    def __init__(self, path, vector_size, **trainer_options):
        super().__init__(path)
        self.vector_size = vector_size
        if not trainer_options:
            trainer_options = Word2VecEmbeddings.default_trainer_options
        self.trainer_options = trainer_options
        self.trainer_options.setdefault("callbacks", [EpochPrinter()])
        self.try_load()

    def _train(self, tokenizer, corpus):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        id_corpus = tokenizer.tokenize(corpus)
        trainer = gensim.models.Word2Vec(vector_size=self.vector_size, min_count=0, sorted_vocab=False, **self.trainer_options)
        # Make sure the embeddings use the same vocabulary as the tokenizer.
        tokenized_corpus = self.chunked_id_to_token(tokenizer, id_corpus)
        token_frequencies = self.count_token_frequencies(tokenizer.get_vocabulary(), tokenized_corpus)
        trainer.build_vocab_from_freq(token_frequencies, corpus_count=len(corpus), update=False)
        # Train and save the embeddings.
        opts = {k: v for k, v in self.trainer_options.items() if k in ["compute_loss", "callbacks"]}
        # Apparently gensim does not remember the above two options if not trained during initialization.
        for epoch in range(trainer.epochs):
            tokenized_corpus = tqdm(self.chunked_id_to_token(tokenizer, id_corpus),
                                    desc=f"Epoch #{epoch} (loss {trainer.get_latest_training_loss()})", total=len(corpus))
            trainer.train(tokenized_corpus, total_examples=trainer.corpus_count, epochs=1, **opts)
        wv = trainer.wv
        del trainer
        wv.save(str(self.path))
        # We could also use the vectors directly,
        # but let's make sure they can be correctly loaded.
        del wv
        return self.try_load()
