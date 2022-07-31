import abc
from os import PathLike
from pathlib import Path

from sentencepiece import SentencePieceTrainer
from keras_nlp import tokenizers
import tensorflow as tf

__all__ = ["UnicodeCharacterTokenizer", "ByteTokenizer",
           "WordPieceTokenizer", "SentencePieceTokenizer", "TokenizerExtension"]


class TokenizerExtension(abc.ABC):

    """
    An abstract class for extensions to add methods for training and loading.
    We use this to provide a uniform interface to Keras' tokenizers,
    regardless if they require training or not.

    To use a tokenizer, one should initialize it and call `try_train` in case it needs training.
    Afterwards it is guaranteed to be loaded, except if an exception is raised.
    """

    @property
    def bos_id(self):
        return self._bos_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def add_bos(self):
        return self._bos_id >= 0

    @property
    def add_eos(self):
        return self._eos_id >= 0

    @property
    def is_loaded(self):
        return True

    def _add_special_tokens(self, inputs):
        """
        Add special tokens if needed.

        Can be called by subclasses to add BOS- and EOS-ids
        in case they do not provide a separate mechanism for it.

        Adapted from: https://github.com/keras-team/keras-nlp/blob/v0.3.0/keras_nlp/layers/start_end_packer.py
        """
        if not (self.add_bos or self.add_eos):
            return inputs
        input_is_1d = False
        if inputs.shape.rank == 1:
            input_is_1d = True
            inputs = tf.expand_dims(inputs, axis=0)
        batch_size = tf.shape(inputs)[0]
        if self.add_bos:
            start_token_id_tensor = tf.fill((batch_size, 1), self.bos_id)
            inputs = tf.concat([start_token_id_tensor, inputs], axis=-1)
        if self.add_eos:
            end_token_id_tensor = tf.fill((batch_size, 1), self.eos_id)
            inputs = tf.concat([inputs, end_token_id_tensor], axis=-1)
        if input_is_1d:
            inputs = tf.squeeze(inputs, axis=0)
        return inputs

    def _remove_special_tokens(self, inputs):
        """
        Reverse _add_special_tokens.
        """
        if not (self.add_bos or self.add_eos):
            return inputs
        input_is_1d = False
        if inputs.shape.rank == 1:
            input_is_1d = True
            inputs = tf.expand_dims(inputs, axis=0)

        def remove_if_needed(array):
            if self.add_bos and array[0] == self.bos_id:
                array = array[1:]
            if self.add_eos and array[-1] == self.eos_id:
                array = array[:-1]
            return array

        inputs = tf.map_fn(remove_if_needed, inputs)
        if input_is_1d:
            inputs = tf.squeeze(inputs, axis=0)
        return inputs

    def try_load(self):
        """
        Load the tokenizer, if it is not already loaded.

        Returns:
            A boolean whether an action was performed.
        """
        if self.is_loaded:
            return False
        return self._load()

    def _load(self):
        return False

    def try_train(self, corpus):
        """
        Train the tokenizer, if it is not already loaded.

        Args:
            corpus: A sequence of strings to train on.

        Returns:
            A boolean whether an action was performed.
        """
        if self.is_loaded:
            return False
        return self._train(corpus)

    def _train(self, corpus):
        return False

# Character-based tokenizers.


class UnicodeCharacterTokenizer(tokenizers.UnicodeCharacterTokenizer, TokenizerExtension):
    """
    Character-based tokenizer which converts characters to their Unicode codepoints
    and uses them as tokens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Control-bytes \x02 and \x03 are mostly unused in normal text.
        self._bos_id = 2
        self._eos_id = 3

    def tokenize(self, inputs):
        tokens = super().tokenize(inputs)
        return self._add_special_tokens(tokens)

    def detokenize(self, inputs):
        inputs = self._remove_special_tokens(inputs)
        return super().detokenize(inputs)


class ByteTokenizer(tokenizers.ByteTokenizer, TokenizerExtension):
    """
    Converts characters to their underlying byte-representation and uses that as tokens.
    This limits the vocabulary size to 256.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Control-bytes \x02 and \x03 are mostly unused in normal text,
        # and valid multibyte UTF-8 sequences never contain them.
        self._bos_id = 2
        self._eos_id = 3

    def tokenize(self, inputs):
        tokens = super().tokenize(inputs)
        return self._add_special_tokens(tokens)

    def detokenize(self, inputs):
        inputs = self._remove_special_tokens(inputs)
        return super().detokenize(inputs)

# Vocabulary-based tokenizers.


class WordPieceTokenizer(tokenizers.WordPieceTokenizer, TokenizerExtension):
    """
    Word-level tokenizer that uses an existing vocabulary.
    """
    pass  # special tokens & custom loading/training of vocabulary not implemented by us.


class SentencePieceTokenizer(tokenizers.SentencePieceTokenizer, TokenizerExtension):

    """
    Subword tokenizer that needs to be trained on a corpus.
    """

    default_trainer_options = {
        "normalization_rule_name": "identity",
        "unk_surface": "�",
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
    }

    def __init__(self, proto, trainer_options=None, add_bos=None, add_eos=None, *args, **kwargs):
        self.proto = proto
        if not trainer_options:
            trainer_options = SentencePieceTokenizer.default_trainer_options
        self.trainer_options = trainer_options
        self._bos_id = self.trainer_options.get("bos_id", -1)
        self._eos_id = self.trainer_options.get("eos_id", -1)
        if add_bos is None:
            add_bos = self.bos_id >= 0
        self._add_bos = add_bos
        if add_eos is None:
            add_eos = self.eos_id >= 0
        self._add_eos = add_eos
        self._args = args
        self._kwargs = kwargs
        self._is_loaded = False
        self.try_load()

    def get_config(self):
        config = super().get_config()
        config.update({"add_bos": self.add_bos, "add_eos": self.add_eos,
                       "bos_id": self.bos_id, "eos_id": self._eos_id})
        return config

    @property
    def add_bos(self):
        if self.is_loaded:
            return self._sentence_piece.add_bos
        return self._add_bos

    @add_bos.setter
    def add_bos(self, do_it):
        if self.is_loaded:
            self._sentence_piece.add_bos = do_it
        self._add_bos = do_it

    @property
    def add_eos(self):
        if self.is_loaded:
            return self._sentence_piece.add_eos
        return self._add_eos

    @add_eos.setter
    def add_eos(self, do_it):
        if self.is_loaded:
            self._sentence_piece.add_eos = do_it
        self._add_eos = do_it

    @property
    def is_loaded(self):
        return self._is_loaded

    def _load(self):
        """
        Load and initialize the tokenizer, if:
        - proto is an path to an existing non-empty file, or
        - proto is a string/bytes object with model-data.

        Raises:
            Exception: The tokenizer could not be loaded
                from the existing path or object containing model-data.
                Will not raise exceptions in case the path does not exist yet.
        """
        proto = self.proto
        if isinstance(proto, (str, PathLike)):
            proto_path = Path(proto)
            try:
                if not (proto_path.exists() and proto_path.stat().st_size):
                    return False
                # Keras' SentencePieceTokenizer accepts paths as strings only,
                # so convert it back from pathlib's Path before continuing.
                proto = str(proto_path)
            except OSError:
                if isinstance(proto, PathLike):
                    raise
                # Accessing the path pointed to by the string failed,
                # so assume it actually directly contains model-data.
        super().__init__(proto, *self._args, **self._kwargs)
        self._is_loaded = True
        self.add_bos = self._add_bos
        self.add_eos = self._add_eos
        return True

    def _train(self, corpus):
        proto_path = Path(self.proto)
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        # Will only be called if proto points to a not-yet existing path.
        with proto_path.open("wb") as proto_file:
            SentencePieceTrainer.train(sentence_iterator=corpus, model_writer=proto_file, **self.trainer_options)

        return self.try_load()
