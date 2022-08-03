import os
import string
from pathlib import Path

import dataset_loader
from tokenizers import *
from embeddings import *

import keras
import keras_nlp
import tensorflow as tf

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

# See https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec for available options.
Word2VecEmbeddings.default_trainer_options.update({
    "hs": 1, # enable historical softmax
    "workers": len(os.sched_getaffinity(0)),
})

# See https://github.com/google/sentencepiece/blob/master/doc/options.md for available options.
SentencePieceTokenizer.default_trainer_options.update({
    "vocab_size": 16000,
    "required_chars": string.ascii_letters + string.digits + string.punctuation,
})

PIPELINES_PATH = Path(__file__).parent / "pipelines"
TOKENIZERS_PATH = PIPELINES_PATH / "tokenizers"
MODELS_PATH = PIPELINES_PATH / "models"
EMBEDDINGS_PATH = PIPELINES_PATH / "embeddings"

TOKENIZER_INITS = {
    "subword": lambda lang:
        SentencePieceTokenizer(TOKENIZERS_PATH / f"sentencepiece_{lang}.proto", name=f"{lang}_tokenizer"),
    "character": lambda lang: ByteTokenizer(lowercase=False, name=f"{lang}_tokenizer"),
}

PIPELINE_PRESETS = {
    "tiny": {
        "tokenizer_init": TOKENIZER_INITS["subword"],
        "intermediate_state_size": 32,
        "encoder_options": {
            "embeddings_size": 32,
        },
        "decoder_options": {
            "embeddings_size": 32,
        },
    },
    "small": {
        "tokenizer_init": TOKENIZER_INITS["subword"],
        "intermediate_state_size": 32,
        "encoder_options": {
            "embeddings_size": 256,
        },
        "decoder_options": {
            "embeddings_size": 256,
        },
    },
    "base": {
        "tokenizer_init": TOKENIZER_INITS["subword"],
        "intermediate_state_size": 128,
        "encoder_options": {
            "embeddings_size": 256,
            "embeddings_type": "word2vec_cbow",
            "embeddings_trainable": False,
        },
        "decoder_options": {
            "embeddings_size": 256,
            "embeddings_type": "word2vec_cbow",
            "embeddings_trainable": False,
        },
    },
    "base_attn": {
        "tokenizer_init": TOKENIZER_INITS["subword"],
        "intermediate_state_size": 128,
        "encoder_options": {
            "embeddings_size": 256,
            "embeddings_type": "word2vec_cbow",
            "embeddings_trainable": False,
        },
        "decoder_options": {
            "embeddings_size": 256,
            "embeddings_type": "word2vec_cbow",
            "embeddings_trainable": False,
            "use_attention": True,
        },
    },
    "byte_tiny": {
        "tokenizer_init": TOKENIZER_INITS["character"],
        "intermediate_state_size": 32,
        "encoder_options": {
            "embeddings_size": 32,
        },
        "decoder_options": {
            "embeddings_size": 32,
        },
    },
    "byte_base": {
        "tokenizer_init": TOKENIZER_INITS["character"],
        "intermediate_state_size": 128,
        "encoder_options": {
            "embeddings_size": 32,
        },
        "decoder_options": {
            "embeddings_size": 32,
        },
    },
}


def assert_valid_languages(from_language, to_language, pivot_language=None):
    if pivot_language and pivot_language != "en":
        raise NotImplementedError("Can only use 'en' as a pivot language!")
    if "en" not in (from_language, to_language, pivot_language):
        raise NotImplementedError("Can only translate to or from english, "
                                  "but none of the specified languages were 'en'! "
                                  "You may want to use: pivot_language='en'")
    if not pivot_language and to_language == from_language:
        raise ValueError("Cannot translate from a language to itself without a pivot language!")


def translation_pipeline(from_language, to_language, pivot_language=None, tokenizer_init=None,
                         optimizer=None, generation_options=None, pivot_dual_core=True, **model_options):
    if pivot_language and pivot_dual_core:
        # Create "dual-core" pivot model. (Does not require additional training. Also, it is not a keras.Model.)
        assert_valid_languages(from_language, to_language, pivot_language)
        to_pivot_model = translation_pipeline(from_language, pivot_language, tokenizer_init=tokenizer_init,
                                              optimizer=optimizer, generation_options=generation_options,
                                              **model_options)
        from_pivot_model = translation_pipeline(pivot_language, to_language, tokenizer_init=tokenizer_init,
                                                optimizer=optimizer, generation_options=generation_options,
                                                **model_options)
        return PivotTranslationPipeline(to_pivot_model, from_pivot_model)

    layers = translation_layers(from_language, to_language, pivot_language=pivot_language,
                                tokenizer_init=tokenizer_init, optimizer=optimizer, **model_options)
    from_tokenizer, to_tokenizer, core = layers
    model = TranslationPipeline(from_tokenizer, to_tokenizer, core, generation_options)
    model.remember_languages(from_language, to_language)
    if not optimizer:
        optimizer = "adam"
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


def translation_layers(from_language, to_language, pivot_language=None, tokenizer_init=None,
                       optimizer=None, **model_options):
    assert_valid_languages(from_language, to_language, pivot_language)
    if pivot_language:
        # Create "split-core" pivot model. (Would require additional training.)
        layers = translation_layers(from_language, pivot_language, tokenizer_init=tokenizer_init, **model_options)
        from_tokenizer, _, to_pivot_core = layers
        layers = translation_layers(pivot_language, to_language, tokenizer_init=tokenizer_init, **model_options)
        _, to_tokenizer, from_pivot_core = layers
        return from_tokenizer, to_tokenizer, combine_pivot_cores(to_pivot_core, from_pivot_core)

    non_english = from_language
    if non_english == "en":
        non_english = to_language

    # Load tokenizers.
    if not tokenizer_init:
        tokenizer_init = TOKENIZER_INITS["subword"]
    tokenizers = {}
    dataset = None
    for language in (from_language, to_language):
        tokenizer = tokenizer_init(language)
        tokenizers[language] = tokenizer
        if tokenizer.is_loaded:
            continue
        if not dataset:
            dataset = dataset_loader.load_dataset(non_english, merge_empty=False, shuffle=True)
        language_index = 1 if language == "en" else 0
        corpus = map(lambda x: x[language_index], dataset)
        tokenizer.try_train(corpus)

    # Load pre-trained embeddings, if any.
    for language, options in ([from_language, model_options.get("encoder_options", {})],
                              [to_language, model_options.get("decoder_options", {})]):
        type = options.pop("embeddings_type", None)
        size = options["embeddings_size"]
        path = EMBEDDINGS_PATH / f"{type}_{size}_{language}"
        if type == "word2vec_cbow":
            embeddings = Word2VecEmbeddings(path, size, sg=0)
        elif type == "word2vec_skip":
            embeddings = Word2VecEmbeddings(path, size, sg=1)
        elif type is None:
            continue
        else:
            raise NotImplementedError("Unknown embedding type: " + repr(type))
        if not embeddings.is_loaded:
            if not dataset:
                dataset = dataset_loader.load_dataset(non_english, merge_empty=False, shuffle=True)
            language_index = 1 if language == "en" else 0
            corpus = list(map(lambda x: x[language_index], dataset))
            embeddings.try_train(tokenizers[language], corpus)
        options["embeddings_initializer"] = embeddings.keras_embeddings_initializer()

    core = build_model(tokenizers[from_language].vocabulary_size(), tokenizers[to_language].vocabulary_size(),
                       **model_options)
    if not optimizer:
        optimizer = "adam"
    core.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return tokenizers[from_language], tokenizers[to_language], core


def combine_pivot_cores(to_pivot_core, from_pivot_core):
    """
    Alternative way to create a pivot-based translation model:
    This creates a new core by connecting the first core's encoder to the second core's decoder
    through a mediation layer, which tries to convert the internal state of each core.

    Naturally, such an intermediate layer needs additional training.
    """
    encoder, _ = split_encoder_decoder(to_pivot_core)
    _, decoder = split_encoder_decoder(from_pivot_core)
    decoder_state_size = decoder.input_shape[1][-1]

    # split_encoder_decoder and therefore TranslationPipeline relies on the fact
    # that a core only consists of an encoder and decoder.
    # Thus, we first need to create a new "encoder"-layer which always includes the mediation layer,
    # even if split later on.
    from_tokens = encoder.input
    states, (last_outputs, final_memory) = encoder(from_tokens)
    states = keras.layers.Dense(decoder_state_size, name="intermediate_states_mediation")(states)
    last_outputs = keras.layers.Dense(decoder_state_size, name="previous_output_mediation")(last_outputs)
    final_memory = keras.layers.Dense(decoder_state_size, name="initial_memory_mediation")(final_memory)
    final_state = [last_outputs, final_memory]
    mediated_encoder = keras.Model(inputs=encoder.input, outputs=[states, final_state], name="encoder")

    from_tokens, to_tokens = to_pivot_core.input
    states, encoder_state = mediated_encoder(from_tokens)
    decoder_output = decoder([to_tokens, states, encoder_state])
    return keras.Model(inputs=[from_tokens, to_tokens], outputs=decoder_output, name="pivot_split_core")


def train_pipeline(model, from_texts, to_texts, epochs=1):
    """
    Trains a TranslationPipeline model.

    We could train the pipeline by training the core-model with tokenized inputs directly,
    or by using the pipeline's capabilities to apply tokenization automatically, as we do here.
    Furthermore, the pipeline discards the optional "attention_weights" output from the core,
    which should be ignored for the model loss.
    All in all, this means that it is more convenient to train using the pipeline, rather than the core.

    Args:
        model: A TranslationPipeline to train.
        from_texts: A sequence of texts written in the language the pipeline translates from.
        to_texts: A sequence of texts written in the language the pipeline translates to.
            Must be of the same length as from_texts.
    """
    # NOTE:
    # This implementation is inefficient, because each training example is fed to the model one-by-one.
    # Also this triggeres multiple warnings
    # Normally one would feed a batch of examples at a time, but for that, the inputs and labels need to be padded.
    # However, our models cannot ignore padding yet.
    from_texts = tf.convert_to_tensor(from_texts)
    to_texts = tf.convert_to_tensor(to_texts)
    train_labels = model.to_tokenizer(to_texts)[:, 1:]
    for epoch in tqdm(range(epochs), desc="Training epoch"):
        for i in tqdm(range(train_labels.get_shape()[0])):
            model.fit([from_texts[i:i+1], to_texts[i:i+1]], train_labels[i:i+1].to_tensor(), verbose=0)


class PivotTranslationPipeline:

    def __init__(self, to_pivot_model, from_pivot_model):
        self.to_pivot_model = to_pivot_model
        self.from_pivot_model = from_pivot_model

    def translate(self, input_text, **generation_options):
        pivot_text = self.to_pivot_model.translate(input_text, **generation_options)
        output_text = self.from_pivot_model.translate(pivot_text, **generation_options)
        return output_text


class TranslationPipeline(keras.Model):

    def __init__(self, from_tokenizer, to_tokenizer, core, default_generation_options=None, **kwargs):
        super().__init__(**kwargs)
        self._from_language = None
        self._to_language = None
        self.from_tokenizer = from_tokenizer
        self.to_tokenizer = to_tokenizer
        self.core = core
        self.encoder, self.decoder = split_encoder_decoder(self.core)
        self.default_generation_options = {
            "prompt": [to_tokenizer.bos_id],
            "max_length": 64,
            "num_beams": 4,
            "end_token_id": to_tokenizer.eos_id,
            "from_logits": True,
        }
        if default_generation_options:
            self.default_generation_options.update(default_generation_options)

    @property
    def from_language(self):
        return self._from_language

    @property
    def to_language(self):
        return self._to_language

    def remember_languages(self, from_language, to_language):
        """
        Instruct the pipeline to remember its languages so they can be accessed externally, if needed.
        (It itsself does not actually need to know which languages are involved.)
        """
        self._from_language = from_language
        self._to_language = to_language

    def get_config(self):
        config = super().get_config()
        config.update({
            "default_generation_options": self.default_generation_options,
        })
        if self.from_language:
            config["from_language"] = self.from_language
        if self.to_language:
            config["to_language"] = self.to_language
        return config

    def translate(self, input_text, return_attention=False, **generation_options):
        input_tokens = self.from_tokenizer([input_text])
        intermediate_states, initial_state = self.encoder(input_tokens)

        # The beam-search could also be performed using the core-model directly.
        # But this would be inefficient:
        # In each iteration the encoder receives the same input-text and produces the same outputs.
        # Instead, we compute the outputs of the encoder once,
        # and provide them to the decoder in each beam-search iteration.

        if return_attention:
            attentions = []
            def token_logits(tokens):
                logits, attention_weights = self.decoder([tokens, intermediate_states, initial_state])
                attentions.append(attention_weights)
                return logits.to_tensor()[:, -1, :]
        else:
            def token_logits(tokens):
                logits = self.get_logits(self.decoder([tokens, intermediate_states, initial_state]))
                return logits[:, -1, :]

        gen_kwargs = self.default_generation_options.copy()
        gen_kwargs.update(generation_options)
        output_tokens = keras_nlp.utils.beam_search(token_logits, **gen_kwargs)
        output_text = self.to_tokenizer(output_tokens, mode="detokenize")
        if return_attention:
            attention_stack = tf.concat(attentions, axis=1)
            return output_text.numpy().decode(), attention_stack
        return output_text.numpy().decode()

    @tf.function
    def get_logits(self, decoder_output):
        logits = decoder_output
        if isinstance(self.decoder.output_shape, list):
            logits = decoder_output[0]
        if isinstance(logits, tf.RaggedTensor):
            logits = logits.to_tensor()
        return logits

    def call(self, inputs, shift_decoder_inputs=True, return_attention=False, *args, **kwargs):
        from_text, to_text = inputs
        from_tokens = self.from_tokenizer(from_text)
        to_tokens = self.to_tokenizer(to_text)

        if shift_decoder_inputs:
            to_tokens = to_tokens[:, :-1]

        decoder_output = self.core([from_tokens, to_tokens], *args, **kwargs)
        if return_attention:
            return decoder_output
        return self.get_logits(decoder_output)


def build_model(from_vocab_size, to_vocab_size, intermediate_state_size, encoder_options=None, decoder_options=None):
    if not encoder_options:
        encoder_options = {}
    if not decoder_options:
        decoder_options = {}

    # To use the final state of the encoder,
    # the decoder needs to use the same size for its initial layer.
    # As such both models are build to target the same intermediate state size.
    encoder = build_encoder(from_vocab_size, intermediate_state_size, **encoder_options)
    decoder = build_decoder(to_vocab_size, intermediate_state_size, **decoder_options)

    from_tokens = keras.layers.Input(shape=(None,), name="from_tokens")
    to_tokens = keras.layers.Input(shape=(None,), name="to_tokens")
    # The encoder output is a sequence of all states + final hidden state of the encoder.
    states, encoder_state = encoder(from_tokens)
    # The decoder output are logits for each token + optional attention weights.
    decoder_output = decoder([to_tokens, states, encoder_state])
    return keras.Model(inputs=[from_tokens, to_tokens], outputs=decoder_output, name="core")


def build_embedding_layer(vocab_size, embeddings_size, embeddings_initializer=None, embeddings_trainable=True):
    return keras.layers.Embedding(vocab_size, embeddings_size,
                                  embeddings_initializer=embeddings_initializer, trainable=embeddings_trainable)


def build_encoder(vocab_size, state_size, **embedding_options):
    tokens = keras.layers.Input((None,), name="encoder_tokens")
    x = tokens
    x = build_embedding_layer(vocab_size, **embedding_options)(x)
    x, last_outputs, final_memory = keras.layers.LSTM(state_size, return_sequences=True, return_state=True)(x)
    final_state = [last_outputs, final_memory]
    return keras.Model(inputs=tokens, outputs=[x, final_state], name="encoder")


def build_decoder(vocab_size, state_size, use_attention=False, **embedding_options):
    tokens = keras.layers.Input((None,), name="decoder_tokens")
    intermediate_states = keras.layers.Input((None, state_size), name="intermediate_states")
    initial_state = [keras.layers.Input((state_size,)), keras.layers.Input((state_size,))]

    outputs = []
    x = tokens
    x = build_embedding_layer(vocab_size, **embedding_options)(x)
    x = keras.layers.LSTM(state_size, return_sequences=True)(x, initial_state=initial_state)
    if use_attention:
        decoder_states = x
        query = decoder_states
        value = intermediate_states
        # The scaled dot-product attention used in transformer-architectures
        # could be constructed by freezing the scale-variable of a keras.layers.Attention.
        # Or perhaps even better by using keras.layers.MultiHeadAttention with one head.
        # But let's just use Bahdanau-style additive attention.
        query = keras.layers.Dense(state_size, use_bias=False)(query)
        key = keras.layers.Dense(state_size, use_bias=False)(value)
        x, attention_weights = keras.layers.AdditiveAttention()([query, value, key], return_attention_scores=True)
        outputs.append(attention_weights)
        # Combine attention and (recurrent) decoder states.
        x = keras.layers.Concatenate()([x, decoder_states])
        x = keras.layers.Dense(state_size, activation="tanh", use_bias=False)(x)
    x = keras.layers.Dense(vocab_size, name="logits_layer")(x)
    outputs.append(x)
    return keras.Model(inputs=[tokens, intermediate_states, initial_state], outputs=outputs[::-1], name="decoder")


def split_encoder_decoder(core):
    encoder = keras.Model(inputs=core.get_layer("encoder").input,
                          outputs=core.get_layer("encoder").output, name="encoder")
    decoder = keras.Model(inputs=core.get_layer("decoder").input,
                          outputs=core.get_layer("decoder").output, name="decoder")
    return encoder, decoder
