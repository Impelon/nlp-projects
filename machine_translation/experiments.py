import json
import time
import statistics
import random
from pathlib import Path
from functools import lru_cache

import translation as t
import dataset_loader

import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.stem import SnowballStemmer

import matplotlib.pyplot as plt

import tensorflow as tf
import keras.utils
# Configure a seed so all experiments are reproducible.
keras.utils.set_random_seed(42)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

DEFAULT_SAMPLING_RATIO = 0.1
NLTK_DOWNLOAD_PATH = t.PIPELINES_PATH / "nltk"
NLTK_DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = Path(".").parent / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

nltk.data.path.insert(0, str(NLTK_DOWNLOAD_PATH))
nltk.download("punkt", download_dir=NLTK_DOWNLOAD_PATH)
nltk.download("wordnet", download_dir=NLTK_DOWNLOAD_PATH)
nltk.download("omw-1.4", download_dir=NLTK_DOWNLOAD_PATH)
word_tokenize = nltk.tokenize.word_tokenize

STEMMER_FOR_LANGUAGE = {
    "en": SnowballStemmer("english"),
    "nl": SnowballStemmer("dutch"),
    "sv": SnowballStemmer("swedish"),
    "ro": SnowballStemmer("romanian"),
    "de": SnowballStemmer("german"),
    "da": SnowballStemmer("danish"),
}


@lru_cache(maxsize=2)
def get_dataset(language, **loader_options):
    """
    Cacheing wrapper around the dataset loader.
    """
    return dataset_loader.load_dataset(language, **loader_options)


def get_correspondences(from_language, to_language, training_split=0.8, **loader_options):
    non_english = from_language
    if non_english == "en":
        non_english = to_language

    dataset = get_dataset(non_english, **loader_options)

    from_index, to_index = 0, 1
    if from_language == "en":
        from_index, to_index = 1, 0
    from_texts = tf.convert_to_tensor(list(map(lambda x: x[from_index], dataset)))
    to_texts = tf.convert_to_tensor(list(map(lambda x: x[to_index], dataset)))

    split_index = int(len(from_texts) * training_split)
    train_from_texts, test_from_texts = from_texts[:split_index], from_texts[split_index:]
    train_to_texts, test_to_texts = to_texts[:split_index], to_texts[split_index:]
    return train_from_texts, train_to_texts, test_from_texts, test_to_texts


def get_trained_pipeline(from_language, to_language, pipeline_preset_name,
                         sampling_ratio=DEFAULT_SAMPLING_RATIO, epochs=1):
    model_options = t.PIPELINE_PRESETS[pipeline_preset_name]
    return get_custom_trained_pipeline(from_language, to_language, pipeline_preset_name, sampling_ratio,
                                       epochs=epochs, **model_options)


def get_custom_trained_pipeline(from_language, to_language, name, sampling_ratio, epochs=1, **model_options):
    model_path = t.MODELS_PATH / f"{from_language}_{to_language}" / name
    model = t.translation_pipeline(from_language, to_language, **model_options)
    try:
        model.load_weights(str(model_path))
        return model
    except Exception:
        pass
    from_texts, to_texts, _, _ = get_correspondences(from_language, to_language, sampling_ratio=sampling_ratio)
    t.train_pipeline(model, from_texts, to_texts, epochs=epochs)
    model.save_weights(str(model_path))
    return model


def evaluate_pipeline(model, texts_to_translate, target_translations, **generation_options):
    hypotheses = []
    start = time.monotonic()
    for text in tqdm(texts_to_translate, desc="Generating translations"):
        hypotheses.append(model.translate(text, **generation_options))
    end = time.monotonic()
    results = evaluate_translations(target_translations, hypotheses, language=model.to_language)
    results["seconds_per_generated_translation"] = (end - start) / len(hypotheses)
    return results


def evaluate_translations(references, hypotheses, language="en", include_samples=10):
    # Convert tensors of byte-sequences to text if needed.
    if isinstance(references, tf.Tensor):
        references = map(bytes.decode, references.numpy())
    if isinstance(hypotheses, tf.Tensor):
        hypotheses = map(bytes.decode, hypotheses.numpy())

    results = {}
    smoothing = SmoothingFunction().method2  # ORANGE smoothing
    stemmer = STEMMER_FOR_LANGUAGE[language]

    references = [word_tokenize(reference) for reference in references]
    hypotheses = [word_tokenize(hypothesis) for hypothesis in hypotheses]

    if include_samples > 0:
        sample_indices = random.sample(range(len(references)), include_samples)
        results["samples"] = [(" ".join(references[i]), " ".join(hypotheses[i])) for i in sample_indices]

    scores = []
    for reference, hypothesis in tqdm(zip(references, hypotheses), desc="Calculating METEOR"):
        score = single_meteor_score(reference, hypothesis, stemmer=stemmer)
        scores.append(score)

    for k, v in statistics_for_collection(scores).items():
        results[f"meteor_{k}"] = v

    references = [[list(map(stemmer.stem, reference))] for reference in references]
    hypotheses = [list(map(stemmer.stem, hypothesis)) for hypothesis in hypotheses]

    bleu_score = corpus_bleu(references, tqdm(hypotheses, desc="Calculating BLEU"), smoothing_function=smoothing)
    results["corpus_bleu"] = bleu_score

    return results


def statistics_for_collection(values):
    metrics = {}
    metrics["mean"] = statistics.mean(values)
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    metrics["lower_quartile"], metrics["median"], metrics["higher_quartile"] = quartiles
    iqr = quartiles[2] - quartiles[0]
    bound_multiplier = 1.5
    metrics["lower_bound"] = min(filter(lambda x: x >= quartiles[0] - bound_multiplier * iqr, values))
    metrics["higher_bound"] = max(filter(lambda x: x <= quartiles[2] + bound_multiplier * iqr, values))
    metrics["amount_low_outliers"] = sum(map(lambda x: int(x < metrics["lower_bound"]), values))
    metrics["amount_high_outliers"] = sum(map(lambda x: int(x > metrics["higher_bound"]), values))
    metrics["min"] = min(values)
    metrics["max"] = max(values)
    metrics["std"] = statistics.pstdev(values)
    return metrics


def save_results(name, results):
    with (RESULTS_PATH / (name + ".json")).open("w") as f:
        json.dump(results, f, indent=4, sort_keys=True)


# Experiments


def experiment_embeddings():
    results = {}
    _, _, test_from_texts, test_to_texts = get_correspondences("nl", "en", training_split=0.8, sampling_ratio=0.01)
    genopts = {"num_beams": 2, "max_length": 16}
    model_configuration = t.PIPELINE_PRESETS["small"]

    model_configuration["encoder_options"]["embeddings_type"] = "word2vec_cbow"
    model_configuration["encoder_options"]["embeddings_trainable"] = False
    model_configuration["decoder_options"]["embeddings_type"] = "word2vec_cbow"
    model_configuration["decoder_options"]["embeddings_trainable"] = False
    model = get_custom_trained_pipeline("nl", "en", "exp_emb_word2vec_cbow", 0.01, **model_configuration)
    results["word2vec_cbow"] = evaluate_pipeline(model, test_from_texts, test_to_texts, **genopts)
    model_configuration["encoder_options"]["embeddings_type"] = "word2vec_skip"
    model_configuration["encoder_options"]["embeddings_trainable"] = False
    model_configuration["decoder_options"]["embeddings_type"] = "word2vec_skip"
    model_configuration["decoder_options"]["embeddings_trainable"] = False
    model = get_custom_trained_pipeline("nl", "en", "exp_emb_word2vec_skip", 0.01, **model_configuration)
    results["word2vec_skip"] = evaluate_pipeline(model, test_from_texts, test_to_texts, **genopts)
    model_configuration["encoder_options"]["embeddings_type"] = None
    model_configuration["encoder_options"]["embeddings_trainable"] = True
    model_configuration["decoder_options"]["embeddings_type"] = None
    model_configuration["decoder_options"]["embeddings_trainable"] = True
    model = get_custom_trained_pipeline("nl", "en", "exp_emb_learned", 0.01, **model_configuration)
    results["learned"] = evaluate_pipeline(model, test_from_texts, test_to_texts, **genopts)
    # Train for a few more epochs...
    model = get_custom_trained_pipeline("nl", "en", "exp_emb_learned_3x", 0.01, epochs=3, **model_configuration)
    results["learned"] = evaluate_pipeline(model, test_from_texts, test_to_texts, **genopts)
    save_results("experiment_embeddings", results)


def experiment_tokenizer():
    results = {}
    _, _, test_from_texts, test_to_texts = get_correspondences("nl", "en", training_split=0.8, sampling_ratio=0.01)
    genopts = {"num_beams": 2, "max_length": 10}
    model = get_custom_trained_pipeline("nl", "en", "exp_base", 0.01, **t.PIPELINE_PRESETS["base"])
    results["word_level"] = evaluate_pipeline(model, test_from_texts, test_to_texts, **genopts)
    # English words are 5 characters long on average;
    # as such we need to drastically increase the maximum length for a character-based model.
    genopts = {"num_beams": 2, "max_length": 50}
    model = get_custom_trained_pipeline("nl", "en", "exp_byte_base", 0.01, **t.PIPELINE_PRESETS["byte_base"])
    results["character_level"] = evaluate_pipeline(model, test_from_texts, test_to_texts, **genopts)
    save_results("experiment_tokenizer", results)


def experiment_state_size():
    pass


def experiment_attention():
    results = {}
    _, _, test_from_texts, test_to_texts = get_correspondences("nl", "en", training_split=0.8, sampling_ratio=0.01)
    genopts = {"num_beams": 2, "max_length": 16}
    model = get_custom_trained_pipeline("nl", "en", "exp_base", 0.01, **t.PIPELINE_PRESETS["base"])
    results["no_attention"] = evaluate_pipeline(model, test_from_texts, test_to_texts, **genopts)
    model = get_custom_trained_pipeline("nl", "en", "exp_base_attn", 0.01, **t.PIPELINE_PRESETS["base_attn"])
    results["with_attention"] = evaluate_pipeline(model, test_from_texts, test_to_texts, **genopts)
    save_results("experiment_attention", results)


def experiment_both_directions():
    # Reduce amount of test data by setting a higher training-split.
    # We need to massively reduce the amount of test data,
    # as generating the translation for a single example takes over a second.
    results = {}
    _, _, test_from_texts, test_to_texts = get_correspondences("nl", "en", training_split=0.98,
                                                               sampling_ratio=DEFAULT_SAMPLING_RATIO)
    genopts = {"num_beams": 2, "max_length": 32}
    model = get_trained_pipeline("nl", "en", "base_attn")
    results["nl_to_en"] = evaluate_pipeline(model, test_from_texts, test_to_texts, **genopts)
    _, _, test_from_texts, test_to_texts = get_correspondences("en", "nl", training_split=0.98,
                                                               sampling_ratio=DEFAULT_SAMPLING_RATIO)
    genopts = {"num_beams": 2, "max_length": 32}
    model = get_trained_pipeline("en", "nl", "base_attn")
    results["en_to_nl"] = evaluate_pipeline(model, test_from_texts, test_to_texts, **genopts)
    save_results("experiment_both_directions", results)


def experiment_pivot():
    pass


def experiment_plot_attention():
    pass

