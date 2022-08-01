from functools import lru_cache

import translation as t
import dataset_loader

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# Configure a seed so all experiments are reproducible.
keras.utils.set_random_seed(42)

SAMPLING_RATIO = 0.0001


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
    from_texts = tf.convert_to_tensor(list(map(lambda x: x[1], dataset)))
    to_texts = tf.convert_to_tensor(list(map(lambda x: x[0], dataset)))

    split_index = int(len(from_texts) * training_split)
    train_from_texts, test_from_texts = from_texts[:split_index], from_texts[split_index:]
    train_to_texts, test_to_texts = to_texts[:split_index], to_texts[split_index:]
    return train_from_texts, train_to_texts, test_from_texts, test_to_texts


def get_trained_pipeline(from_language, to_language, pipeline_preset_name):
    model_path = t.MODELS_PATH / f"{from_language}_{to_language}" / pipeline_preset_name
    model = t.translation_pipeline(from_language, to_language, **t.PIPELINE_PRESETS[pipeline_preset_name])
    try:
        model.load_weights(str(model_path))
        return model
    except Exception:
        pass
    from_texts, to_texts, _, _ = get_correspondences(from_language, to_language, sampling_ratio=SAMPLING_RATIO)
    t.train_pipeline(model, from_texts, to_texts)
    model.save_weights(str(model_path))
    return model


# Experiments


def experiment_state_size():
    pass


def experiment_tokenizer():
    pass


def experiment_embeddings():
    pass


def experiment_attention():
    pass


def experiment_pivot():
    pass


def experiment_plot_attention():
    pass

