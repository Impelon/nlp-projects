# Machine Translation

## About

This code and the associated report are a result of a three-week long assignment from the *Natural Language Processing* module.
The task was to implement a Seq2Seq model translation-pipeline (pretraining, tokenization, RNN encoder-decoder model, attention layer, evaluation) from scratch using the provided [Europarl dataset](https://www.statmt.org/europarl/).
By the end we were left with this modular pipeline where different parts of the models can be switched out between themselves. It is however a bit rough around the edges when it comes to user-friendlyness.

In the report we go into further details regarding our experiments and our results, however it should be noted that **the rigorousness of our experiments and quality of our writing were limited due to the tight time constraints**.
Furthermore, the report does not follow all aspects of a scientific paper (in structure and content), instead describing our work through the lens of the assignment.

All in all, we were able to achieve passable translations for very short sequences consisting of only a few words, but we did not expect much better quality given that training had to be done on our own personal machines. This limited the training data and model sizes and the extent of our evaluation.

## Code Organization

- `analyze_dataset.py`
  - Contains our code to extract insights from the dataset.
- `translation.py`
  - Contains our code to create models for translation between languages.
  - Creates complete pipelines, including tokenizer, embeddings, encoder and decoder.
    - Functions building the models are near the end of the file.
    - Available presets for model configurations are available near the top of the file.
    - Can create pipelines using English as a pivot-language.
  - Provides methods for training pipelines.
- `experiments.py`
  - Contains all our experiments using different models, approaches and languages.
  - Includes code for evaluation using BLEU and METEOR.
  - Automatically saves/loads/trains models and stores evaluation results.
- `preprocessing.py`
  - Contains our code to preprocess pairs of correspondences.
- `tokenizers.py`
  - Contains some extensions around Keras' tokenizers so we can employ them more easily.
- `embeddings.py`
  - Contains some extensions around Gensim's vectorization models so we can employ them more easily.
- `dataset_loader.py`
  - Contains our code to load a translation-dataset.
  - Optionally performs data sampling and shuffles the data.
  - Performs some preprocessing concerning the alignment of correspondences (e.g. merging empty translations).
- `unittests.py`
  - Includes unit tests for some of our functions.
