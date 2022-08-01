# Machine Translation

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
