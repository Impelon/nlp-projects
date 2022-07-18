# Machine Translation

## Code Organization

- `analyze_dataset.py`
  - Contains our code to extract insights from the dataset.
- `translation.py`
  - Contains our code to create models for translation between languages.
  - Creates complete pipelines, including tokenizer, embedding, encoder and decoder.
- `experiments.py`
  - Contains all our experiments using different models, approaches and languages.
- `preprocessing.py`
  - Contains our code to preprocess pairs of correspondences.
- `dataset_loader.py`
  - Contains our code to load a translation-dataset.
  - Optionally performs data sampling and shuffles the data.
  - Performs some preprocessing concerning the alignment of correspondences (e.g. merging empty translations).