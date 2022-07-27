import random
import warnings
from pathlib import Path
from collections import deque

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

DATA_PATH = Path(__file__).parent / "data"


def merge_pairs(*pairs):
    """
    Merge all pairs and return the result.
    """
    firsts, seconds = zip(*pairs)
    return ("".join(firsts), "".join(seconds))


def merge_empty_pairs(pairs):
    """
    Yield pairs from the given iterable, merging empty pairs with surrounding ones.

    Sometimes correspondences will be misaligned: If either element of the pair is empty,
    its translation is usually found in the previous or subsequent pair.
    By merging such pairs we can solve this problem.

    However, we cannot know if the sentence belongs to
    the previous pair or the subsequent one, therefore we merge all 3.
    This way we are guaranteed to be correct, at the expense of generating some longer correspondences.
    """
    previous = deque(maxlen=2)
    merge_next = False
    for current in pairs:
        if len(current[0]) == 0 and len(current[1]) == 0:
            # Skip completely empty pairs. They do not belong anywhere
            # and should not trigger a merge of surrounding pairs.
            continue

        if merge_next:
            merged = merge_pairs(*previous, current)
            previous.clear()
            previous.append(merged)
            merge_next = False
        else:
            if len(previous) == previous.maxlen:
                yield previous.popleft()
            previous.append(current)

        if len(current[0]) == 0 or len(current[1]) == 0:
            merge_next = True

    if merge_next:
        yield merge_pairs(*previous)
    else:
        yield from previous


def load_dataset(language, sampling_ratio=1, merge_empty=True, shuffle=True, data_path=None, seed=42):
    random.seed(seed)  # Configure a seed so the data-sampling is reproducible.
    dataset = iterate_dataset(language, data_path=data_path)
    if merge_empty:
        dataset = merge_empty_pairs(dataset)
    dataset = list(tqdm(dataset, desc=f"Loading dataset for {language}"))
    sample_length = int(len(dataset) * sampling_ratio)
    if shuffle:
        return random.sample(dataset, k=sample_length)
    del dataset[sample_length:]
    return dataset


def iterate_dataset(language, data_path=None):
    if not data_path:
        data_path = DATA_PATH

    lang = language.lower()
    lang_paths = list(data_path.rglob(f"europarl-v7.{lang}-en.{lang}"))
    en_paths = list(data_path.rglob(f"europarl-v7.{lang}-en.en"))
    for paths in (lang_paths, en_paths):
        if not paths:
            raise LookupError(f"No dataset for this language was found at: {data_path}")
        elif len(paths) > 1:
            sep = "\n - "
            warnings.warn("Multiple datasets found for the specified language:" +
                          sep + sep.join(map(str, paths)) + "\n" +
                          "Using first entry!")

    with lang_paths[0].open("r") as lang_file:
        with en_paths[0].open("r") as en_file:
            for lang_line, en_line in zip(lang_file, en_file):
                yield lang_line.strip(), en_line.strip()
