import re
import unicodedata

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

# In decreasing frequency across the studied datasets:
VALID_LANGUAGE_CODES = {"DE", "EN", "FR", "PL", "PT", "IT", "NL", "ES", "EL", "RO", "HU",
                        "SV", "SK", "CS", "LT", "FI", "DA", "BG", "SL", "GA", "ET", "LV", "MT"}

PATTERNS_TO_REPLACE = {
    "language_code": (re.compile(r"\((" + r"|".join(VALID_LANGUAGE_CODES) + r")\)"), ""),
    "enumeration": (re.compile(r"^\d+.$"), ""),
    "trailing_punctuation": (re.compile(r"\s([^\w\s]*[.?!]+)$"), r"\1"),
    "trailing_or_leading_characters": (re.compile(r"^\W+\s|\s\W+$"), ""),
    "extra_whitespace": (re.compile(r"\s+"), " "),
}

# Map for character-level normalization.
CHARACTERS_TO_REPLACE = {
    "’": "'",
    "‘": "'",
    "ʻ": "'",
    "´": "'",
    "„": "\"",
    "‟": "\"",
    "”": "\"",
    "ˮ": "\"",
    "“": "\"",
    "ʺ": "\"",
    "″": "\"",
    "―": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    "\xad": "-",
    "…": "...",
    "‚": ",",
    "¸": ",",
    "⁄": "/",
    "º": "°",
    "˚": "°",
    "·": ";",
    "\u200b": "",
    "\u202a": "",
    "\u202b": "",
    "\u202c": "",
    "\u202d": "",
    "\u202e": "",
}
REPLACEMENT_TABLE = str.maketrans(CHARACTERS_TO_REPLACE)

# Approximate phonetic transliteration tables by modern pronunciation.
# We do not actually use these, as visual confusion is the main concern.
PHON_GREEK2LATIN_TABLE = str.maketrans("ΑαΒβΓγΔδΕεΖζΗηΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΩω",
                                       "AaVvGgDdEeZzIiIiKkLlMmNnXxOoPpRrSssTtIiOo")
PHON_GREEK2LATIN_TABLE.update(str.maketrans({
    "Θ": "Th",
    "θ": "th",
    "Φ": "Ph",
    "φ": "ph",
    "Χ": "Kh",
    "χ": "kh",
    "Ψ": "Ps",
    "ψ": "ps",
}))

PHON_CYRILLIC2LATIN_TABLE = str.maketrans("АаБбВвГгДдЕеЗзИиКкЛлМмНнОоПпРрСсТтУуФфХхЪъЫыЬьЭэ",
                                          "AaBbVvGgDdEeZzIiKkLlMmNnOoPpRrSsTtUuFfHh''Yy''Ee")
PHON_CYRILLIC2LATIN_TABLE.update(str.maketrans({
    "Ж": "Zh",
    "ж": "zh",
    "Ц": "Ts",
    "ц": "ts",
    "Я": "Ja",
    "я": "ja",
    "Ч": "Ch",
    "ч": "ch",
    "Ш": "Sh",
    "ш": "sh",
    "Щ": "Shch",
    "щ": "shch",
    "Ю": "Ju",
    "ю": "ju",
}))
# Transliteration tables by visual similarity.
GREEK2LATIN_TABLE = PHON_GREEK2LATIN_TABLE.copy()
GREEK2LATIN_TABLE.update(str.maketrans("ΒβΥυΡρΗηΧχ", "BßYuPpHnXx"))
CYRILLIC2LATIN_TABLE = PHON_CYRILLIC2LATIN_TABLE.copy()
CYRILLIC2LATIN_TABLE.update(str.maketrans("ВвНнРрСсУуХх", "BBHHPpCcYyXx"))
FOREIGN2LATIN_TABLE = GREEK2LATIN_TABLE.copy()
FOREIGN2LATIN_TABLE.update(CYRILLIC2LATIN_TABLE)


def calculate_foreign_symbol_rate(text):
    if not text:
        return 0
    decomposed = unicodedata.normalize("NFD", text)
    foreign_ordinals = filter(FOREIGN2LATIN_TABLE.__contains__, map(ord, decomposed))
    return len(tuple(foreign_ordinals)) / len(decomposed)


def transliterate_foreign_symbols_to_latin(text):
    # Decompose accents into separate characters.
    decomposed = unicodedata.normalize("NFD", text)
    transliterated = decomposed.translate(FOREIGN2LATIN_TABLE)
    # Recompose accented characters.
    recomposed = unicodedata.normalize("NFC", transliterated)
    return recomposed


def preprocess_text(text, replace_characters=None):
    # Unicode normalization.
    # (Not needed, as apparently the dataset is already normalized.)
    # text = unicodedata.normalize("NFC", text)

    # Custom character-level normalization.
    if replace_characters is None or replace_characters:
        text = text.translate(REPLACEMENT_TABLE)
    # Remove markup and fix whitespace.
    for pattern, replacement in PATTERNS_TO_REPLACE.values():
        text = pattern.sub(replacement, text)
    return text.strip()


def preprocess_correspondence(pair, replace_characters=None, transliterate_foreign=True, foreign_ratios=None):
    # While there are some legitimate phrases using the greek or cyrillic alphabet,
    # there are many examples where characters are used from these alphabets
    # instead of identical looking ones from the latin alphabet.
    # Therefore, we replace greek or cyrillic characters when they do not show up on both sides of a correspondence.
    if transliterate_foreign:
        if not foreign_ratios:
            foreign_ratios = (calculate_foreign_symbol_rate(pair[0]), calculate_foreign_symbol_rate(pair[1]))
        if foreign_ratios[0] > 0 and foreign_ratios[1] == 0:
            pair = (transliterate_foreign_symbols_to_latin(pair[0]), pair[1])
        elif foreign_ratios[1] > 0 and foreign_ratios[0] == 0:
            pair = (pair[0], transliterate_foreign_symbols_to_latin(pair[1]))
    # Do regular preprocessing.
    return (preprocess_text(pair[0], replace_characters=replace_characters),
            preprocess_text(pair[1], replace_characters=replace_characters))


def preprocess_dataset(dataset, replace_characters=None, transliterate_foreign=None, max_foreign_ratio=0.15):
    if transliterate_foreign is None:
        transliterate_foreign = True
    for pair in tqdm(dataset, "Preprocessing dataset"):
        foreign_ratios = None
        if transliterate_foreign:
            foreign_ratios = (calculate_foreign_symbol_rate(pair[0]), calculate_foreign_symbol_rate(pair[1]))
            if sum(foreign_ratios) > max_foreign_ratio and not all(foreign_ratios):
                # Skip correspondences where foreign symbols are only included one language
                # and represent a major part of the sentence:
                # That indicates we are dealing with a sentence written in a foreign language,
                # which should not be part of the dataset.
                continue

        # Do regular preprocessing.
        pair = preprocess_correspondence(pair, replace_characters=replace_characters,
                                         transliterate_foreign=transliterate_foreign, foreign_ratios=foreign_ratios)
        if len(pair[0]) > 0 and len(pair[1]) > 0:
            # Ignore any new/remaining malformed correspondences.
            yield pair
