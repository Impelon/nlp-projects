import html
import re

POBOX_PATTERN = re.compile(r"(po ?box\w*( \d+)?)|(box ?(?=\w*\d)\w+)", re.IGNORECASE)
TELEPHONE_PATTERN = re.compile(r"(\+?[\d.*]{5,}\d)|(\b\d{5})")
URL_PATTERN = re.compile(r"(?=\S*[a-zA-z]{2})(https?://)?([\w-]+\.){2,}[\w+-]+(/\S+)*/?")
MONEY_PATTERN = re.compile(r"[£$]\d+([.,]\d+)?")

# This list of stopwords was manually selected by us from inspecting the most common 200 tokens in the dataset
# and removing semantically significant words from the list.
SMS_STOP_WORDS = {"i", "a", "y", "u", "s", "2", "t", "m", "4", "r", "n", "d", "k", "e", "b", "to", "you", "the", "and",
                  "in", "is", "me", "my", "it", "for", "your", "of", "that", "have", "on", "now", "are", "can", "so",
                  "but", "not", "or", "do", "we", "at", "ur", "get", "will", "if", "be", "with", "just", "no", "this",
                  "how", "up", "when", "ok", "what", "from", "all", "out", "ll", "then", "got", "there", "was", "he",
                  "its", "am", "only", "as", "one", "by", "going", "she", "about", "lor", "da", "our", "hi", "tell",
                  "they", "please", "any", "pls", "her", "did", "been", "dear", "who", "well", "where", "re", "has",
                  "much", "oh", "an", "hey", "him", "more", "too", "wat", "had", "yes", "way", "ve", "should", "say",
                  "right", "already", "ask", "said", "doing", "yeah", "really", "im", "why", "them", "very", "let",
                  "would", "cos", "also", "sure", "over", "us", "first", "his", "were", "which"}

CHARACTERS_TO_REPLACE = {
    "…": "...",
    "–": "-",
    "’": "'",
    "“": "\"",
    "\x92": "'",
    "\x94": " ",
    "\x93": "",
    "ü": "u",
    "Ü": "U",
}

# Unused.
SLANG_TO_REPLACE = {
    re.compile("\bu\b"): "you",
    re.compile("'s"): "is",
    re.compile("\b2\b"): "to",
    re.compile("n't"): "not",
    re.compile("'m"): "am",
    re.compile("\b4\b"): "for",
    re.compile("\br\b"): "are",
    re.compile("\bn\b"): "and",
    re.compile("'d"): "would",
    re.compile("\bd\b"): "the",
    re.compile("\bk\b"): "ok",
    re.compile("\be\b"): "the",
    re.compile("\bb\b"): "be",
}


def preprocess(text):
    text = html.unescape(text)
    for to_be_replaced, replacement in CHARACTERS_TO_REPLACE.items():
        text = text.replace(to_be_replaced, replacement)

    text = POBOX_PATTERN.sub(" POBox ", text)
    text = TELEPHONE_PATTERN.sub(" phonenumber ", text)
    text = URL_PATTERN.sub(" URL ", text)
    text = MONEY_PATTERN.sub(" sumofmoney ", text)

    return text


# Same as the pattern used by sklearn.feature_extraction.text
WORD_PATTERN = re.compile(r"\b\w\w+\b")


def tokenize(text):
    return WORD_PATTERN.findall(text)
