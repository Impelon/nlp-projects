import html
import re

POBOX_PATTERN = re.compile(r"(po ?box\w*( \d+)?)|(box ?(?=\w*\d)\w+)", re.IGNORECASE)
TELEPHONE_PATTERN = re.compile(r"(\+?[\d.*]{5,}\d)|(\b\d{5})")
URL_PATTERN = re.compile(r"(?=\S*[a-zA-z]{2})(https?://)?([\w-]+\.){2,}[\w+-]+(/\S+)*/?")

# TODO tokenization?
# - urls
# - money

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

def preprocess(text):
    text = html.unescape(text)
    for to_be_replaced, replacement in CHARACTERS_TO_REPLACE:
        text = text.replace(to_be_replaced, replacement)
    # TODO use patterns from above
    return text