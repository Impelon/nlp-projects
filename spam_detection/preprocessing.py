import html
import re

POBOX_PATTERN = re.compile(r"(po ?box\w*( \d+)?)|(box ?(?=\w*\d)\w+)", re.IGNORECASE)
TELEPHONE_PATTERN = re.compile(r"(\+?[\d.*]{5,}\d)|(\b\d{5})")
URL_PATTERN = re.compile(r"(?=\S*[a-zA-z]{2})(https?://)?([\w-]+\.){2,}[\w+-]+(/\S+)*/?")

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

def preprocess(text):
    text = html.unescape(text)
    for to_be_replaced, replacement in CHARACTERS_TO_REPLACE:
        text = text.replace(to_be_replaced, replacement)
    # TODO use patterns from above
    return text