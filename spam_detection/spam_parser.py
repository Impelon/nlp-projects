from pathlib import Path

DEFAULT_LOCATION = Path(__file__).parent /  "data" / "SMSSpamCollection"

def parse_from_path(path):
    with open(path, "r") as file:
        return parse_lines(file)

def parse_lines(lines):
    spam = []
    ham = []
    for line in lines:
        category, sms = line.split("\t", 1)
        if category == "spam":
            spam.append(sms)
        elif category == "ham":
            ham.append(sms)
        else:
            raise RuntimeError("Undefined sms category: " + category)
    return spam, ham
