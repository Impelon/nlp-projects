# Python's xml-libraries are littered with red warnings about security vulnerabilities.
# As we are constructing a primitive parser and don't need to interpret complex XML entities,
# a simple HTML parser ought to do the job.
import html.parser
import collections
from pathlib import Path

class TMXParser(html.parser.HTMLParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translation_units = collections.deque()
        self.reset_current_unit()

    def reset_current_unit(self):
        self._current_unit = {}
        self.reset_current_variant()

    def reset_current_variant(self, language=None):
        self._current_language = language
        self.reset_current_segment()

    def reset_current_segment(self):
        self._current_segment = None

    def finalize_current_unit(self):
        if self._current_unit:
            self.translation_units.append(self._current_unit)
        self.reset_current_unit()

    def finalize_current_variant(self):
        if self._current_language and self._current_segment:
            self._current_unit[self._current_language] = "".join(self._current_segment)
        self.reset_current_variant()

    def start_current_segment(self):
        self._current_segment = []

    def handle_starttag(self, tag, attrs):
        if tag == "tu":
            self.reset_current_unit()
        elif tag == "tuv":
            for name, value in attrs:
                if name == "xml:lang":
                    self.reset_current_variant(value.lower())
        elif tag == "seg":
            self.start_current_segment()

    def handle_endtag(self, tag):
        if tag == "tu":
            self.finalize_current_unit()
        elif tag in ("tuv", "seg"):
            self.finalize_current_variant()

    def handle_data(self, data):
        if self._current_segment is not None:
            self._current_segment.append(data)

def save_translation_units(units):
    while units:
        unit = units.popleft()
        name = "-".join(sorted(unit.keys(), key=lambda x: chr(255) if x == "en" else x))
        directory = Path(name)
        directory.mkdir(parents=True, exist_ok=True)
        for language, text in unit.items():
            with (directory / f"converted.{name}.{language}").open("a") as f:
                f.write(text + "\n")



if __name__ == "__main__":
    import sys

    parser = TMXParser()
    for line in sys.stdin.readlines():
        parser.feed(line)
        save_translation_units(parser.translation_units)
    save_translation_units(parser.translation_units)
