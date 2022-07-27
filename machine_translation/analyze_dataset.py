import re
import itertools
from collections import Counter
from pathlib import Path

import preprocessing as pre
import dataset_loader as loader

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(rc={"figure.figsize": (10, 8)})
sns.set_theme(context="paper", style="white", font_scale=1.5, font="serif")

CONTEMPORARY_YEAR_PATTERN = re.compile(r"\b[1-2]\d{3}\b")

LANGUAGES = ["NL", "SV", "DA", "DE", "RO"]

IMAGES_PATH = Path(__file__).parent / "images"


def count_characters(dataset, start=128):
    return Counter(char for entry in dataset for char in entry[0] + entry[1] if ord(char) >= start)


def show_pattern_by_language():
    k = 25
    years = {}
    language_codes = {}
    for lang in LANGUAGES:
        years[lang] = Counter()
        # years[f"EN ({lang})"] = Counter()
        language_codes[lang] = Counter()
        language_codes[f"EN ({lang})"] = Counter()

        for lang_line, en_line in loader.iterate_dataset(lang):
            years[lang].update(CONTEMPORARY_YEAR_PATTERN.findall(lang_line))
            # years[f"EN ({lang})"].update(CONTEMPORARY_YEAR_PATTERN.findall(en_line))
            language_codes[lang].update(pre.LANGUAGE_CODE_PATTERN.findall(lang_line))
            language_codes[f"EN ({lang})"].update(pre.LANGUAGE_CODE_PATTERN.findall(en_line))
    year_dfs = []
    language_code_dfs = []
    for lang, year_counter in years.items():
        df = pd.DataFrame.from_records([(int(year), lang, value) for year, value in year_counter.most_common(n=k)],
                                       columns=["year", "language", "frequency"])
        year_dfs.append(df)
    years_df = pd.concat(year_dfs)
    for lang, code_counter in language_codes.items():
        df = pd.DataFrame.from_records([(code, lang, value) for code, value in code_counter.items()],
                                       columns=["code", "language", "frequency"])
        language_code_dfs.append(df)
    language_codes_df = pd.concat(language_code_dfs)
    ax = sns.barplot(data=years_df, x="year", y="frequency", hue="language")
    ax.set_title(f"top-{k} years present in the correspondences")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    sns.move_legend(ax, "upper right")
    plt.savefig(IMAGES_PATH / "years_by_language.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    ax = sns.barplot(data=language_codes_df, x="code", y="frequency", hue="language",
                     hue_order=sorted(language_codes_df["language"].unique(),
                                      key=lambda x: "ZZZ" + x if x.startswith("EN") else x))
    ax.set_title(f"language codes present in the correspondences")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    sns.move_legend(ax, "upper right")
    plt.savefig(IMAGES_PATH / "language_codes_by_language.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def show_fixed_alignments():
    print("Ratio of correspondences affected by fixable alignment issues:")
    for lang in LANGUAGES:
        raw_length = sum(1 for _ in loader.iterate_dataset(lang))
        merged_length = sum(1 for _ in loader.merge_empty_pairs(loader.iterate_dataset(lang)))
        print(f"{lang}: {(raw_length - merged_length) / raw_length:.3%} ({raw_length - merged_length} pairs)")


def explore_interactively():
    """
    It's quite hard to explore the datasets with an editor,
    as the files are very large, and to examine a pair of correspondences,
    both files need to be opened at corresponding positions.

    Instead, we built this interactive shell for exploring the data, mimicking the python interpreter.
    """
    ilocals = {}
    iglobals = {
        "languages": LANGUAGES,
        "p": lambda dataset: list(pre.preprocess_dataset(dataset)),
        "l": lambda lang: loader.load_dataset(lang, shuffle=False),
        "lraw": lambda lang: list(loader.iterate_dataset(lang)),
        "la": lambda ratio=0.1: list(itertools.chain.from_iterable(
            loader.load_dataset(lang, sampling_ratio=ratio, shuffle=False) for lang in LANGUAGES)),
        "anyf": lambda func, iterable:
            list(filter(lambda x: func(x[1][0]) or func(x[1][1]), enumerate(iterable))),
        "anyfr": lambda pattern, iterable:
            list(filter(lambda x: re.search(pattern, x[1][0]) or re.search(pattern, x[1][1]), enumerate(iterable))),
        "allf": lambda func, iterable:
            list(filter(lambda x: func(x[1][0]) and func(x[1][1]), enumerate(iterable))),
        "allfr": lambda pattern, iterable:
            list(filter(lambda x: re.search(pattern, x[1][0]) and re.search(pattern, x[1][1]), enumerate(iterable))),
        "show": lambda i, indexed, context=0:
            "\n".join(map(lambda x: str(x[1]), indexed[i-context:i+context+1])),
        "showby": lambda sequence, i, indexed, context=1:
            "\n".join(map(str, sequence[indexed[i][0]-context:indexed[i][0]+context+1])),
        "count_characters": count_characters,
    }
    print("Predefined globals:", list(iglobals.keys()))
    while True:
        lines = []
        try:
            while not lines or lines[-1].endswith(":") or lines[-1].startswith((" ", "\t")):
                lines.append(input("    " if lines else ">>> "))
        except KeyboardInterrupt as ex:
            print(ex)
        except EOFError:
            break
        command = "\n".join(lines)
        if command:
            if len(lines) == 1 and not lines[0].startswith(("del ", "from ", "import ")):
                if not re.match(r"\w+\s*=\s*", command):
                    command += "\nif _ is not None:\n\tprint(_)"
                command = "_ = " + command
            try:
                exec(command, iglobals, ilocals)
                iglobals.update(ilocals)
            except BaseException as ex:
                print(ex)
    print()


if __name__ == "__main__":
    functions = list(filter(lambda x: callable(x[1]) and x[0].startswith("show_"), globals().items()))


    def execute_all():
        for _, f in functions:
            f()


    options = [("explore_interactively", explore_interactively), ("all", execute_all)] + functions

    print("Choose an option to perform. To execute all predefined analysis steps just choose 'all'.")
    print("\n".join(f"{i}. {name}" for i, (name, _) in enumerate(options)))
    while True:
        index = input("Choose index: ")
        if index.isdigit():
            index = int(index)
            if 0 <= index < len(options):
                break
        print("Invalid index!")

    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    options[index][1]()
