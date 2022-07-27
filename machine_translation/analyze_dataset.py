import re
from collections import Counter
from pathlib import Path

import dataset_loader as loader

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(context="paper", style="white", font_scale=1.5, font="serif")

CONTEMPORARY_YEAR_PATTERN = re.compile(r"\b[1-2]\d{3}\b")

LANGUAGES = ["NL", "SV", "DA", "DE", "RO"]

IMAGES_PATH = Path(__file__).parent / "images"

if __name__ == "__main__":
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    print("Ratio of correspondences affected by fixable alignment issues:")
    for lang in LANGUAGES:
        raw_length = sum(1 for _ in loader.iterate_dataset(lang))
        merged_length = sum(1 for _ in loader.merge_empty_pairs(loader.iterate_dataset(lang)))
        print(f"{lang}: {(raw_length - merged_length) / raw_length:.3%} ({raw_length - merged_length} pairs)")

    k = 25
    years = {}
    for lang in LANGUAGES:
        years[lang] = Counter()
        # years[f"EN ({lang})"] = Counter()

        for lang_line, en_line in loader.iterate_dataset(lang):
            years[lang].update(CONTEMPORARY_YEAR_PATTERN.findall(lang_line))
            # years[f"EN ({lang})"].update(CONTEMPORARY_YEAR_PATTERN.findall(en_line))
    year_dfs = []
    for lang, year_counter in years.items():
        df = pd.DataFrame.from_records([(int(year), lang, value) for year, value in year_counter.most_common(n=k)],
                                       columns=["year", "language", "frequency"])
        year_dfs.append(df)
    years_df = pd.concat(year_dfs)
    ax = sns.barplot(data=years_df, x="year", y="frequency", hue="language")
    ax.set_title(f"top-{k} years present in the correspondences")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    sns.move_legend(ax, "upper right")
    plt.savefig(IMAGES_PATH / "years_by_language.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
